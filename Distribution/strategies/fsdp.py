#!/usr/bin/env python3
# quick FSDP benchmark: shards the model, runs inference and saves a video

"""Example:
  torchrun --nproc_per_node=2 fsdp.py \
      --model_id cerspense/zeroscope_v2_XL \
      --prompt "a rocket flying off into space, 4k, detailed" \
      --steps 50 --num_frames 16
"""

from tqdm import tqdm
import os, time, argparse, logging, numpy as np
import torch
import torch.distributed as dist
from torch.nn import ModuleList, Sequential, ModuleDict
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    CPUOffload,
    MixedPrecision,
)
from torch.distributed.fsdp.fully_sharded_data_parallel import ShardingStrategy
from diffusers import DiffusionPipeline
from transformers import CLIPModel, CLIPProcessor
import pynvml, cv2
import csv, datetime, socket, random
from huggingface_hub import HfFolder

# ---- logging ----
logging.basicConfig(level=logging.WARNING,
                    format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("FSDPBench")
logger.info(f"Torch Version: {torch.__version__}")

# ---- helpers ----
local_rank = int(os.environ.get("LOCAL_RANK", 0))
torch.cuda.set_device(local_rank)
pynvml.nvmlInit()

def vram_mb():
    h = pynvml.nvmlDeviceGetHandleByIndex(local_rank)
    return pynvml.nvmlDeviceGetMemoryInfo(h).used // 1024**2

class FSDPBenchmark:
    def __init__(self, args):
        self.args = args
        dist.init_process_group("nccl")
        self.rank = dist.get_rank(); self.ws = dist.get_world_size()

        
        max_retries = 3
        retry_delay = 5  # seconds
        
        for attempt in range(max_retries):
            try:
                pipe = DiffusionPipeline.from_pretrained(
                    args.model_id,
                    torch_dtype=torch.float16,
                    low_cpu_mem_usage=True,
                    device_map=None,
                    use_safetensors=False,
                    local_files_only=attempt > 0  # Try local cache after first attempt
                )
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"Attempt {attempt + 1} failed: {str(e)}. Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    raise

        # FSDP config
        mp = MixedPrecision(param_dtype=torch.float16,
                            reduce_dtype=torch.float16,
                            buffer_dtype=torch.float16)
        def auto_wrap_policy(module, recurse, nonwrapped_numel):
            if isinstance(module, (ModuleList, Sequential, ModuleDict)):
                return False
            return nonwrapped_numel >= 10_000_000

        fsdp_kw = dict(
            auto_wrap_policy=auto_wrap_policy,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            cpu_offload=CPUOffload(offload_params=True),
            mixed_precision=mp,
            device_id=torch.cuda.current_device(),
            use_orig_params=False
        )

        # Wrap UNet and text encoder
        self.unet = FSDP(pipe.unet, **fsdp_kw)
        self.text_encoder = FSDP(pipe.text_encoder, **fsdp_kw)

     
        fsdp_vae_kw = {**fsdp_kw, "cpu_offload": CPUOffload(offload_params=False)}
        vae = pipe.vae
        for name, submod in vae.named_children():
            if any(p.requires_grad for p in submod.parameters()):
                setattr(vae, name, FSDP(submod, **fsdp_vae_kw))
        self.vae = vae

        pipe.unet = self.unet
        pipe.text_encoder = self.text_encoder
        pipe.vae = self.vae
        self.pipe = pipe

        

    def run(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        static_vram = vram_mb()

        # Prepare dual prompts for classifier-free guidance
        toks = self.pipe.tokenizer(
            [self.args.prompt, ""],
            padding="max_length",
            max_length=self.pipe.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        )
        input_ids = toks.input_ids.to(self.args.device)  # [2, L]
        with torch.no_grad():
            emb_all = self.text_encoder(input_ids)[0]      # [2, D]
        cond_emb, uncond_emb = emb_all[0:1], emb_all[1:2]

        # Scheduler setup
        self.pipe.scheduler.set_timesteps(self.args.steps,
                                          device=self.args.device)

        # Initialise noisy latents
        h = self.args.height // 8
        w = self.args.width // 8
        lat = torch.randn(
            1, self.unet.config.in_channels,
            self.args.num_frames, h, w,
            device=self.args.device, dtype=torch.float16
        ) * self.pipe.scheduler.init_noise_sigma

        # Denoising loop with guidance
        timesteps = self.pipe.scheduler.timesteps
        for t in tqdm(timesteps, desc=f"[Rank {self.rank}] denoising",
                      disable=(self.rank != 0), leave=False):
            # duplicate latents for uncond + cond
            x = torch.cat([lat, lat], dim=0)
            emb = torch.cat([uncond_emb, cond_emb], dim=0)

            with torch.no_grad():
                noise_pred = self.unet(x, t,
                                       encoder_hidden_states=emb).sample
            uncond_pred, cond_pred = noise_pred.chunk(2, dim=0)
            guided = uncond_pred + 7.5 * (cond_pred - uncond_pred)

            lat = self.pipe.scheduler.step(guided, t, lat).prev_sample

        peak_local = torch.cuda.max_memory_allocated() // 1024 ** 2
        peak_all = torch.tensor(peak_local, device=self.args.device)

        # emulate network latency before reduce
        if self.args.emu_rtt_ms > 0:
            time.sleep(self.args.emu_rtt_ms / 1000.0)
        t0_red = time.time()
        dist.all_reduce(peak_all, op=dist.ReduceOp.MAX)
        net_reduce_s = time.time() - t0_red

        # Decode latents one frame at a time
        lat_cpu = lat.cpu().float()  # [1,4,T,h,w]
        frames = []
        for idx in range(lat_cpu.shape[2]):
            # Keep temporal dimension for decode_latents: [1,4,1,h,w]
            single_lat = lat_cpu[:, :, idx:idx+1, :, :]
            with torch.no_grad():
                img = self.pipe.decode_latents(single_lat)[0]   # tensor (3,1,H,W)
            
            # Squeeze the temporal dim & put channels last
            img = img[:, 0, :, :]               # ->  (3,H,W)
            img = img.permute(1, 2, 0)          # ->  (H,W,3)
            img = (img.clamp(0, 1) * 255).byte().cpu().numpy()
            
            frames.append(img)

        # Write video on rank 0 only
        if self.rank == 0:
            out_file = "output.mp4"
            fps = 8
            height_px, width_px = self.args.height, self.args.width
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(out_file, fourcc,
                                    fps, (width_px, height_px))
            logger.warning(f"[Rank {self.rank}] Writing {len(frames)} frames")
            logger.warning(f"[Rank {self.rank}] Video settings: {width_px}x{height_px} @ {fps}fps")

            for frame in frames:
                writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

            writer.release()
            logger.info(f"[Rank {self.rank}] saved video ->  {out_file}")
            # Verify file was written
            if os.path.exists(out_file):
                size = os.path.getsize(out_file)
                logger.info(f"[Rank {self.rank}] Output file size: {size} bytes")
            else:
                logger.error(f"[Rank {self.rank}] Output file was not created!")

        # synchronise before measuring end VRAM
        dist.barrier()

        # ---- end VRAM across ranks ----
        end_local = vram_mb()
        end_all = torch.tensor(end_local, device=self.args.device)
        dist.all_reduce(end_all, op=dist.ReduceOp.MAX)

        return dict(
            world_size=self.ws,
            chunk_size=0,
            overlap=0,
            peak_vram_mb=int(peak_all.item()),
            end_vram_mb=int(end_all.item()),
            network_bytes=0,
            net_gather_s=0,
            net_reduce_s=net_reduce_s,
            temp_instab="",
            flow_err="",
        )

def one_run(a):
    bench = FSDPBenchmark(a)
    t0 = time.time()
    res = bench.run()
    elapsed = time.time() - t0

    if dist.get_rank() == 0:
        res.update({
            "timestamp": datetime.datetime.utcnow().isoformat(timespec="seconds"),
            "host": socket.gethostname(),
            "mode": "fsdp",
            "num_frames": a.num_frames,
            "latency_s": elapsed,
            "throughput_fps": round(a.num_frames / elapsed, 3),
        })

    
        res.setdefault("network_bytes",0)
        res.setdefault("net_gather_s",0)
        res.setdefault("net_reduce_s",res.get("net_reduce_s",0))
        res.setdefault("temp_instab","")
        res.setdefault("flow_err","")

        header = [
            "timestamp","host","mode","world_size","num_frames",
            "chunk_size","overlap","latency_s","throughput_fps",
            "peak_vram_mb","end_vram_mb",
            "network_bytes","net_gather_s","net_reduce_s",
            "temp_instab","flow_err"
        ]

        write_hdr = not os.path.exists(a.out_csv)
        with open(a.out_csv, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=header)
            if write_hdr:
                w.writeheader()
            w.writerow({k: res.get(k, "") for k in header})
        logger.info(f"Metrics appended ->  {a.out_csv}")

    if dist.is_initialized():
        dist.destroy_process_group()

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model_id",
                   default="cerspense/zeroscope_v2_XL")
    p.add_argument("--prompt",
                   default="a rocket flying off into space, hd, detailed")
    p.add_argument("--num_frames", type=int, default=25)
    p.add_argument("--steps",      type=int, default=50)
    p.add_argument("--device",     default="cuda")
    p.add_argument("--height",     type=int, default=576)
    p.add_argument("--width",      type=int, default=1024)
    p.add_argument("--out",        default="fsdp_results.npy")  # legacy
    p.add_argument("--out_csv",    default="benchmarks.csv")
    p.add_argument("--emu_bw_mbps", type=float, default=0,
                   help="throttle bandwidth in Mbps (unused for fsdp baseline)")
    p.add_argument("--emu_rtt_ms", type=float, default=0,
                   help="one-way latency in ms")
    p.add_argument("--emu_jitter_ms", type=float, default=0,
                   help="jitter stddev in ms")
    args = p.parse_args()

    one_run(args)
