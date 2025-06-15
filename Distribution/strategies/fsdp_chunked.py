#!/usr/bin/env python3
# FSDP + chunked latent generation


import os, time, argparse, logging, random
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import cv2, pynvml
from tqdm import tqdm
from torch.nn import ModuleList, Sequential, ModuleDict
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, CPUOffload, MixedPrecision
from torch.distributed.fsdp.fully_sharded_data_parallel import ShardingStrategy
from torch.distributed.fsdp.wrap import wrap
from diffusers import DiffusionPipeline
import csv, datetime, socket


local_rank = int(os.getenv("LOCAL_RANK", 0))

torch.cuda.set_device(local_rank)
device = torch.device(f"cuda:{local_rank}")

class _RankFilter(logging.Filter):
    def __init__(self, rank: int):
        super().__init__()
        self.rank = rank
    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "rank"):
            record.rank = self.rank
        return True

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [Rank %(rank)d] %(levelname)s: %(message)s")
logging.getLogger().addFilter(_RankFilter(local_rank))

LOG = logging.getLogger("FSDPSplit")

pynvml.nvmlInit()
def vram_mb():
    h = pynvml.nvmlDeviceGetHandleByIndex(local_rank)
    return pynvml.nvmlDeviceGetMemoryInfo(h).used // 1024**2

class DistributedVideoDiffuser:
    def __init__(self, cfg):
        self.cfg = cfg  
        dist.init_process_group("nccl", device_id=torch.device(f"cuda:{local_rank}"))
        self.rank  = dist.get_rank()
        self.world = dist.get_world_size()

        # Clear CUDA cache before loading model
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        LOG.info("Loading pipeline on CPU", extra={"rank":self.rank})
        pipe = DiffusionPipeline.from_pretrained(
            cfg.model_id,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            use_safetensors=False,
            device_map=None
        )

        #  FSDP config
        mp = MixedPrecision(torch.float16, torch.float16, torch.float16)
        def wrap_policy(module, recurse, nonwrapped_numel):
            if isinstance(module, (ModuleList, Sequential, ModuleDict)): return False
            return nonwrapped_numel >= 10_000_000
        fsdp_kwargs = dict(
            auto_wrap_policy = wrap_policy,
            sharding_strategy= ShardingStrategy.FULL_SHARD,
            cpu_offload      = CPUOffload(offload_params=True),
            mixed_precision  = mp,
            device_id        = local_rank,
            use_orig_params  = False
        )

        # shard the heavy modules
        self.unet         = FSDP(pipe.unet,         **fsdp_kwargs)
        self.text_encoder = FSDP(pipe.text_encoder, **fsdp_kwargs)

        # shard each VAE block, but leave outer wrapper intact
        vae = pipe.vae
        fsdp_kw_vae = dict(fsdp_kwargs, cpu_offload=CPUOffload(offload_params=True))
        for nm, sm in vae.named_children():
            if any(p.requires_grad for p in sm.parameters()):
                setattr(vae, nm, FSDP(sm, **fsdp_kw_vae))
        self.vae = vae 

        pipe.unet         = self.unet
        pipe.text_encoder = self.text_encoder
        pipe.vae          = self.vae
        self.pipe = pipe

        # scheduler & text embeddings
        pipe.scheduler.set_timesteps(cfg.steps, device=cfg.device)
        toks = pipe.tokenizer([cfg.prompt, ""],
                              padding="max_length",
                              max_length=pipe.tokenizer.model_max_length,
                              truncation=True,
                              return_tensors="pt")
        with torch.no_grad():
            emb = self.text_encoder(toks.input_ids.to(cfg.device))[0]
        self.cond_emb, self.uncond_emb = emb[:1], emb[1:]

    def _denoise(self, lat: torch.Tensor) -> torch.Tensor:
        LOG.info(f"Entering _denoise() with latent shape {lat.shape}", extra={"rank": self.rank})
        sched, gs = self.pipe.scheduler, self.cfg.guidance_scale
        LOG.info(f"Starting denoising loop with {len(sched.timesteps)} steps", extra={"rank": self.rank})
        
        for t in tqdm(sched.timesteps, 
                     desc=f"Rank {self.rank}: denoising",
                     position=self.rank,
                     leave=False):
            LOG.debug(f"Step {t.item()}", extra={"rank": self.rank})
            x = sched.scale_model_input(torch.cat([lat]*2), t)
            emb = torch.cat([self.uncond_emb, self.cond_emb], dim=0)
            
            LOG.debug("Running UNet", extra={"rank": self.rank})
            with torch.no_grad():
                noise = self.unet(x, t, encoder_hidden_states=emb).sample
            u, c = noise.chunk(2)
            lat = sched.step(u + gs*(c-u), t, lat).prev_sample
            LOG.debug("Step complete", extra={"rank": self.rank})
            
        LOG.info("Denoising loop complete", extra={"rank": self.rank})
        return lat

    def __call__(self):
        cfg = self.cfg
        T, H, W = cfg.num_frames, cfg.height//8, cfg.width//8

        LOG.info("Starting chunk size calculation", extra={"rank": self.rank})
        # Calculate optimal chunk size based on number of ranks and frames
        if cfg.chunk_size <= 0:  # Auto-calculate if not specified
            min_chunk_size = max(4, T // (self.world * 2))
            max_chunk_size = min(16, T // self.world)
            cs = min(max_chunk_size, max(min_chunk_size, T // self.world))
        else:
            cs = cfg.chunk_size

        ov = min(cfg.overlap, cs // 3)

        
        def compute_chunks(chunk_sz: int) -> list:
            rng=[]
            i=0
            while i < T:
                rng.append((i, min(i+chunk_sz, T)))
                i += chunk_sz - ov
            return rng

        ranges = compute_chunks(cs)
        # try to tweak chunk_size upward until divisible
        if len(ranges) % self.world != 0:
            for delta in range(1, cs):  # at most cs-1 increments
                test_cs = cs + delta
                test_ranges = compute_chunks(test_cs)
                if len(test_ranges) % self.world == 0:
                    cs = test_cs
                    ranges = test_ranges
                    LOG.info(f"Adjusted chunk_size -> {cs} so that {len(ranges)} chunks divisible by {self.world}", extra={"rank": self.rank})
                    break
        # fallback: if still not divisible just pad with last frame chunk
        if len(ranges) % self.world != 0:
            need = self.world - (len(ranges) % self.world)
            last_start, last_end = ranges[-1]
            for _ in range(need):
                ranges.append((last_start, last_end))
            LOG.warning(f"Padded ranges to {len(ranges)} to match world size", extra={"rank": self.rank})

        LOG.info(f"Using chunk_size={cs}, overlap={ov} with {self.world} ranks", 
                extra={"rank": self.rank})

        # shared base noise
        LOG.info("Generating base noise", extra={"rank": self.rank})
        torch.manual_seed(0)
        base = torch.randn(1, self.unet.config.in_channels, T, H, W,
                            device=cfg.device, dtype=torch.float16)
        base *= self.pipe.scheduler.init_noise_sigma

        # chunk assignment
        LOG.info("Calculating chunk ranges", extra={"rank": self.rank})
        my_ranges = [r for i,r in enumerate(ranges) if i % self.world == self.rank]

        LOG.info(f"My chunks = {my_ranges}", extra={"rank":self.rank})
        
        # Process chunks with synchronisation
        out = []
        for chunk_idx, (s,e) in enumerate(my_ranges):
            LOG.info(f"Starting chunk {chunk_idx+1}/{len(my_ranges)}: {s}-{e}", extra={"rank": self.rank})
            chunk = base[:,:,s:e].clone()
            LOG.info(f"Chunk {s}-{e} has shape {chunk.shape}", extra={"rank": self.rank})
            
            LOG.info(f"Starting denoising for chunk {s}-{e}", extra={"rank": self.rank})
            # wrap in no_grad to free scheduler intermediate memory
            with torch.no_grad():
                denoised = self._denoise(chunk)
            LOG.info(f"Denoising complete for chunk {s}-{e}", extra={"rank": self.rank})
            
            LOG.info(f"Moving chunk {s}-{e} to CPU", extra={"rank": self.rank})
            out.append((s,e,denoised.cpu()))
            LOG.info(f"Finished processing chunk {s}-{e}", extra={"rank": self.rank})

        # Ensure all ranks have finished their work before gathering
        LOG.info("Waiting for all ranks to complete processing", extra={"rank": self.rank})
        dist.barrier()
        
        # Network emulation + gather
        gathered = [None] * self.world
        payload_bytes = sum((e - s) * self.unet.config.in_channels * 2 for (s, e, _) in out)
        if cfg.emu_bw_mbps > 0:
            time.sleep(payload_bytes / (cfg.emu_bw_mbps * 1e6 / 8))
        if cfg.emu_rtt_ms > 0:
            delay = random.gauss(cfg.emu_rtt_ms, cfg.emu_jitter_ms)
            time.sleep(max(0.0, delay / 1000.0))
        t0_net = time.time()
        dist.all_gather_object(gathered, out)
        net_gather_s = time.time() - t0_net
        LOG.info("all_gather complete", extra={"rank": self.rank})
        
        # All ranks continue past gather so that decode (a collective) is entered by everyone

        LOG.info("Starting final processing on rank 0", extra={"rank": self.rank})
        # stitch & blend
        full   = torch.zeros_like(base.cpu())
        weight = torch.zeros((1,1,T,1,1))
        ramp   = torch.linspace(0,1,ov).view(1,1,ov,1,1) if ov>0 else None

        LOG.info("Starting blending", extra={"rank": self.rank})
        for lst in gathered:
            for s,e,latc in lst:
                length = e - s
                w = torch.ones((1, 1, length, 1, 1))
                if ov > 0:
                    # If the chunk is shorter than ov, clamp the ramp length
                    k = min(ov, length)
                    if k > 0:
                        w[:, :, :k] = ramp[:, :, :k]
                        w[:, :, -k:] = torch.flip(ramp[:, :, :k], [2])
                full[:,:,s:e]  += latc * w
                weight[:,:,s:e]+= w

        lat = full / weight.clamp(min=1e-6)
        LOG.info("Blending complete", extra={"rank": self.rank})

        #Decode frames collectively (all ranks must enter)
        LOG.info("Starting frame decoding (collective)", extra={"rank": self.rank})
        frames = []
        for t_idx in tqdm(range(T),
                           desc=f"Rank {self.rank}: decoding",
                           position=self.rank,
                           leave=False):
            LOG.debug(f"Decoding frame {t_idx+1}/{T}", extra={"rank": self.rank})
            # VAE expects 4D (B,C,H,W); remove time dim
            z = lat[:, :, t_idx].to(cfg.device)
            with torch.no_grad():
                img_lat = self.vae.decode(z / 0.18215).sample

            img = (img_lat[0].permute(1, 2, 0) * 0.5 + 0.5).clamp(0, 1)
            frames.append((img * 255).byte().cpu().numpy())

        #Boundary-only temporal inconsistency metrics
        temp_instab = None
        flow_err = None
        if self.rank == 0 and len(frames) > 1:
            # identify boundaries between chunks (sorted globally)
            ranges_sorted = sorted(ranges, key=lambda x: x[0])
            boundary_ends = [e for (s, e) in ranges_sorted[:-1]]
            l1_diffs, flow_diffs = [], []
            for idx in boundary_ends:
                if 0 < idx < len(frames):
                    f_prev = frames[idx - 1]
                    f_next = frames[idx]
                    # L1 pixel difference
                    l1_diffs.append(np.mean(np.abs(f_next.astype(np.float32) - f_prev.astype(np.float32))))
                    # optical flow warping error
                    prev_gray = cv2.cvtColor(f_prev, cv2.COLOR_BGR2GRAY)
                    next_gray = cv2.cvtColor(f_next, cv2.COLOR_BGR2GRAY)
                    flow = cv2.calcOpticalFlowFarneback(prev_gray, next_gray, None,
                                                        0.5, 3, 15, 3, 5, 1.2, 0)
                    h, w = prev_gray.shape
                    flow_x = (np.arange(w)[None, :] + flow[:, :, 0]).astype(np.float32)
                    flow_y = (np.arange(h)[:, None] + flow[:, :, 1]).astype(np.float32)
                    warp_prev = cv2.remap(f_prev, flow_x, flow_y, cv2.INTER_LINEAR)
                    flow_diffs.append(np.mean(np.abs(warp_prev.astype(np.float32) - f_next.astype(np.float32))))

            temp_instab = float(np.mean(l1_diffs)) if l1_diffs else None
            flow_err = float(np.mean(flow_diffs)) if flow_diffs else None

        # Only rank-0 writes the video file
        if self.rank == 0:
            LOG.info("Writing video to out.mp4", extra={"rank": self.rank})
            vw = cv2.VideoWriter("out.mp4",
                                  cv2.VideoWriter_fourcc(*"mp4v"),
                                  cfg.fps, (cfg.width, cfg.height))
            for f in frames:
                vw.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
            vw.release()
            LOG.warning("Saved out.mp4", extra={"rank": 0})
        peak_local = torch.cuda.max_memory_allocated() // 1024 ** 2
        peak_all = torch.tensor(peak_local, device=cfg.device)
        # emulate network latency for reduce
        if cfg.emu_rtt_ms > 0:
            time.sleep(cfg.emu_rtt_ms / 1000.0)
        t0_red = time.time()
        dist.all_reduce(peak_all, op=dist.ReduceOp.MAX)
        net_reduce_s = time.time() - t0_red

        end_vram = vram_mb()
        LOG.info(f"Process complete. End VRAM {end_vram} MB", extra={"rank": self.rank})
        return {
            "world_size": self.world,
            "chunk_size": cs,
            "overlap": ov,
            "num_frames": T,
            "peak_vram_mb": int(peak_all.item()),
            "end_vram_mb": end_vram,
            "network_bytes": int(payload_bytes),
            "net_gather_s": net_gather_s,
            "net_reduce_s": net_reduce_s,
            "temp_instab": temp_instab,
            "flow_err": flow_err,
        }

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--model_id",       default="cerspense/zeroscope_v2_XL")
    p.add_argument("--prompt",         default="a rocket in space, 4k")
    p.add_argument("--num_frames",     type=int, default=25)
    p.add_argument("--steps",          type=int, default=50)
    p.add_argument("--guidance_scale", type=float, default=7.5)
    p.add_argument("--chunk_size",     type=int,   default=0)  # 0 means auto-calculate
    p.add_argument("--overlap",        type=int,   default=4)
    p.add_argument("--fps",            type=int,   default=8)
    p.add_argument("--height",         type=int,   default=576)
    p.add_argument("--width",          type=int,   default=1024)
    p.add_argument("--device",         default="cuda")
    p.add_argument("--mode",           default="split")
    p.add_argument("--context_weight", type=float, default=0.35)
    p.add_argument("--out_csv",        default="benchmarks.csv")
    
    p.add_argument("--emu_bw_mbps", type=float, default=0,
                   help="throttle bandwidth in Mbps (0 = no throttle)")
    p.add_argument("--emu_rtt_ms", type=float, default=0,
                   help="one-way latency in ms (0 = none)")
    p.add_argument("--emu_jitter_ms", type=float, default=0,
                   help="jitter stddev in ms")
    cfg=p.parse_args()

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    t0=time.time()
    bench=DistributedVideoDiffuser(cfg)
    res=bench()

    if bench.rank == 0:
        elapsed = time.time() - t0
        res.update({
            "timestamp": datetime.datetime.utcnow().isoformat(timespec="seconds"),
            "host": socket.gethostname(),
            "mode": cfg.mode,
            "latency_s": elapsed,
            "throughput_fps": round(cfg.num_frames / elapsed, 3),
        })

        header = [
            "timestamp","host","mode","world_size","num_frames",
            "chunk_size","overlap","latency_s","throughput_fps",
            "peak_vram_mb","end_vram_mb",
            "network_bytes","net_gather_s","net_reduce_s",
            "temp_instab","flow_err"
        ]

        write_hdr = not os.path.exists(cfg.out_csv)
        with open(cfg.out_csv, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=header)
            if write_hdr:
                w.writeheader()
            # ensure all header keys present
            w.writerow({k: res.get(k, "") for k in header})
        LOG.warning(f"Metrics appended to {cfg.out_csv}", extra={"rank": 0})

    if dist.is_initialized():
        dist.destroy_process_group()

if __name__=="__main__":
    main()
