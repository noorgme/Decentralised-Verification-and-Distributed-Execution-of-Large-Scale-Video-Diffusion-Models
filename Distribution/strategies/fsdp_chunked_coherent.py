# FSDP + chunking + CCI and inter-chunk smoothing benchmark.

import os
import time
import csv
import datetime
import socket
import argparse
import logging
import random
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import cv2
import pynvml
from tqdm import tqdm
from torch.nn import ModuleList, Sequential, ModuleDict
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, CPUOffload, MixedPrecision
from torch.distributed.fsdp.fully_sharded_data_parallel import ShardingStrategy
from diffusers import DiffusionPipeline

local_rank = int(os.getenv("LOCAL_RANK", 0))


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
torch.cuda.set_device(local_rank)
def vram_mb():
    h = pynvml.nvmlDeviceGetHandleByIndex(local_rank)
    return pynvml.nvmlDeviceGetMemoryInfo(h).used // 1024**2

class DistributedVideoDiffuser:
    def __init__(self, cfg):
        self.cfg = cfg
        dist.init_process_group("nccl")
        self.rank  = dist.get_rank()
        self.world = dist.get_world_size()

        LOG.info("Loading pipeline on CPU", extra={"rank":self.rank})
        pipe = DiffusionPipeline.from_pretrained(
            cfg.model_id,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            use_safetensors=False,
            device_map=None
        )

        mp = MixedPrecision(torch.float16, torch.float16, torch.float16)
        def wrap_policy(module, recurse, nonwrapped_numel):
            if isinstance(module, (ModuleList, Sequential, ModuleDict)): return False
            return nonwrapped_numel >= 10_000_000
        fsdp_kwargs = dict(
            auto_wrap_policy = wrap_policy,
            sharding_strategy= ShardingStrategy.FULL_SHARD,
            cpu_offload      = CPUOffload(offload_params=True),
            mixed_precision  = mp,
            device_id        = torch.cuda.current_device(),
            use_orig_params  = False
        )

        if cfg.use_fsdp:
            self.unet         = FSDP(pipe.unet,         **fsdp_kwargs)
            self.text_encoder = FSDP(pipe.text_encoder, **fsdp_kwargs)
        else:
            self.unet         = pipe.unet.to(cfg.device)
            self.text_encoder = pipe.text_encoder.to(cfg.device)

        vae = pipe.vae
        fsdp_kw_vae = dict(fsdp_kwargs, cpu_offload=CPUOffload(offload_params=True))
        for nm, sm in vae.named_children():
            if any(p.requires_grad for p in sm.parameters()):
                setattr(vae, nm, FSDP(sm, **fsdp_kw_vae) if cfg.use_fsdp else sm)
        self.vae = vae

        pipe.unet         = self.unet
        pipe.text_encoder = self.text_encoder
        pipe.vae          = self.vae
        self.pipe = pipe

        pipe.scheduler.set_timesteps(cfg.steps, device=cfg.device)
        toks = pipe.tokenizer([cfg.prompt, ""],
                              padding="max_length",
                              max_length=pipe.tokenizer.model_max_length,
                              truncation=True,
                              return_tensors="pt")
        with torch.no_grad():
            emb = self.text_encoder(toks.input_ids.to(cfg.device))[0]
        self.cond_emb, self.uncond_emb = emb[:1], emb[1:]

        if cfg.use_ctx:
            C = self.unet.config.in_channels
            shape_ctx = (1, C, 1, cfg.height // 8, cfg.width // 8)
            
            # Compute global context as average latent over all timesteps on rank 0
            if self.rank == 0:
                # Create initial noise for the full sequence
                torch.manual_seed(0)
                full_noise = torch.randn(1, C, cfg.num_frames, cfg.height // 8, cfg.width // 8, 
                                       device=cfg.device, dtype=torch.float16)
                full_noise *= self.pipe.scheduler.init_noise_sigma
                
                # Compute average latent across all timesteps
                ctx = full_noise.mean(dim=2, keepdim=True)  # Average over time dimension
                LOG.info(f"Computed global context with shape {ctx.shape}", extra={"rank": self.rank})
            else:
                ctx = torch.empty(shape_ctx, device=cfg.device, dtype=torch.float16)
            
            # Broadcast the global context to all ranks
            dist.broadcast(ctx, src=0)
            self.ctx = ctx
        else:
            self.ctx = None

    def _denoise(self, lat: torch.Tensor) -> torch.Tensor:
        LOG.info(f"Entering _denoise() with latent shape {lat.shape}", extra={"rank": self.rank})
        sched, gs = self.pipe.scheduler, self.cfg.guidance_scale
        for t in tqdm(sched.timesteps, desc=f"Rank {self.rank}: denoising", position=self.rank, leave=False):
            x = sched.scale_model_input(torch.cat([lat]*2), t)
            if self.ctx is not None:
                # Inject global context: repeat context to match chunk length and add weighted context
                ctx_rep = self.ctx.repeat(1, 1, lat.shape[2], 1, 1)
                x = x + self.cfg.context_weight * ctx_rep
            emb = torch.cat([self.uncond_emb, self.cond_emb], dim=0)
            with torch.no_grad():
                noise = self.unet(x, t, encoder_hidden_states=emb).sample
            u, c = noise.chunk(2)
            lat = sched.step(u + gs*(c-u), t, lat).prev_sample
        return lat

    def __call__(self):
        cfg = self.cfg
        T, H, W = cfg.num_frames, cfg.height//8, cfg.width//8

        if cfg.no_chunking:
            cs, ov = T, 0
        else:
            if cfg.chunk_size <= 0:
                min_chunk = max(4, T // (self.world * 2))
                max_chunk = min(16, T // self.world)
                cs = min(max_chunk, max(min_chunk, T // self.world))
            else:
                cs = cfg.chunk_size
            ov = cfg.overlap if cfg.overlap > 0 else max(4, cs // 3)

        def compute_chunks(sz):
            rng, i = [], 0
            while i < T:
                rng.append((i, min(i+sz, T)))
                i += sz - ov
            return rng

        ranges = compute_chunks(cs)
        if len(ranges) % self.world != 0:
            for d in range(1, cs):
                test = compute_chunks(cs+d)
                if len(test) % self.world == 0:
                    cs, ranges = cs+d, test
                    break
        if len(ranges) % self.world != 0:
            need = self.world - (len(ranges) % self.world)
            s,e = ranges[-1]
            ranges += [(s,e)] * need

        LOG.info(f"chunk={cs}, overlap={ov}, world={self.world}", extra={"rank":self.rank})
        torch.manual_seed(0)
        base = torch.randn(1, self.unet.config.in_channels, T, H, W, device=cfg.device, dtype=torch.float16)
        base *= self.pipe.scheduler.init_noise_sigma

        my_ranges = [r for i,r in enumerate(ranges) if i % self.world == self.rank]
        out = []
        for s,e in my_ranges:
            chunk = base[:,:,s:e].clone()
            with torch.no_grad():
                den = self._denoise(chunk)
            out.append((s,e,den.cpu()))
        dist.barrier()

        gathered = [None]*self.world
        payload_bytes = sum((e-s)*self.unet.config.in_channels*2 for (s,e,_) in out)
        if cfg.emu_bw_mbps > 0:
            time.sleep(payload_bytes / (cfg.emu_bw_mbps * 1e6 / 8))
        if cfg.emu_rtt_ms > 0:
            delay = random.gauss(cfg.emu_rtt_ms, cfg.emu_jitter_ms)
            time.sleep(max(0.0, delay / 1000.0))    
        t0_net = time.time()
        dist.all_gather_object(gathered, out)
        gather_time = time.time() - t0_net

        full = torch.zeros_like(base.cpu())
        weight = torch.zeros((1,1,T,1,1))
        ramp = torch.linspace(0,1,ov).view(1,1,ov,1,1) if ov>0 else None
        for lst in gathered:
            for s,e,latc in lst:
                length = e-s
                w = torch.ones((1,1,length,1,1))
                if ov>0:
                    k = min(ov, length)
                    w[:,:,:k]  = ramp[:,:,:k]
                    w[:,:,-k:] = torch.flip(ramp[:,:,:k],[2])
                full[:,:,s:e] += latc * w
                weight[:,:,s:e] += w
        lat = full / weight.clamp(min=1e-6)

        frames=[]
        for i in range(T):
            z = lat[:,:,i].to(cfg.device)
            with torch.no_grad():
                img_lat = self.vae.decode(z/0.18215).sample
            img = (img_lat[0].permute(1,2,0)*0.5+0.5).clamp(0,1)
            frames.append((img*255).byte().cpu().numpy())

        temp_instab = None
        flow_err = None
        if self.rank == 0 and len(frames) > 1 and not cfg.no_chunking:
            ranges_sorted = sorted(ranges, key=lambda x: x[0])
            boundary_ends = [e for (s,e) in ranges_sorted[:-1]]
            l1_diffs = []
            flow_diffs = []
            for idx in boundary_ends:
                if 0 < idx < len(frames):
                    f_prev, f_next = frames[idx-1], frames[idx]
                    l1_diffs.append(np.mean(np.abs(f_next.astype(np.float32) - f_prev.astype(np.float32))))
                    prev_gray = cv2.cvtColor(f_prev, cv2.COLOR_BGR2GRAY)
                    next_gray = cv2.cvtColor(f_next, cv2.COLOR_BGR2GRAY)
                    flow = cv2.calcOpticalFlowFarneback(prev_gray, next_gray,
                                                        None, 0.5, 3, 15, 3, 5, 1.2, 0)
                    h, w = prev_gray.shape
                    flow_x = (np.arange(w)[None,:] + flow[:,:,0]).astype(np.float32)
                    flow_y = (np.arange(h)[:,None] + flow[:,:,1]).astype(np.float32)
                    warp_prev = cv2.remap(f_prev, flow_x, flow_y, cv2.INTER_LINEAR)
                    flow_diffs.append(np.mean(np.abs(warp_prev.astype(np.float32) - f_next.astype(np.float32))))
            temp_instab = float(np.mean(l1_diffs)) if l1_diffs else None
            flow_err = float(np.mean(flow_diffs)) if flow_diffs else None

        if self.rank == 0:
            vw = cv2.VideoWriter("out.mp4", cv2.VideoWriter_fourcc(*"mp4v"), cfg.fps, (cfg.width, cfg.height))
            for f in frames: vw.write(cv2.cvtColor(f,cv2.COLOR_RGB2BGR))
            vw.release()

        peak_local = torch.cuda.max_memory_allocated()//1024**2
        peak_all   = torch.tensor(peak_local, device=cfg.device)
        if cfg.emu_rtt_ms > 0:
            time.sleep(cfg.emu_rtt_ms / 1000.0)
        t0_red = time.time()
        dist.all_reduce(peak_all, op=dist.ReduceOp.MAX)
        reduce_time = time.time() - t0_red
        end_v = vram_mb()

        return {
            "world_size": self.world,
            "chunk_size": cs,
            "overlap": ov,
            "num_frames": T,
            "peak_vram_mb": int(peak_all.item()),
            "end_vram_mb": end_v,
            "network_bytes": int(payload_bytes),
            "net_gather_s": gather_time,
            "net_reduce_s": reduce_time,
            "temp_instab": temp_instab,
            "flow_err": flow_err,
        }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_id", default="cerspense/zeroscope_v2_XL")
    p.add_argument("--prompt", default="a rocket in space, 4k")
    p.add_argument("--num_frames", type=int, default=32)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--guidance_scale", type=float, default=7.5)
    p.add_argument("--chunk_size", type=int, default=0)
    p.add_argument("--overlap", type=int, default=4)
    p.add_argument("--fps", type=int, default=8)
    p.add_argument("--height", type=int, default=576)
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--device", default="cuda")
    p.add_argument("--mode", choices=["fsdp","chunk","hybrid","hybrid_ctx"], default="hybrid_ctx")
    p.add_argument("--context_weight", type=float, default=0.35)
    p.add_argument("--emu_bw_mbps", type=float, default=0,
                   help="throttle bandwidth in Mbps (0 = no throttle)")
    p.add_argument("--emu_rtt_ms", type=float, default=0,
                   help="one-way latency in ms (0 = no extra delay)")
    p.add_argument("--emu_jitter_ms", type=float, default=0,
                   help="jitter stddev in ms (0 = no jitter)")
    p.add_argument("--out_csv", default="results.csv")
    cfg = p.parse_args()

    cfg.use_fsdp    = cfg.mode in ["fsdp","hybrid","hybrid_ctx"]
    cfg.no_chunking = cfg.mode == "fsdp"
    cfg.use_ctx     = cfg.mode == "hybrid_ctx"

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    bench = DistributedVideoDiffuser(cfg)
    res = bench()

    if bench.rank == 0:
        elapsed = time.time() - t0
        res.update({
            "timestamp": datetime.datetime.utcnow().isoformat(timespec="seconds"),
            "host": socket.gethostname(),
            "mode": cfg.mode,
            "latency_s": elapsed,
            "throughput_fps": round(cfg.num_frames/elapsed,3),
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
            if write_hdr: w.writeheader()
            w.writerow({k: res.get(k, "") for k in header})
        LOG.warning(f"Metrics appended ->  {cfg.out_csv}", extra={"rank":0})

    if dist.is_initialized():
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
