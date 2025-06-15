#!/usr/bin/env python3

import os, time, argparse, logging, csv, datetime, socket, random

local_rank = int(os.getenv("LOCAL_RANK", 0))
import numpy as np
import torch, torch.distributed as dist
import torch.nn.functional as F
import cv2, pynvml
from tqdm import tqdm
from diffusers import DiffusionPipeline

class _RankFilter(logging.Filter):
    def __init__(self, rank: int):
        super().__init__()
        self._rank = rank

    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "rank"):
            record.rank = self._rank
        return True

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [Rank %(rank)d] %(levelname)s: %(message)s")
logging.getLogger().addFilter(_RankFilter(local_rank))

LOG = logging.getLogger("ChunkOnly")

pynvml.nvmlInit()
torch.cuda.set_device(local_rank)

def vram_mb():
    handle = pynvml.nvmlDeviceGetHandleByIndex(local_rank)
    return pynvml.nvmlDeviceGetMemoryInfo(handle).used // 1024 ** 2

class ChunkedVideoDiffuser:
    def __init__(self, cfg):
        self.cfg = cfg
        dist.init_process_group("nccl")
        self.rank  = dist.get_rank()
        self.world = dist.get_world_size()

        LOG.info("Loading pipeline on GPU ", extra={"rank": self.rank})
        pipe = DiffusionPipeline.from_pretrained(
            cfg.model_id,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map=None,
            use_safetensors=False,
        )
        pipe.to(cfg.device)
        self.pipe = pipe

        # Scheduler & text embeddings
        pipe.scheduler.set_timesteps(cfg.steps, device=cfg.device)
        toks = pipe.tokenizer([cfg.prompt, ""],
                              padding="max_length",
                              max_length=pipe.tokenizer.model_max_length,
                              truncation=True,
                              return_tensors="pt")
        with torch.no_grad():
            emb = pipe.text_encoder(toks.input_ids.to(cfg.device))[0]
        self.cond_emb, self.uncond_emb = emb[:1], emb[1:]

    def _denoise(self, lat: torch.Tensor) -> torch.Tensor:
        sched, gs = self.pipe.scheduler, self.cfg.guidance_scale
        for t in sched.timesteps:
            x   = sched.scale_model_input(torch.cat([lat] * 2), t)
            emb = torch.cat([self.uncond_emb, self.cond_emb], 0)
            with torch.no_grad():
                noise = self.pipe.unet(x, t, encoder_hidden_states=emb).sample
            u, c = noise.chunk(2)
            lat  = sched.step(u + gs * (c - u), t, lat).prev_sample
        return lat

    def __call__(self):
        cfg = self.cfg
        T, H, W = cfg.num_frames, cfg.height // 8, cfg.width // 8

        if cfg.chunk_size <= 0:
            min_chunk = max(4, T // (self.world * 2))
            max_chunk = min(16, T // self.world)
            cs = min(max_chunk, max(min_chunk, T // self.world))
        else:
            cs = cfg.chunk_size
        ov = min(cfg.overlap, cs // 3)

        def make_ranges(sz):
            rng, i = [], 0
            while i < T:
                rng.append((i, min(i + sz, T)))
                i += sz - ov
            return rng

        ranges = make_ranges(cs)
        if len(ranges) % self.world != 0:
            for delta in range(1, cs):
                test = make_ranges(cs + delta)
                if len(test) % self.world == 0:
                    cs, ranges = cs + delta, test
                    break
        if len(ranges) % self.world != 0:
            need = self.world - (len(ranges) % self.world)
            last_s, last_e = ranges[-1]
            ranges += [(last_s, last_e)] * need

        LOG.info(f"chunk_size={cs}, overlap={ov}, world={self.world}", extra={"rank":self.rank})

        # Shared base noise (identical seed)
        torch.manual_seed(0)
        base = torch.randn(1, self.pipe.unet.config.in_channels, T, H, W,
                           device=cfg.device, dtype=torch.float16)
        base *= self.pipe.scheduler.init_noise_sigma

        my_ranges = [r for i, r in enumerate(ranges) if i % self.world == self.rank]

        out = []
        for s, e in tqdm(my_ranges, desc="Denoising chunks", disable=self.rank != 0):
            chunk = base[:, :, s:e].clone()
            with torch.no_grad(): den = self._denoise(chunk)
            out.append((s, e, den.cpu()))

        dist.barrier()

        gathered = [None] * self.world
        payload_bytes = sum((e - s) * self.pipe.unet.config.in_channels * 2 for (s, e, _) in out)
        if cfg.emu_bw_mbps > 0:
            time.sleep(payload_bytes / (cfg.emu_bw_mbps * 1e6 / 8))
        if cfg.emu_rtt_ms > 0:
            delay = random.gauss(cfg.emu_rtt_ms, cfg.emu_jitter_ms)
            time.sleep(max(0.0, delay / 1000.0))
        t0_net = time.time()
        dist.all_gather_object(gathered, out)
        net_gather_s = time.time() - t0_net

        full   = torch.zeros_like(base.cpu())
        weight = torch.zeros((1, 1, T, 1, 1))
        ramp   = torch.linspace(0, 1, ov).view(1, 1, ov, 1, 1) if ov > 0 else None
        for lst in gathered:
            for s, e, latc in lst:
                length = e - s
                w = torch.ones((1, 1, length, 1, 1))
                if ov > 0:
                    k = min(ov, length)
                    if k > 0:
                        w[:, :, :k]  = ramp[:, :, :k]
                        w[:, :, -k:] = torch.flip(ramp[:, :, :k], [2])
                full[:, :, s:e]  += latc * w
                weight[:, :, s:e] += w
        lat = full / weight.clamp(min=1e-6)

        frames = []
        for idx in tqdm(range(T), desc="Decoding frames", disable=self.rank != 0):
            z = lat[:, :, idx].to(cfg.device, dtype=torch.float16)
            with torch.no_grad():
                img_lat = self.pipe.vae.decode(z / 0.18215).sample
            img = (img_lat[0].permute(1, 2, 0) * 0.5 + 0.5).clamp(0, 1)
            frames.append((img * 255).byte().cpu().numpy())

        if self.rank == 0:
            vw = cv2.VideoWriter("out.mp4", cv2.VideoWriter_fourcc(*"mp4v"),
                                 cfg.fps, (cfg.width, cfg.height))
            for f in frames:
                vw.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
            vw.release()
            LOG.warning("Saved out.mp4", extra={"rank": 0})

        peak_local = torch.cuda.max_memory_allocated() // 1024 ** 2
        peak_all = torch.tensor(peak_local, device=cfg.device)
        dist.barrier()

        # emulate network latency for reduce
        if cfg.emu_rtt_ms > 0:
            time.sleep(cfg.emu_rtt_ms / 1000.0)
        t0_red = time.time()
        dist.all_reduce(peak_all, op=dist.ReduceOp.MAX)
        net_reduce_s = time.time() - t0_red

        end_local = vram_mb()
        end_all   = torch.tensor(end_local, device=cfg.device)
        dist.all_reduce(end_all, op=dist.ReduceOp.MAX)
    
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

        return {
            "world_size": self.world,
            "chunk_size": cs,
            "overlap": ov,
            "num_frames": T,
            "peak_vram_mb": int(peak_all.item()),
            "end_vram_mb": int(end_all.item()),
            "network_bytes": int(payload_bytes),
            "net_gather_s": net_gather_s,
            "net_reduce_s": net_reduce_s,
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
    p.add_argument("--out_csv", default="benchmarks.csv")
    p.add_argument("--emu_bw_mbps", type=float, default=0,
                   help="throttle bandwidth in Mbps (0 = no throttle)")
    p.add_argument("--emu_rtt_ms", type=float, default=0,
                   help="one-way latency in ms (0 = none)")
    p.add_argument("--emu_jitter_ms", type=float, default=0,
                   help="jitter stddev in ms")
    cfg = p.parse_args()

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()

    bench = ChunkedVideoDiffuser(cfg)
    res   = bench()

    if dist.get_rank() == 0:
        elapsed = time.time() - t0
        res.update({
            "timestamp": datetime.datetime.utcnow().isoformat(timespec="seconds"),
            "host": socket.gethostname(),
            "mode": "chunk", 
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
            w.writerow({k: res.get(k, "") for k in header})
        LOG.warning(f"Metrics appended ->  {cfg.out_csv}", extra={"rank": 0})

    if dist.is_initialized():
        dist.destroy_process_group()

if __name__ == "__main__":
    main() 