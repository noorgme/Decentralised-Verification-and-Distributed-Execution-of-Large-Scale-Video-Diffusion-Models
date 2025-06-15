# fsdp_benchmark.py
import os
import time
import argparse
import logging
import pickle
import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    CPUOffload,
    MixedPrecision
)
from torch.distributed.fsdp.wrap import wrap
from diffusers import DiffusionPipeline
from transformers import CLIPProcessor, CLIPModel
import pynvml
import cv2

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("FSDPBench")

local_rank = int(os.environ.get("LOCAL_RANK", 0))
torch.cuda.set_device(local_rank)
device = torch.device(f"cuda:{local_rank}")

def measure_vram_mb():
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    return info.used // 1024**2

def decode_latents(vae, latents, device):
    # latents: [1, C, F, h, w]
    b, c, f, h, w = latents.shape
    lat = latents.permute(0,2,1,3,4).reshape(-1, c, h, w)
    if device.startswith("cuda"):
        lat = lat.half()
    with torch.no_grad():
        frames = vae.decode(lat/0.18215).sample
    frames = frames.reshape(b, f, -1, h*8, w*8).permute(0,2,1,3,4)
   
    return (frames/2 + 0.5).clamp(0,1)

def compute_flow_consistency(video, device):
    # video: [1, C, F, H, W]
    frames = (video[0].permute(1,2,3,0).cpu().numpy()*255).astype(np.uint8)
    flows = []
    for i in range(len(frames)-1):
        g0 = cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY)
        g1 = cv2.cvtColor(frames[i+1], cv2.COLOR_RGB2GRAY)
        flow = cv2.calcOpticalFlowFarneback(
            g0, g1, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )
        flows.append(flow)
    flows = torch.from_numpy(np.stack(flows)).to(device)
    diffs = torch.norm(torch.diff(flows, dim=0), dim=-1)
    return float((1.0/(1.0+diffs.mean())).cpu().item())

def compute_prompt_fidelity(clip, proc, video, prompt, device):
    # take center frame
    _, C, F, H, W = video.shape
    frame = (video[0,:,F//2].permute(1,2,0)*255).cpu().numpy().astype(np.uint8)
    inputs = proc(text=[prompt], images=[frame], return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        out = clip(**inputs)
    return float(F.cosine_similarity(out.text_embeds, out.image_embeds).item())



class FSDPBenchmark:
    def __init__(self, args):
        self.args = args
        dist.init_process_group("nccl")
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        # load pipeline
        pipe = DiffusionPipeline.from_pretrained(
            args.model_id,
            torch_dtype=torch.float16,
            use_safetensors=False,
            low_cpu_mem_usage=True
        )
        # wrap UNet + text encoder
        mp_policy = MixedPrecision(
            param_dtype=torch.float16,
            reduce_dtype=torch.float16,
            buffer_dtype=torch.float16
        )
        self.unet = FSDP(
            wrap(pipe.unet),
            cpu_offload=CPUOffload(offload_params=False),
            mixed_precision=mp_policy
            
        )
        self.text_encoder = FSDP(
            wrap(pipe.text_encoder),
            cpu_offload=CPUOffload(offload_params=False),
            mixed_precision=mp_policy
        )
      
        pipe.unet = self.unet.to(args.device)
        pipe.text_encoder = self.text_encoder.to(args.device)
        self.pipe = pipe
        self.vae = pipe.vae
        self.scheduler = pipe.scheduler
        self.tokenizer = pipe.tokenizer
        # CLIP
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(args.device)
        self.proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
       
        pynvml.nvmlInit()

    def prepare(self):
        # text
        toks = self.tokenizer(
            self.args.prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        )
        with torch.no_grad():
            txt = self.text_encoder(toks.input_ids.to(self.args.device))[0]
        # scheduler
        self.scheduler.set_timesteps(self.args.steps, device=self.args.device)
        # latents
        lat = torch.randn(
            (1,
             self.unet.config.in_channels,
             self.args.num_frames,
             self.vae.sample_size,
             self.vae.sample_size),
            device=self.args.device,
            dtype=torch.float16
        ) * self.scheduler.init_noise_sigma
        return lat, txt

    def run(self):
        # measure static VRAM
        static_vram = measure_vram_mb()
        # reset peak stats
        if self.args.device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        lat, txt = self.prepare()
        # full denoise
        for i, t in enumerate(self.scheduler.timesteps):
            inp = torch.cat([lat]*2)
            inp = self.scheduler.scale_model_input(inp, t)
            with torch.no_grad():
                noise = self.unet(inp, t, encoder_hidden_states=txt).sample
            u, cnd = noise.chunk(2)
            noise = u + 7.5*(cnd - u)
            lat = self.scheduler.step(noise, t, lat).prev_sample
        t1 = time.time()
        # measure peak VRAM
        peak_vram = (torch.cuda.max_memory_allocated() // 1024**2) if self.args.device.startswith("cuda") else 0
        end_vram = measure_vram_mb()
        # decode & quality
        video = decode_latents(self.vae, lat, self.args.device)
        flow_consistency = compute_flow_consistency(video, self.args.device)
        prompt_fid = compute_prompt_fidelity(
            self.clip, self.proc, video, self.args.prompt, self.args.device
        )
        # gather metrics
        metrics = {
            "world_size": self.world_size,
            "static_vram_mb": static_vram,
            "peak_vram_mb": peak_vram,
            "end_vram_mb": end_vram,
            "latency_s": t1 - t0,
            "flow_consistency": flow_consistency,
            "prompt_fidelity": prompt_fid
        }
        return metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="cerspense/zeroscope_v2_576w")
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--out", type=str, default="fsdp_results.npy")
    args = parser.parse_args()

    bench = FSDPBenchmark(args)
    metrics = bench.run()

    if bench.rank == 0:
        # save or append
        path = args.out
        if os.path.exists(path):
            allm = list(np.load(path, allow_pickle=True))
        else:
            allm = []
        allm.append(metrics)
        np.save(path, np.array(allm, dtype=object))
        logger.info(f"Saved metrics: {metrics}")
