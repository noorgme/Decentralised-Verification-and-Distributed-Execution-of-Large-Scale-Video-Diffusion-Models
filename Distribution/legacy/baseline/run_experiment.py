import os
import time
import torch
import pynvml
import psutil
import numpy as np
import pandas as pd
import cv2
import torch.nn.functional as F
from diffusers import DiffusionPipeline
from transformers import CLIPProcessor, CLIPModel
from typing import Dict, Any, List, Tuple
from pathlib import Path
import logging
import time
import torch
import pynvml
import psutil
import numpy as np
import cv2
import torch.nn.functional as F
from diffusers import DiffusionPipeline
from transformers import CLIPProcessor, CLIPModel
from typing import Dict, Any, List, Tuple
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BaselineBenchmark:
    """
    Benchmark for Strategy A: full-latent denoise on one device.
    Measures end-to-end latency, VRAM usage, network traffic,
    flow consistency, prompt fidelity, overlap MSE (zero), and speedup.
    """
    def __init__(self,
                 model_id: str = "cerspense/zeroscope_v2_576w",
                 device: str = None,
                 num_frames: int = 16,
                 height: int = 320,
                 width: int = 576,
                 num_steps: int = 50,
                 prompt: str = "A rocket flying off into space, cinematic, 4K"):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_id = model_id
        self.num_frames = num_frames
        self.height = height
        self.width = width
        self.num_steps = num_steps
        self.prompt = prompt
        # initialize models
        self._load_diffusion()
        self._load_clip()
        # initialize NVML
        pynvml.nvmlInit()

    def _load_diffusion(self):
        logger.info(f"Loading diffusion pipeline: {self.model_id}")
        self.pipe = DiffusionPipeline.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16 if self.device.startswith("cuda") else torch.float32,
            use_safetensors=False,
            low_cpu_mem_usage=True
        ).to(self.device)
        self.vae = self.pipe.vae
        self.unet = self.pipe.unet
        self.scheduler = self.pipe.scheduler
        self.tokenizer = self.pipe.tokenizer
        self.text_encoder = self.pipe.text_encoder
        self.vae.eval()
        self.unet.eval()

    def _load_clip(self):
        logger.info("Loading CLIP model for prompt fidelity")
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_model.eval()

    def _measure_vram(self) -> Tuple[int,int]:
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        used = info.used // 1024**2
        total = info.total // 1024**2
        return used, total

    def _measure_network(self) -> Tuple[int,int]:
        counters = psutil.net_io_counters()
        return counters.bytes_sent, counters.bytes_recv

    def prepare_latents(self) -> Tuple[torch.Tensor, torch.Tensor]:
        # text encoding
        inputs = self.tokenizer(
            self.prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        )
        text_ids = inputs.input_ids.to(self.device)
        with torch.no_grad():
            text_emb = self.text_encoder(text_ids)[0]
        # scheduler
        self.scheduler.set_timesteps(self.num_steps, device=self.device)
        # initial latent
        latents = torch.randn(
            (1, self.unet.config.in_channels,
             self.num_frames,
             self.height//8,
             self.width//8),
            device=self.device,
            dtype=torch.float16 if self.device.startswith("cuda") else torch.float32
        ) * self.scheduler.init_noise_sigma
        return latents, text_emb

    def run_inference(self) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Run a single inference pass and return the generated video and metrics.
        """
        try:
            # measure baseline metrics
            start_bytes = self._measure_network()[0]
            start_time = time.time()
            start_vram = self._measure_vram()[0]
            # reset peak counter
            if self.device.startswith("cuda"):
                torch.cuda.reset_peak_memory_stats()
            
            # prepare inputs
            latents, text_emb = self.prepare_latents()
            
            # denoise full latent
            for t in self.scheduler.timesteps:
                latent_input = torch.cat([latents] * 2)
                latent_input = self.scheduler.scale_model_input(latent_input, t)
                with torch.no_grad():
                    # Expand text embeddings to match batch size of latent_input
                    emb = text_emb.expand(latent_input.size(0), -1, -1)
                    noise_pred = self.unet(latent_input, t, encoder_hidden_states=emb).sample
                u, c = noise_pred.chunk(2)
                noise_pred = u + 7.5 * (c - u)
                latents = self.scheduler.step(noise_pred, t, latents).prev_sample
            
            # decode video
            video = self.decode_latents(latents)
            
            # measure final metrics
            end_time = time.time()
            end_vram = self._measure_vram()[0]
            peak_vram  = (torch.cuda.max_memory_allocated() // 1024**2) if self.device.startswith("cuda") else end_vram
            end_bytes = self._measure_network()[0]
            
            metrics = {
                'latency_s': end_time - start_time,
                'vram_used_start_mb': start_vram,
                'vram_used_end_mb': end_vram,
                'vram_peak_mb':       peak_vram,
                'network_bytes_sent': end_bytes - start_bytes,
                'network_bytes_recv': None,  # unchanged
            }
            
            return video, metrics
            
        except Exception as e:
            logger.error(f"Error during inference: {e}")
            raise
        finally:
            # Clear any cached tensors
            torch.cuda.empty_cache()

    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            b, c, f, h, w = latents.shape
            lat = latents.permute(0,2,1,3,4).reshape(-1,c,h,w)
            if lat.dtype != self.vae.dtype:
                lat = lat.to(self.vae.dtype)
            frames = self.vae.decode(lat/0.18215).sample
            frames = frames.reshape(b,f,-1,h*8,w*8).permute(0,2,1,3,4)
            return (frames/2+0.5).clamp(0,1)

    def compute_flow_consistency(self, frames: torch.Tensor) -> float:
        """
        Compute consistency of optical flow between consecutive frames.
        Expects `frames` shape (B, C, F, H, W) with C=3.
        """
        # Process first batch element
        fn = frames[0]  # shape (C, F, H, W)
        # Permute to (F, H, W, C)
        frames_np = (fn.permute(1, 2, 3, 0).cpu().numpy() * 255).astype(np.uint8)

        flows = []
        for i in range(frames_np.shape[0] - 1):
            prev = cv2.cvtColor(frames_np[i], cv2.COLOR_RGB2GRAY)
            curr = cv2.cvtColor(frames_np[i+1], cv2.COLOR_RGB2GRAY)
            flow = cv2.calcOpticalFlowFarneback(
                prev, curr, None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0
            )
            flows.append(flow)

        # Stack and compute magnitude differences
        fts = torch.from_numpy(np.stack(flows)).to(frames.device)
        diffs = torch.diff(fts, dim=0)
        mags = torch.norm(diffs, dim=-1)
        return float(1.0 / (1.0 + mags.mean()))

    def compute_prompt_fidelity(self, frames: torch.Tensor) -> float:
        # average CLIP similarity across frames
        imgs = []
        for fr in frames[0].permute(1,2,3,0).cpu().numpy():
            imgs.append((fr*255).astype(np.uint8))
        inputs = self.clip_processor(text=[self.prompt], images=imgs, return_tensors="pt", padding=True).to(self.device)
        out = self.clip_model(**inputs)
        sims = torch.cosine_similarity(out.image_embeds, out.text_embeds.unsqueeze(1), dim=-1)
        return float(sims.mean())

    def run(self, runs: int = 5) -> List[Dict[str, Any]]:
        results = []
        for i in range(runs):
            logger.info(f"Baseline run {i+1}/{runs}")
            video, meta = self.run_inference()
            fc = self.compute_flow_consistency(video)
            pf = self.compute_prompt_fidelity(video)
            meta.update({'flow_consistency': fc, 'prompt_fidelity': pf, 'overlap_mse': 0.0, 'speedup':1.0})
            results.append(meta)
        return results

if __name__ == "__main__":
    bench = BaselineBenchmark()
    data = bench.run(runs=10)
    # Convert list of dictionaries to DataFrame and save as CSV
    df = pd.DataFrame(data)
    out = Path("baseline_results.csv")
    df.to_csv(out, index=False)
    logger.info(f"Saved baseline results to {out}")
