"""Simple coordinator that splits a latent, kicks off worker jobs and
stitches the results back into a video."""

import os
import time
import torch
import pickle
import argparse
import subprocess
import numpy as np
from pathlib import Path
import logging
from typing import List, Dict, Tuple, Optional

from diffusers import DiffusionPipeline
from diffusers.utils import export_to_video

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DistributedCoordinator:
    
    
    def __init__(
        self,
        model_id: str = "cerspense/zeroscope_v2_576w",
        work_dir: str = "./output",
        device: str = "cuda",
        num_inference_steps: int = 50,
        chunk_size: int = 8,
        overlap: int = 2,
        num_workers: int = 2
    ):
        
        self.model_id = model_id
        self.work_dir = work_dir
        self.device = device
        self.num_inference_steps = num_inference_steps
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.num_workers = num_workers
        
        # Initialise directories
        os.makedirs(os.path.join(work_dir, "chunks"), exist_ok=True)
        os.makedirs(os.path.join(work_dir, "results"), exist_ok=True)
        
        # Initialise pipeline
        self.pipe = None
        self.vae = None
        self.text_encoder = None
        self.tokenizer = None
        self.scheduler = None
        self.unet = None

    def load_models(self):
        logger.info(f"Loading pipeline components from {self.model_id}")
        self.pipe = DiffusionPipeline.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            use_safetensors=False
        ).to(self.device)
        
        # Extract components
        self.vae = self.pipe.vae
        self.text_encoder = self.pipe.text_encoder
        self.tokenizer = self.pipe.tokenizer
        self.scheduler = self.pipe.scheduler
        self.unet = self.pipe.unet

    def prepare_latents(self, prompt: str, num_frames: int = 16) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepare initial latents and text embeddings."""

        # Tokenise text prompt
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        )
        text_input_ids = text_inputs.input_ids
        
        # Get text embeddings
        with torch.no_grad():
            text_embeddings = self.text_encoder(text_input_ids.to(self.device))[0]
        
        # Prepare scheduler
        self.scheduler.set_timesteps(self.num_inference_steps, device=self.device)
        
        # Create initial latents
        latents = torch.randn(
            (1, self.unet.config.in_channels, num_frames, 40, 72), # (batch_size, channels, frames, height, width)
            device=self.device,
            dtype=torch.float16 if self.device == "cuda" else torch.float32
        )
        latents = latents * self.scheduler.init_noise_sigma
        
        return latents, text_embeddings

    def split_into_chunks(self, latents: torch.Tensor, text_embeddings: torch.Tensor) -> List[Dict]:
        """Split latents into overlapping chunks."""
        chunks = []
        total_frames = latents.shape[2]
        
        for i in range(0, total_frames, self.chunk_size - self.overlap):
            start_idx = i
            end_idx = min(i + self.chunk_size, total_frames)
            
            # Extract chunk
            chunk = latents[:, :, start_idx:end_idx, :, :]
            
            chunks.append({
                'chunk': chunk,
                'text_embeddings': text_embeddings,
                'start_idx': start_idx,
                'end_idx': end_idx
            })
        
        return chunks

    def save_chunk(self, chunk_data: Dict, chunk_idx: int) -> str:
        """Save a chunk to disk."""
        chunk_file = os.path.join(self.work_dir, "chunks", f"chunk_{chunk_idx}.pkl")
        with open(chunk_file, 'wb') as f:
            pickle.dump({
                'chunk': chunk_data['chunk'].cpu().numpy(),
                'text_embeddings': chunk_data['text_embeddings'].cpu().numpy(),
                'start_idx': chunk_data['start_idx'],
                'end_idx': chunk_data['end_idx']
            }, f)
        return chunk_file

    def launch_worker(self, chunk_file: str, worker_idx: int) -> str:
        """Launch a worker process to process a chunk."""
        output_file = os.path.join(self.work_dir, "results", f"result_{worker_idx}.pkl")
        cmd = [
            "python", "distributed_worker.py",
            "--chunk_file", chunk_file,
            "--output_file", output_file,
            "--model_id", self.model_id,
            "--num_inference_steps", str(self.num_inference_steps)
        ]
        
        process = subprocess.Popen(cmd)
        return output_file

    def load_results(self, output_files: List[str]) -> List[Dict]:
        """Load processed chunks from disk."""
        processed_chunks = []
        
        for i, output_file in enumerate(output_files):
            if not os.path.exists(output_file):
                logger.warning(f"Result file {output_file} not found")
                continue
            
            try:
                with open(output_file, 'rb') as f:
                    result_data = pickle.load(f)
                
                # Convert numpy arrays to tensors
                result_data['chunk'] = torch.tensor(result_data['chunk'], 
                                                  dtype=torch.float16 if self.device == "cuda" else torch.float32,
                                                  device=self.device)
                processed_chunks.append(result_data)
                logger.info(f"Loaded result for chunk {i}")
                
            except Exception as e:
                logger.error(f"Error loading result file {output_file}: {str(e)}")
                import traceback
                logger.error(traceback.format_exc())
                continue
        
        return processed_chunks

    def stitch_chunks(self, processed_chunks: List[Dict], original_shape: Tuple[int, ...]) -> torch.Tensor:
        """Stitch processed chunks back together."""
        batch_size, channels, num_frames, height, width = original_shape
        logger.info(f"Stitching {len(processed_chunks)} chunks to shape {original_shape}")
        
        # Initialise the output tensor and weights
        output = torch.zeros(original_shape, device=self.device)
        weights = torch.zeros((num_frames,), device=self.device)
        
        # Add each chunk with appropriate weighting
        for chunk_info in processed_chunks:
            chunk = chunk_info['chunk']
            start_idx = chunk_info['start_idx']
            end_idx = chunk_info['end_idx']
            
            # Create weights for blending
            chunk_frames = end_idx - start_idx
            chunk_weights = torch.ones((chunk_frames,), device=self.device)
            
            # Apply weights
            output[:, :, start_idx:end_idx, :, :] += chunk * chunk_weights.view(1, 1, -1, 1, 1)
            weights[start_idx:end_idx] += chunk_weights
        
        # Normalise by weights to get the average
        weights = weights.view(1, 1, -1, 1, 1)
        output = output / (weights + 1e-8)
        
        logger.info(f"Successfully stitched chunks into tensor of shape {output.shape}")
        return output

    def decode_to_video(self, latents: torch.Tensor, output_path: str) -> str:
        """Decode latents to video and save to file."""
        logger.info("Decoding latents to pixel space")
        with torch.no_grad():
            # Convert to float16 if using CUDA
            if self.device == "cuda":
                latents = latents.half()
            
            # Reshape for VAE decoding
            batch_size, channels, num_frames, height, width = latents.shape
            latents_reshaped = latents.permute(0, 2, 1, 3, 4).reshape(-1, channels, height, width)
            
            # Decode all frames
            video_frames = self.vae.decode(latents_reshaped / 0.18215).sample
            
            # Reshape back
            video_frames = video_frames.reshape(batch_size, num_frames, -1, height * 8, width * 8)
            video_frames = video_frames.permute(0, 2, 1, 3, 4)
        
        # Post-process
        video_frames = (video_frames / 2 + 0.5).clamp(0, 1)
        video_frames = video_frames.cpu().numpy()
        
        # Save video
        logger.info(f"Saving video to {output_path}")
        temp_frames = (video_frames[0] * 255).astype(np.uint8)
        temp_frames = np.transpose(temp_frames, (1, 2, 3, 0))
        export_to_video(temp_frames, output_path, fps=8)
        
        return output_path

    def cleanup(self):
        """Clean up temporary files."""
        if os.path.exists(self.work_dir):
            logger.info(f"Cleaning up temporary directory: {self.work_dir}")
            try:
                os.removedirs(self.work_dir)
            except OSError:
                logger.info("Directory not empty, skipping cleanup")

def main():
    parser = argparse.ArgumentParser(description="Distributed video generation coordinator")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt for video generation")
    parser.add_argument("--output", type=str, default="output.mp4", help="Output video path")
    parser.add_argument("--num_frames", type=int, default=16, help="Number of frames to generate")
    parser.add_argument("--model_id", type=str, default="cerspense/zeroscope_v2_576w", help="Model ID")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    args = parser.parse_args()
    
    # Initialise coordinator
    coordinator = DistributedCoordinator(
        model_id=args.model_id,
        device=args.device
    )
    
    try:
        # Load models
        coordinator.load_models()
        
        # Prepare latents
        latents, text_embeddings = coordinator.prepare_latents(args.prompt, args.num_frames)
        
        # Split into chunks
        chunks = coordinator.split_into_chunks(latents, text_embeddings)
        
        # Save chunks and launch workers
        chunk_files = []
        output_files = []
        
        for i, chunk in enumerate(chunks):
            chunk_file = coordinator.save_chunk(chunk, i)
            chunk_files.append(chunk_file)
        #     output_file = coordinator.launch_worker(chunk_file, i)
        #     output_files.append(output_file)
        
        # # Wait for workers to complete
        # for process in processes:
        #     process.wait()
        
        # # Load and stitch results
        # processed_chunks = coordinator.load_results(output_files)
        # stitched_latents = coordinator.stitch_chunks(processed_chunks, latents.shape)
        
        # # Decode to video
        # coordinator.decode_to_video(stitched_latents, args.output)
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)
    
    finally:
        
        coordinator.cleanup()

if __name__ == "__main__":
    main() 