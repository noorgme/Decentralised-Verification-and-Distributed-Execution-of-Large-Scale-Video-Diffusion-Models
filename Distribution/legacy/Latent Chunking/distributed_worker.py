#!/usr/bin/env python
"""Simple worker that handles one latent chunk so we can spread jobs across
several machines or processes."""

import os
import sys
import time
import torch
import json
import pickle
import argparse
import numpy as np
from pathlib import Path
import logging

from diffusers import DiffusionPipeline

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LatentChunkWorker:
    """
    Worker for processing a chunk of a video latent.
    """
    
    def __init__(self, 
                 model_id: str = "cerspense/zeroscope_v2_576w",
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """Initialise the worker."""
        self.model_id = model_id
        self.device = device
        
        logger.info(f"Loading Zeroscope model {model_id} on {device}")
        self.pipe = DiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            use_safetensors=False
        ).to(device)
        
        self.unet = self.pipe.unet
        self.scheduler = self.pipe.scheduler
        
        logger.info("Model loaded successfully")
    
    def process_chunk(self, chunk_number: int, num_inference_steps: int = 25):
        try:
            script_dir = Path(__file__).parent
            work_dir = script_dir / "temp_distributed_work"
            chunks_dir = work_dir / "chunks"
            output_dir = work_dir / "processed_chunks"
            
            chunks_dir.mkdir(parents=True, exist_ok=True)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            chunk_file = chunks_dir / f"chunk_{chunk_number}.pkl"
            output_file = output_dir / f"processed_chunk_{chunk_number}.pkl"
            
            logger.info(f"Processing chunk {chunk_number}")
            logger.info(f"Input file: {chunk_file}")
            logger.info(f"Output file: {output_file}")
            
            if not chunk_file.exists():
                raise FileNotFoundError(f"Chunk file {chunk_file} does not exist")
                
            with open(chunk_file, 'rb') as f:
                chunk_data = pickle.load(f)
            
            latents = torch.tensor(chunk_data['chunk'], 
                                 dtype=torch.float16 if self.device == "cuda" else torch.float32,
                                 device=self.device)
            text_embeddings = torch.tensor(chunk_data['text_embeddings'], 
                                         dtype=torch.float16 if self.device == "cuda" else torch.float32,
                                         device=self.device)
            start_idx = chunk_data['start_idx']
            end_idx = chunk_data['end_idx']
            
            logger.info(f"Loaded chunk data: frames {start_idx}-{end_idx}, shape {latents.shape}")
            
            self.scheduler.set_timesteps(num_inference_steps)
            
            for i, t in enumerate(self.scheduler.timesteps):
                latent_model_input = torch.cat([latents] * 2)
                
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                
                
                with torch.no_grad():
                    noise_pred = self.unet(
                        latent_model_input, t, encoder_hidden_states=text_embeddings
                    ).sample
                
                
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + 7.5 * (noise_pred_text - noise_pred_uncond)
                
                
                latents = self.scheduler.step(noise_pred, t, latents).prev_sample
                
                if i % 5 == 0:
                    logger.info(f"Step {i+1}/{num_inference_steps}")
            
            # Ensure latents are in the correct precision for VAE decoding
            if self.device == "cuda":
                latents = latents.half()
            
            
            output_data = {
                'chunk': latents.detach().cpu().numpy(),
                'start_idx': start_idx,
                'end_idx': end_idx
            }
            
            
            temp_output = output_file.with_suffix('.tmp')
            with open(temp_output, 'wb') as f:
                pickle.dump(output_data, f)
            
            
            os.rename(temp_output, output_file)
            
            logger.info(f"Successfully processed chunk and saved to {output_file}")
            
        except Exception as e:
            logger.error(f"Error processing chunk: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            raise


def main():
    """
    
    """
    parser = argparse.ArgumentParser(description="Distributed worker for video latent processing")
    parser.add_argument("--chunk-number", type=int, required=True, 
                        help="Number of the chunk to process")
    parser.add_argument("--model-id", type=str, default="cerspense/zeroscope_v2_576w",
                        help="Zeroscope model identifier")
    parser.add_argument("--steps", type=int, default=25,
                        help="Number of inference steps")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use (cuda or cpu)")
    
    args = parser.parse_args()
    
    
    worker = LatentChunkWorker(model_id=args.model_id, device=args.device)
    
    
    worker.process_chunk(
        chunk_number=args.chunk_number,
        num_inference_steps=args.steps
    )


if __name__ == "__main__":
    main() 