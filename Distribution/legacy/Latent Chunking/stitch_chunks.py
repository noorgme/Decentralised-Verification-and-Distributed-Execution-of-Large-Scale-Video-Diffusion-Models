#!/usr/bin/env python
"""Stitch worker outputs into one video - handy if the coordinator fell
over mid-run."""

import os
import sys
import logging
import pickle
import torch
import numpy as np
from pathlib import Path
from diffusers import DiffusionPipeline
from diffusers.utils import export_to_video

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def stitch_chunks(processed_chunks: list, original_shape: tuple, device: str = "cuda") -> torch.Tensor:
    """
    Stitch processed chunks back together, averaging the overlapping frames.
    
    Inputs:
    - processed_chunks: List of dictionaries with processed chunks and their indices
    - original_shape: Shape of the original latent tensor to reconstruct
    - device: Device to use for processing
        
    Outputs:
    - Stitched tensor with the original shape
    """
    batch_size, channels, num_frames, height, width = original_shape
    logger.info(f"Stitching {len(processed_chunks)} chunks to shape {original_shape}")
    
    # Initialise the output tensor and a weight tensor for averaging
    output = torch.zeros(original_shape, device=device)
    weights = torch.zeros((num_frames,), device=device)
    
    # Add each chunk to the output with appropriate weighting
    for chunk_info in processed_chunks:
        chunk = chunk_info['chunk']
        start_idx = chunk_info['start_idx']
        end_idx = chunk_info['end_idx']
        
        # Create weights for smooth blending
        chunk_frames = end_idx - start_idx
        chunk_weights = torch.ones((chunk_frames,), device=device)
        
        # Apply the weights
        output[:, :, start_idx:end_idx, :, :] += chunk * chunk_weights.view(1, 1, -1, 1, 1)
        weights[start_idx:end_idx] += chunk_weights
    
    # Normalise by weights to get the average
    weights = weights.view(1, 1, -1, 1, 1)
    output = output / (weights + 1e-8)
    
    logger.info(f"Successfully stitched chunks into tensor of shape {output.shape}")
    return output

def main():
    # Hardcoded paths relative to this script's directory
    script_dir = Path(__file__).parent
    results_dir = script_dir / "temp_distributed_work" / "chunks"
    output_file = script_dir / "temp_distributed_work" / "output.mp4"
    model_id = "cerspense/zeroscope_v2_576w"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    
    output_dir = output_file.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Using results directory: {results_dir}")
    logger.info(f"Output file will be saved to: {output_file}")
    
    try:
        
        processed_chunks = []
        if not results_dir.exists():
            raise ValueError(f"Results directory {results_dir} does not exist")
            
        for filename in os.listdir(results_dir):
            if filename.endswith(".pkl"):
                filepath = results_dir / filename
                logger.info(f"Loading {filepath}")
                with open(filepath, 'rb') as f:
                    chunk_data = pickle.load(f)
                    processed_chunks.append(chunk_data)
        
        if not processed_chunks:
            raise ValueError(f"No .pkl files found in {results_dir}")
        
        # Sort chunks by start index
        processed_chunks.sort(key=lambda x: x['start_idx'])
        
        # Determine original shape from the chunks
        first_chunk = processed_chunks[0]['chunk']
        last_chunk = processed_chunks[-1]['chunk']
        num_frames = processed_chunks[-1]['end_idx']
        original_shape = (1, first_chunk.shape[1], num_frames, first_chunk.shape[3], first_chunk.shape[4])
        
        # Convert numpy arrays to tensors
        for chunk in processed_chunks:
            chunk['chunk'] = torch.tensor(chunk['chunk'], 
                                        dtype=torch.float16 if device == "cuda" else torch.float32,
                                        device=device)
        
        # Stitch the chunks
        stitched_latents = stitch_chunks(processed_chunks, original_shape, device)
        
        # Load the VAE
        logger.info(f"Loading VAE from {model_id}")
        pipe = DiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            use_safetensors=False
        ).to(device)
        
        # Decode the latents to pixel space
        logger.info("Decoding latents to pixel space")
        with torch.no_grad():
            # Convert stitched latents to float16 if using CUDA
            if device == "cuda":
                stitched_latents = stitched_latents.half()
            
            # Reshape latents for VAE decoding
            batch_size, channels, num_frames, height, width = stitched_latents.shape
            # Reshape to (batch * frames, channels, height, width)
            latents_reshaped = stitched_latents.permute(0, 2, 1, 3, 4).reshape(-1, channels, height, width)
            
            # Decode all frames at once
            video_frames = pipe.vae.decode(latents_reshaped / 0.18215).sample
            logger.info(f"VAE decoder output shape: {video_frames.shape}")
            
            # Reshape back to (batch, frames, channels, height, width)
            # The VAE decoder outputs (batch*frames, channels, height, width)
            # We need to reshape to (batch, frames, channels, height, width)
            video_frames = video_frames.reshape(batch_size, num_frames, -1, height * 8, width * 8)
            logger.info(f"After reshape: {video_frames.shape}")
            
            # Permute to (batch, channels, frames, height, width)
            video_frames = video_frames.permute(0, 2, 1, 3, 4)
            logger.info(f"After permute: {video_frames.shape}")
            
            # Verify the channel count
            if video_frames.shape[1] != 3:
                logger.error(f"VAE decoder produced {video_frames.shape[1]} channels, expected 3")
                raise ValueError(f"Unexpected number of channels: {video_frames.shape[1]}")
        
        # Post-process the frames
        video_frames = (video_frames / 2 + 0.5).clamp(0, 1)
        video_frames = video_frames.cpu().numpy()
        logger.info(f"After numpy conversion: {video_frames.shape}")
        
        # Ensure frames have 3 channels (RGB)
        if video_frames.shape[1] != 3:
            logger.warning(f"Frames have {video_frames.shape[1]} channels, converting to RGB")
            # If we have more than 3 channels, take the first 3
            if video_frames.shape[1] > 3:
                video_frames = video_frames[:, :3, :, :, :]
            # If we have less than 3 channels, repeat the last channel
            else:
                video_frames = np.repeat(video_frames, 3 // video_frames.shape[1], axis=1)
            logger.info(f"After channel adjustment: {video_frames.shape}")
        
        # Save the video
        logger.info(f"Saving video to {output_file}")
        
        # Use diffusers utility to export frames to video
        # Reshape from (channels, frames, height, width) to (frames, height, width, channels)
        temp_frames = (video_frames[0] * 255).astype(np.uint8)
        temp_frames = np.transpose(temp_frames, (1, 2, 3, 0))  # (frames, height, width, channels)
        logger.info(f"Final frames shape before export: {temp_frames.shape}")
        export_to_video(temp_frames, str(output_file), fps=8)
        
        logger.info("Successfully stitched and saved video!")
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main() 