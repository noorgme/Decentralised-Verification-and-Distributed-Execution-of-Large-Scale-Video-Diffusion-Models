"""Quick helpers for chopping up video latents, running them on different GPUs or
processes, then stitching the results back together.  Written for the Zeroscope
pipeline but generic enough for other diffusion set-ups."""

import os
import time
import torch
import numpy as np
from typing import List, Tuple, Dict, Optional, Union, Callable
import multiprocessing as mp
from diffusers import DiffusionPipeline
from diffusers.utils import export_to_video
import logging
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LatentChunker:
    """
    Manages the chunking, distributed processing, and stitching of video latents.
    """
    
    def __init__(self, 
                 num_chunks: int = 2, 
                 overlap_frames: int = 2,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """Initialise the helper.

        Args:
        - num_chunks: How many chunks to make
        - overlap_frames: Frames shared between adjacent chunks
        - device: 'cuda' or 'cpu'
        """
        self.num_chunks = num_chunks
        self.overlap_frames = overlap_frames
        self.device = device
        logger.info(f"Initialised LatentChunker with {num_chunks} chunks, "
                   f"{overlap_frames} overlap frames on {device}")
    
    def split_latent(self, latent: torch.Tensor) -> List[Dict]:
        """
        Split a latent tensor into overlapping chunks along the frame dimension.
        
        Args:
            latent: Tensor of shape [batch, channel, frames, height, width]
            
        Returns:
        - 'chunk': The latent chunk tensor
        - 'start_idx': Starting frame index in the original latent
        - 'end_idx': Ending frame index in the original latent
        """
        # Extract dimensions
        batch_size, channels, num_frames, height, width = latent.shape
        logger.info(f"Splitting latent of shape {latent.shape} into {self.num_chunks} chunks")
        
        # Calculate frames per chunk (accounting for overlap)
        total_overlap = self.overlap_frames * (self.num_chunks - 1)
        effective_frames = num_frames + total_overlap
        frames_per_chunk = effective_frames // self.num_chunks
        
        # Ensure we have enough frames for the requested chunking
        if frames_per_chunk <= self.overlap_frames:
            raise ValueError(f"Cannot create {self.num_chunks} chunks with "
                            f"{self.overlap_frames} overlap frames from {num_frames} total frames. "
                            f"Reduce num_chunks or overlap_frames.")
        
        chunks = []
        for i in range(self.num_chunks):
            # Calculate start and end indices for this chunk
            start_idx = max(0, i * frames_per_chunk - (i * self.overlap_frames))
            end_idx = min(num_frames, start_idx + frames_per_chunk)
            
            # Handle boundary conditions
            if i == self.num_chunks - 1:
                end_idx = num_frames  # Ensure the last chunk goes to the end
            
            # Extract the chunk
            chunk = latent[:, :, start_idx:end_idx, :, :]
            
            chunks.append({
                'chunk': chunk,
                'start_idx': start_idx,
                'end_idx': end_idx
            })
            
            logger.info(f"Created chunk {i+1}/{self.num_chunks}: frames {start_idx}-{end_idx} "
                       f"(shape: {chunk.shape})")
        
        return chunks
    
    def stitch_chunks(self, 
                      processed_chunks: List[Dict], 
                      original_shape: Tuple[int, ...]) -> torch.Tensor:
        """
        Stitch processed chunks back together, averaging the overlapping frames.
        
    """
        batch_size, channels, num_frames, height, width = original_shape
        logger.info(f"Stitching {len(processed_chunks)} chunks to shape {original_shape}")
        
        # Initialise the output tensor and a weight tensor for averaging
        output = torch.zeros(original_shape, device=self.device)
        weights = torch.zeros((num_frames,), device=self.device)
        
        # Add each chunk to the output with appropriate weighting
        for chunk_info in processed_chunks:
            chunk = chunk_info['chunk']
            start_idx = chunk_info['start_idx']
            end_idx = chunk_info['end_idx']
            
            # Create weights for smooth blending
            # For overlapping regions, use a linear ramp to blend chunks
            chunk_weights = torch.ones((end_idx - start_idx,), device=self.device)
            
            # Add the weighted chunk to the output
            output[:, :, start_idx:end_idx, :, :] += chunk * chunk_weights.view(1, 1, -1, 1, 1)
            weights[start_idx:end_idx] += chunk_weights
        
        # Normalise by weights to get the average
        # Add a small epsilon to avoid division by zero
        weights = weights.view(1, 1, -1, 1, 1)
        output = output / (weights + 1e-8)
        
        logger.info(f"Successfully stitched chunks with shape {output.shape}")
        return output


class DistributedZeroscopeGenerator:
    """
    Implements distributed video generation using Zeroscope and latent chunking.
    """
    
    def __init__(self,
                 model_id: str = "cerspense/zeroscope_v2_576w",
                 num_chunks: int = 2,
                 overlap_frames: int = 2,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu",
                 use_multiprocessing: bool = False):
        """
        Initialise the distributed video generator.
        
        """
        self.model_id = model_id
        self.device = device
        self.use_multiprocessing = use_multiprocessing
        
        # Initialise the Zeroscope pipeline
        logger.info(f"Loading Zeroscope model {model_id} on {device}")
        self.pipe = DiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            use_safetensors=False
        ).to(device)
        
        # Initialise the latent chunker
        self.chunker = LatentChunker(num_chunks=num_chunks, 
                                     overlap_frames=overlap_frames,
                                     device=device)
        
        # Identify components for direct access
        try:
            self.vae = self.pipe.vae
            self.unet = self.pipe.unet
            self.text_encoder = self.pipe.text_encoder
            self.tokenizer = self.pipe.tokenizer
            self.scheduler = self.pipe.scheduler
            logger.info("Successfully extracted pipeline components")
        except AttributeError as e:
            logger.error(f"Failed to extract pipeline components: {e}")
            raise
    
    def _prepare_latents(self, prompt: str, num_frames: int, height: int, width: int) -> torch.Tensor:
        """
        Prepare the initial noise latents and encode the prompt.

        """
        # Encode text prompt
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        )
        text_embeddings = self.text_encoder(text_inputs.input_ids.to(self.device))[0]
        
        # Create unconditional embeddings for classifier-free guidance
        max_length = text_inputs.input_ids.shape[-1]
        uncond_input = self.tokenizer(
            [""] * 1, padding="max_length", max_length=max_length, return_tensors="pt"
        )
        uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]
        
        # Concatenate the conditional and unconditional embeddings
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        
        # Create random noise latents
        latents = torch.randn(
            (1, 4, num_frames, height // 8, width // 8),
            device=self.device,
            dtype=torch.float16 if self.device == "cuda" else torch.float32,
        )
        
        # Scale the latents
        latents = latents * self.scheduler.init_noise_sigma
        
        return latents, text_embeddings
    
    def _process_chunk(self, 
                       chunk_info: Dict, 
                       text_embeddings: torch.Tensor, 
                       num_inference_steps: int) -> Dict:
        """
        Process a single latent chunk by running denoising steps.
        
        Args:
            chunk_info: Dictionary containing the chunk and its indices
            text_embeddings: Encoded text prompt embeddings
            num_inference_steps: Number of denoising steps
            
        Returns:
            Updated chunk_info with processed chunk
        """
        latents = chunk_info['chunk']
        start_idx = chunk_info['start_idx']
        end_idx = chunk_info['end_idx']
        
        logger.info(f"Processing chunk frames {start_idx}-{end_idx} with shape {latents.shape}")
        
        # Initialise the scheduler
        self.scheduler.set_timesteps(num_inference_steps)
        
        # Denoising loop
        for i, t in enumerate(self.scheduler.timesteps):
            # Expand the latents for classifier-free guidance
            latent_model_input = torch.cat([latents] * 2)
            
            # Add time dimension to the latents (if needed)
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
            
            # Predict the noise residual with the UNet
            with torch.no_grad():
                noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
            
            # Perform guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + 7.5 * (noise_pred_text - noise_pred_uncond)
            
            # Compute the denoised latents
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample
            
            if i % 5 == 0:
                logger.info(f"Chunk {start_idx}-{end_idx}: Completed step {i+1}/{len(self.scheduler.timesteps)}")
        
        # Update the chunk info with the processed latents
        chunk_info['chunk'] = latents
        return chunk_info
    
    def _process_chunks_serial(self, 
                              chunks: List[Dict], 
                              text_embeddings: torch.Tensor, 
                              num_inference_steps: int) -> List[Dict]:
        """
        Process all chunks sequentially.
        
        Args:
            chunks: List of chunk dictionaries
            text_embeddings: Encoded text prompt embeddings
            num_inference_steps: Number of denoising steps
            
        Returns:
            List of processed chunk dictionaries
        """
        processed_chunks = []
        for i, chunk_info in enumerate(chunks):
            logger.info(f"Processing chunk {i+1}/{len(chunks)} serially")
            processed_chunk = self._process_chunk(chunk_info, text_embeddings, num_inference_steps)
            processed_chunks.append(processed_chunk)
        return processed_chunks
    
    def _process_chunks_parallel(self, 
                                chunks: List[Dict], 
                                text_embeddings: torch.Tensor, 
                                num_inference_steps: int) -> List[Dict]:
        """
        Process all chunks in parallel using multiprocessing.
        
        NOTE: This is a placeholder for a more sophisticated distributed processing
        implementation. In practice, multiple processes or even machines would handle
        different chunks.
        
        Args:
            chunks: List of chunk dictionaries
            text_embeddings: Encoded text prompt embeddings
            num_inference_steps: Number of denoising steps
            
        Returns:
            List of processed chunk dictionaries
        """
        # For now, we'll use the serial implementation
        # In a real implementation, you would use multiprocessing.Pool or
        # a distributed computing framework
        logger.warning("Parallel processing not fully implemented. Using serial processing.")
        return self._process_chunks_serial(chunks, text_embeddings, num_inference_steps)
    
    def generate_video(self, 
                       prompt: str, 
                       num_frames: int = 16, 
                       height: int = 320, 
                       width: int = 576,
                       num_inference_steps: int = 25,
                       output_path: Optional[str] = None) -> Union[str, List[np.ndarray]]:
        """
        Generate a video using distributed latent processing.
        
        Args:
            prompt: Text prompt for the video
            num_frames: Number of frames to generate
            height: Height of the output video
            width: Width of the output video
            num_inference_steps: Number of denoising steps
            output_path: Path to save the video (if None, returns frames)
            
        Returns:
            Path to the output video or list of video frames
        """
        start_time = time.time()
        logger.info(f"Starting distributed video generation for prompt: '{prompt}'")
        
        try:
            # Prepare the initial latents and text embeddings
            latents, text_embeddings = self._prepare_latents(prompt, num_frames, height, width)
            
            # Split the latents into chunks
            chunks = self.chunker.split_latent(latents)
            
            # Process the chunks
            if self.use_multiprocessing:
                processed_chunks = self._process_chunks_parallel(chunks, text_embeddings, num_inference_steps)
            else:
                processed_chunks = self._process_chunks_serial(chunks, text_embeddings, num_inference_steps)
            
            # Stitch the processed chunks back together
            stitched_latents = self.chunker.stitch_chunks(processed_chunks, latents.shape)
            
            # Decode the latents to pixel space using the VAE
            logger.info("Decoding latents to pixel space")
            with torch.no_grad():
                video_frames = self.vae.decode(stitched_latents / 0.18215).sample
            
            # Rescale to [0, 1] and convert to uint8
            video_frames = (video_frames / 2 + 0.5).clamp(0, 1)
            video_frames = video_frames.cpu().permute(0, 2, 1, 3, 4).numpy() 
            
            # Save or return the video
            if output_path:
                logger.info(f"Saving video to {output_path}")
                if not os.path.exists(os.path.dirname(output_path)):
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                
                # Use diffusers utility to export frames to video
                temp_frames = (video_frames[0] * 255).astype(np.uint8)
                export_path = export_to_video(temp_frames, fps=8)
                
                # Rename the file to the desired output path
                os.rename(export_path, output_path)
                
                logger.info(f"Video saved to {output_path}")
                result = output_path
            else:
                # Return the frames directly
                result = video_frames
            
            elapsed_time = time.time() - start_time
            logger.info(f"Video generation completed in {elapsed_time:.2f} seconds")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in video generation: {str(e)}")
            logger.error(traceback.format_exc())
            raise


def test_distributed_generation():
    """
    Test function for the distributed video generator.
    """
    # Configure the distributed generator
    generator = DistributedZeroscopeGenerator(
        num_chunks=2,
        overlap_frames=2,
        use_multiprocessing=False
    )
    
    # Generate a video
    prompt = "A rocket launching into space, cinematic, detailed, 4K"
    output_path = os.path.join("generated_videos", "distributed_test.mp4")
    
    try:
        generator.generate_video(
            prompt=prompt,
            num_frames=16,
            height=320,
            width=576,
            num_inference_steps=25,
            output_path=output_path
        )
        print(f"Video generation successful! Saved to {output_path}")
    except Exception as e:
        print(f"Video generation failed: {str(e)}")


if __name__ == "__main__":
    # Run the test
    test_distributed_generation()
