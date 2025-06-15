#!/usr/bin/env python
"""
Test script for distributed video generation

This script tests the different components of the distributed video
generation pipeline to ensure they work properly.

Usage:
  python test_distributed_generation.py --test [coordinator|worker|miner]
"""

import os
import sys
import time
import argparse
import logging
import asyncio

logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_coordinator():
    """
    Test the distributed coordinator.
    """
    from distribution.distributed_coordinator import DistributedCoordinator
    
    prompt = "A spaceship flying through a nebula, cinematic, high quality"
    output_path = "test_coordinator_output.mp4"
    
    coordinator = DistributedCoordinator(
        num_chunks=2,
        overlap_frames=2
    )
    
    try:
        result_path = coordinator.generate_video(
            prompt=prompt,
            num_frames=16,
            height=256,
            width=256,
            num_inference_steps=20,
            output_path=output_path
        )
        logger.info(f"Video generated successfully: {result_path}")
    except Exception as e:
        logger.error(f"Error generating video: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())

def test_worker():
    """
    Test the distributed worker.
    """
    from distribution.zeroscope.distributed_worker import LatentChunkWorker
    from distribution.distributed_coordinator import DistributedCoordinator
    import pickle
    import torch
    
    coordinator = DistributedCoordinator(num_chunks=2, overlap_frames=2)
    
    os.makedirs("temp_test", exist_ok=True)
    
    prompt = "A rocket launching, cinematic, detailed"
    latents, text_embeddings = coordinator.prepare_latents(
        prompt=prompt, 
        num_frames=16, 
        height=256, 
        width=256
    )
    
    chunks = coordinator.split_latent(latents)
    
    chunk_file = "temp_test/test_chunk.pkl"
    output_file = "temp_test/test_result.pkl"
    
    save_data = {
        'chunk': chunks[0]['chunk'].detach().cpu().numpy(),
        'start_idx': chunks[0]['start_idx'],
        'end_idx': chunks[0]['end_idx'],
        'text_embeddings': text_embeddings.detach().cpu().numpy()
    }
    
    with open(chunk_file, 'wb') as f:
        pickle.dump(save_data, f)
    
    logger.info(f"Saved test chunk to {chunk_file}")
    
    worker = LatentChunkWorker()
    
    try:
        worker.process_chunk(
            chunk_file=chunk_file,
            output_file=output_file,
            num_inference_steps=20
        )
        logger.info(f"Worker processed chunk successfully: {output_file}")
    except Exception as e:
        logger.error(f"Error processing chunk: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())

async def test_miner():
    """
    Test the distributed miner integration.
    """
    from distribution.zeroscope.miner_integration import DistributedMiner
    from template.protocol import InferNet
    import base64

    miner = DistributedMiner(
        num_chunks=2,
        overlap_frames=2,
        use_multiprocessing=False
    )
    
    prompt = "A beautiful sunset over the ocean, cinematic, high quality"
    synapse = InferNet(
        text_prompt=prompt,
        width=256,
        height=256,
        num_frames=16,
        fps=8
    )
    
    try:
        logger.info("Starting video generation...")
        result = await miner.generate_video(synapse, use_distributed=True)
        
        if result.video_data_b64:
            video_bytes = base64.b64decode(result.video_data_b64)
            
            output_path = "test_miner_output.mp4"
            with open(output_path, "wb") as f:
                f.write(video_bytes)
            
            logger.info(f"Test successful! Generated video saved to {output_path}")
        else:
            logger.error("Test failed: No video data returned")
    except Exception as e:
        logger.error(f"Error generating video: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())

def main():
    """
    Main function.
    """
    parser = argparse.ArgumentParser(description="Test distributed video generation")
    parser.add_argument("--test", type=str, choices=["coordinator", "worker", "miner"], required=True,
                        help="Which component to test")
    
    args = parser.parse_args()
    
    if args.test == "coordinator":
        test_coordinator()
    elif args.test == "worker":
        test_worker()
    elif args.test == "miner":
        asyncio.run(test_miner())


if __name__ == "__main__":
    main() 