#!/usr/bin/env python
"""
Benchmark script to compare different chunking strategies for distributed video generation.
Compares shared noise initialisation vs independent noise initialization.
"""

import os
import sys
import time
import logging
import argparse
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Tuple
import cv2
import torch.nn.functional as F
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ChunkingBenchmark:
    def __init__(self, 
                 num_frames: int = 16,
                 chunk_size: int = 8,
                 overlap: int = 2,
                 device: str = "cuda"):
        self.num_frames = num_frames
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.device = device
        
    def generate_noise(self, shared: bool = False) -> List[Dict]:
        """Generate noise for chunks with or without shared initialization."""
        if shared:
            # Generate base noise for the entire sequence
            base_noise = torch.randn((1, 4, self.num_frames, 64, 64), device=self.device)
            
            chunks = []
            for i in range(0, self.num_frames, self.chunk_size - self.overlap):
                start = i
                end = min(i + self.chunk_size, self.num_frames)
                
                # For overlapping regions, use the same noise
                if i > 0:
                    chunk_noise = base_noise[:, :, start:end].clone()
                    chunk_noise[:, :, :self.overlap] = base_noise[:, :, start:start+self.overlap]
                else:
                    chunk_noise = base_noise[:, :, start:end]
                    
                chunks.append({
                    'chunk': chunk_noise,
                    'start_idx': start,
                    'end_idx': end
                })
        else:
            # Generate independent noise for each chunk
            chunks = []
            for i in range(0, self.num_frames, self.chunk_size - self.overlap):
                start = i
                end = min(i + self.chunk_size, self.num_frames)
                chunk_noise = torch.randn((1, 4, end-start, 64, 64), device=self.device)
                
                chunks.append({
                    'chunk': chunk_noise,
                    'start_idx': start,
                    'end_idx': end
                })
        
        return chunks
    
    def compute_optical_flow(self, frames: torch.Tensor) -> torch.Tensor:
        """Compute optical flow between consecutive frames."""
        
        frames_np = (frames.permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)
        
        flows = []
        for i in range(len(frames_np) - 1):
            
            prev = cv2.cvtColor(frames_np[i], cv2.COLOR_RGB2GRAY)
            curr = cv2.cvtColor(frames_np[i+1], cv2.COLOR_RGB2GRAY)
            
            
            flow = cv2.calcOpticalFlowFarneback(
                prev, curr, None, 0.5, 3, 15, 3, 5, 1.2, 0
            )
            flows.append(flow)
        
        return torch.tensor(np.array(flows), device=self.device)
    
    def compute_flow_consistency(self, flows: torch.Tensor) -> float:
        """Compute consistency of optical flow between frames."""
        # Compute magnitude of flow differences between consecutive frames
        flow_diffs = torch.diff(flows, dim=0)
        flow_magnitudes = torch.norm(flow_diffs, dim=-1)
        
        # Lower magnitude differences indicate more consistent flow
        return float(1.0 / (1.0 + flow_magnitudes.mean()))
    
    def compute_overlap_similarity(self, chunks: List[Dict]) -> float:
        """Compute similarity between overlapping regions of chunks."""
        similarities = []
        for i in range(len(chunks) - 1):
            current_chunk = chunks[i]['chunk']
            next_chunk = chunks[i+1]['chunk']
            
            
            overlap_current = current_chunk[:, :, -self.overlap:]
            overlap_next = next_chunk[:, :, :self.overlap]
            
            
            similarity = F.mse_loss(overlap_current, overlap_next)
            similarities.append(float(similarity))
        
        return float(np.mean(similarities))
    
    def run_benchmark(self, num_runs: int = 5) -> Dict:
        """Run benchmark comparing shared and independent noise initialisation."""
        results = {
            'shared_noise': {
                'flow_consistency': [],
                'overlap_similarity': [],
                'generation_time': []
            },
            'independent_noise': {
                'flow_consistency': [],
                'overlap_similarity': [],
                'generation_time': []
            }
        }
        
        for method in ['shared_noise', 'independent_noise']:
            logger.info(f"Running benchmark for {method}")
            
            for run in tqdm(range(num_runs)):
                # Generate noise
                start_time = time.time()
                chunks = self.generate_noise(shared=(method == 'shared_noise'))
                generation_time = time.time() - start_time
                
                # Compute metrics
                overlap_similarity = self.compute_overlap_similarity(chunks)
                
                
                flows = self.compute_optical_flow(chunks[0]['chunk'][0])
                flow_consistency = self.compute_flow_consistency(flows)
                
                
                results[method]['flow_consistency'].append(flow_consistency)
                results[method]['overlap_similarity'].append(overlap_similarity)
                results[method]['generation_time'].append(generation_time)
        
        return results

def main():
    parser = argparse.ArgumentParser(description="Benchmark chunking strategies")
    parser.add_argument("--num-frames", type=int, default=16,
                        help="Number of frames in the video")
    parser.add_argument("--chunk-size", type=int, default=8,
                        help="Size of each chunk")
    parser.add_argument("--overlap", type=int, default=2,
                        help="Number of overlapping frames")
    parser.add_argument("--num-runs", type=int, default=5,
                        help="Number of runs for each method")
    parser.add_argument("--output", type=str, default="benchmark_results.npz",
                        help="Output file for results")
    
    args = parser.parse_args()
    
    
    benchmark = ChunkingBenchmark(
        num_frames=args.num_frames,
        chunk_size=args.chunk_size,
        overlap=args.overlap
    )
    
    
    results = benchmark.run_benchmark(num_runs=args.num_runs)
    
    
    np.savez(args.output, **results)
    logger.info(f"Results saved to {args.output}")
    
    
    for method in results:
        logger.info(f"\n{method} results:")
        for metric in results[method]:
            values = results[method][metric]
            logger.info(f"{metric}: mean={np.mean(values):.4f}, std={np.std(values):.4f}")

if __name__ == "__main__":
    main() 