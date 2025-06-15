#!/usr/bin/env python3
# plot_fsdp_metrics.py - Plot metrics from FSDP benchmark results

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

def plot_results(results_file, output_dir=None):
    """Draw latency, VRAM and efficiency plots for the saved benchmark .npy."""
    # Load results
    results = np.load(results_file, allow_pickle=True).tolist()
    df = pd.DataFrame(results)
    
    # Set output directory
    if output_dir is None:
        output_dir = Path(results_file).stem + '_plots'
    Path(output_dir).mkdir(exist_ok=True)
    
    # Set style
    plt.style.use('default')
    
    # Memory Usage Analysis
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot VRAM usage by world size
    for metric in ['static_vram_mb', 'peak_vram_mb', 'end_vram_mb']:
        ax1.plot(df['world_size'], df[metric], marker='o', label=metric.replace('_', ' ').title())
    
    ax1.set_title('VRAM Usage by World Size')
    ax1.set_xlabel('World Size (Number of GPUs)')
    ax1.set_ylabel('VRAM Usage (MB)')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot VRAM usage by chunk size
    for metric in ['static_vram_mb', 'peak_vram_mb', 'end_vram_mb']:
        ax2.plot(df['chunk_size'], df[metric], marker='o', label=metric.replace('_', ' ').title())
    
    ax2.set_title('VRAM Usage by Chunk Size')
    ax2.set_xlabel('Chunk Size')
    ax2.set_ylabel('VRAM Usage (MB)')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/vram_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Latency Analysis
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot latency by world size
    ax1.plot(df['world_size'], df['latency_s'], marker='o', color='blue')
    ax1.set_title('Latency by World Size')
    ax1.set_xlabel('World Size (Number of GPUs)')
    ax1.set_ylabel('Latency (seconds)')
    ax1.grid(True, alpha=0.3)
    
    # Plot latency by chunk size
    ax2.plot(df['chunk_size'], df['latency_s'], marker='o', color='red')
    ax2.set_title('Latency by Chunk Size')
    ax2.set_xlabel('Chunk Size')
    ax2.set_ylabel('Latency (seconds)')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/latency_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    #  Efficiency Analysis
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Calculate frames per second
    df['fps'] = df['num_frames'] / df['latency_s']
    
    # Plot FPS by world size
    ax1.plot(df['world_size'], df['fps'], marker='o', color='green')
    ax1.set_title('Frames per Second by World Size')
    ax1.set_xlabel('World Size (Number of GPUs)')
    ax1.set_ylabel('Frames per Second')
    ax1.grid(True, alpha=0.3)
    
    # Plot FPS by chunk size
    ax2.plot(df['chunk_size'], df['fps'], marker='o', color='purple')
    ax2.set_title('Frames per Second by Chunk Size')
    ax2.set_xlabel('Chunk Size')
    ax2.set_ylabel('Frames per Second')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/efficiency_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    #  Memory Efficiency
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Calculate memory efficiency (frames per GB of peak VRAM)
    df['frames_per_gb'] = df['num_frames'] / (df['peak_vram_mb'] / 1024)
    
    # Plot memory efficiency by world size
    ax1.plot(df['world_size'], df['frames_per_gb'], marker='o', color='orange')
    ax1.set_title('Memory Efficiency by World Size')
    ax1.set_xlabel('World Size (Number of GPUs)')
    ax1.set_ylabel('Frames per GB of Peak VRAM')
    ax1.grid(True, alpha=0.3)
    
    # Plot memory efficiency by chunk size
    ax2.plot(df['chunk_size'], df['frames_per_gb'], marker='o', color='brown')
    ax2.set_title('Memory Efficiency by Chunk Size')
    ax2.set_xlabel('Chunk Size')
    ax2.set_ylabel('Frames per GB of Peak VRAM')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/memory_efficiency.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    summary = df.describe()
    summary.to_csv(f"{output_dir}/metrics_summary.csv")
    
    print("\nKey Performance Metrics:")
    print(f"Average Latency: {df['latency_s'].mean():.2f} seconds")
    print(f"Average Peak VRAM: {df['peak_vram_mb'].mean():.2f} MB")
    print(f"Average FPS: {df['fps'].mean():.2f}")
    print(f"Average Memory Efficiency: {df['frames_per_gb'].mean():.2f} frames/GB")
    print(f"\nDetailed metrics saved to: {output_dir}/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot FSDP benchmark metrics')
    parser.add_argument('results_file', help='Path to the .npy results file')
    parser.add_argument('--output_dir', help='Directory to save plots (default: results_file_stem_plots)')
    args = parser.parse_args()
    
    plot_results(args.results_file, args.output_dir) 