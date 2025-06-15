#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
import logging


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_results(results_file: str) -> dict:

    with np.load(results_file) as data:
        results = {k: v for k, v in data.items()}
    return results

def plot_metric_comparison(results: dict, metric: str, output_dir: Path):

    plt.figure(figsize=(10, 6))
    

    methods = ['shared_noise', 'independent_noise']
    data = [results[f'{method}_{metric}'] for method in methods]
    

    plt.boxplot(data, labels=methods)
    plt.title(f'Comparison of {metric.replace("_", " ").title()}')
    plt.ylabel('Score')
    

    output_path = output_dir / f'{metric}_comparison.png'
    plt.savefig(output_path)
    plt.close()
    logger.info(f"Saved {metric} comparison plot to {output_path}")

def plot_metric_distribution(results: dict, metric: str, output_dir: Path):

    plt.figure(figsize=(10, 6))
    

    methods = ['shared_noise', 'independent_noise']
    for method in methods:
        sns.kdeplot(results[f'{method}_{metric}'], label=method)
    
    plt.title(f'Distribution of {metric.replace("_", " ").title()}')
    plt.xlabel('Score')
    plt.ylabel('Density')
    plt.legend()
    

    output_path = output_dir / f'{metric}_distribution.png'
    plt.savefig(output_path)
    plt.close()
    logger.info(f"Saved {metric} distribution plot to {output_path}")

def generate_report(results: dict, output_dir: Path):

    report_path = output_dir / 'benchmark_report.txt'
    
    with open(report_path, 'w') as f:
        f.write("Benchmark Results Report\n")
        f.write("======================\n\n")
        
        for method in ['shared_noise', 'independent_noise']:
            f.write(f"{method.replace('_', ' ').title()} Results:\n")
            f.write("-" * 30 + "\n")
            
            for metric in ['flow_consistency', 'overlap_similarity', 'generation_time']:
                values = results[f'{method}_{metric}']
                mean = np.mean(values)
                std = np.std(values)
                
                f.write(f"{metric.replace('_', ' ').title()}:\n")
                f.write(f"  Mean: {mean:.4f}\n")
                f.write(f"  Std: {std:.4f}\n")
                f.write(f"  Min: {np.min(values):.4f}\n")
                f.write(f"  Max: {np.max(values):.4f}\n\n")
        

        f.write("\nStatistical Significance:\n")
        f.write("-" * 30 + "\n")
        
        for metric in ['flow_consistency', 'overlap_similarity', 'generation_time']:
            shared = results[f'shared_noise_{metric}']
            independent = results[f'independent_noise_{metric}']
            

            from scipy import stats
            t_stat, p_value = stats.ttest_ind(shared, independent)
            
            f.write(f"{metric.replace('_', ' ').title()}:\n")
            f.write(f"  t-statistic: {t_stat:.4f}\n")
            f.write(f"  p-value: {p_value:.4f}\n")
            f.write(f"  Significant difference: {'Yes' if p_value < 0.05 else 'No'}\n\n")
    
    logger.info(f"Generated report at {report_path}")

def main():
    parser = argparse.ArgumentParser(description="Analyze benchmark results")
    parser.add_argument("--results", type=str, required=True,
                        help="Path to benchmark results .npz file")
    parser.add_argument("--output-dir", type=str, default="benchmark_analysis",
                        help="Directory to save analysis results")
    
    args = parser.parse_args()
    

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    

    results = load_results(args.results)
    

    for metric in ['flow_consistency', 'overlap_similarity', 'generation_time']:
        plot_metric_comparison(results, metric, output_dir)
        plot_metric_distribution(results, metric, output_dir)
    

    generate_report(results, output_dir)
    
    logger.info("Analysis complete! Results saved to %s", output_dir)

if __name__ == "__main__":
    main() 