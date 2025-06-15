#!/usr/bin/env python3
import argparse, numpy as np, pandas as pd, matplotlib.pyplot as plt
from pathlib import Path
from typing import List

METRICS_DEFAULT = [
    "latency_s",
    "peak_vram_mb",
    "end_vram_mb",
    "fps",
]

def load_metrics(dir_path: Path) -> pd.DataFrame:
    npys: List[Path] = list(dir_path.glob('*.npy'))
    if not npys:
        raise FileNotFoundError(f"No .npy metric files found in {dir_path}")
    records = []
    for npy in npys:
        arr = np.load(npy, allow_pickle=True)
        if arr.size == 0:
            continue
        # each file may be list of dicts
        for rec in arr.tolist():
            rec['source_file'] = npy.name
            records.append(rec)
    if not records:
        raise RuntimeError(f"No metric records found in {dir_path}")
    df = pd.DataFrame(records)
    # Derived metrics
    if 'num_frames' in df.columns and 'latency_s' in df.columns:
        df['fps'] = df['num_frames'] / df['latency_s'].replace(0, float('nan'))
    return df


def plot_metric(df_a: pd.DataFrame, df_b: pd.DataFrame, label_a: str, label_b: str,
                 metric: str, out_dir: Path):
    plt.figure(figsize=(8,5))
    plt.title(f"{metric} comparison")
    # world_size may vary, so plot vs world_size if present else index
    if 'world_size' in df_a.columns and 'world_size' in df_b.columns:
        # Scatter raw points
        plt.scatter(df_a['world_size'], df_a[metric], alpha=0.4, marker='o', label=f"{label_a} runs")
        plt.scatter(df_b['world_size'], df_b[metric], alpha=0.4, marker='s', label=f"{label_b} runs")

        # Mean per world size for line plot
        mean_a = df_a.groupby('world_size')[metric].mean().reset_index().sort_values('world_size')
        mean_b = df_b.groupby('world_size')[metric].mean().reset_index().sort_values('world_size')

        plt.plot(mean_a['world_size'], mean_a[metric], 'o-', linewidth=2, label=f"{label_a} mean")
        plt.plot(mean_b['world_size'], mean_b[metric], 's--', linewidth=2, label=f"{label_b} mean")
        plt.xlabel('World Size')
    else:
        plt.plot(df_a[metric], 'o-', label=label_a)
        plt.plot(df_b[metric], 's--', label=label_b)
        plt.xlabel('Run #')
    plt.ylabel(metric)
    plt.grid(alpha=0.3)
    plt.legend()
    out_path = out_dir / f"{metric}_compare.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved {out_path}")


def main():
    ap = argparse.ArgumentParser(description="Compare FSDP vs hybrid metrics")
    ap.add_argument('baseline_dir', help='Directory with baseline .npy metric files')
    ap.add_argument('hybrid_dir', help='Directory with hybrid .npy metric files')
    ap.add_argument('--metrics', nargs='+', default=METRICS_DEFAULT,
                    help='Metrics to plot')
    ap.add_argument('--out_dir', default='comparison_plots', help='Output directory')
    args = ap.parse_args()

    base_dir = Path(args.baseline_dir).expanduser().resolve()
    hybrid_dir= Path(args.hybrid_dir).expanduser().resolve()
    out_dir  = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(exist_ok=True)

    df_base = load_metrics(base_dir)
    df_hyb  = load_metrics(hybrid_dir)

    # Metrics list for table 
    table_metrics = [m for m in args.metrics if m in df_base.columns and m in df_hyb.columns]
    if 'world_size' in df_base.columns:
        df_base_tbl = df_base.groupby('world_size')[table_metrics].mean().reset_index().rename(columns={m:f"{m}_{base_dir.name}" for m in table_metrics})
        df_hyb_tbl  = df_hyb.groupby('world_size')[table_metrics].mean().reset_index().rename(columns={m:f"{m}_{hybrid_dir.name}" for m in table_metrics})
        table_df = pd.merge(df_base_tbl, df_hyb_tbl, on='world_size', how='outer').sort_values('world_size')
    else:
        # fallback: just mean of all runs
        row_base = df_base[table_metrics].mean().add_suffix(f"_{base_dir.name}")
        row_hyb  = df_hyb[table_metrics].mean().add_suffix(f"_{hybrid_dir.name}")
        table_df = pd.concat([row_base, row_hyb]).to_frame().T

    print("\n==== Summary Table ====")
    print(table_df.to_string(index=False, float_format=lambda x: f"{x:0.2f}"))
    (out_dir / "comparison_table.csv").write_text(table_df.to_csv(index=False))
    print(f"Table saved to {out_dir/'comparison_table.csv'}")

    for m in args.metrics:
        if m not in df_base.columns or m not in df_hyb.columns:
            print(f"Warning: metric '{m}' missing in one of the datasets, skipping")
            continue
        plot_metric(df_base, df_hyb, base_dir.name, hybrid_dir.name, m, out_dir)

    print(f"All plots saved to {out_dir}")

if __name__ == '__main__':
    main() 