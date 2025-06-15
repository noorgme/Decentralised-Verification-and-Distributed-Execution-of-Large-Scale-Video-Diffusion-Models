import pandas as pd
import matplotlib.pyplot as plt

wifi_csv = "ZeroscopeXL/wifi_results.csv"
eth_csv  = "ZeroscopeXL/ethernet_results.csv"

cols = [
    "timestamp","host","mode","world_size","num_frames",
    "chunk_size","overlap","latency_s","throughput_fps",
    "peak_vram_mb","end_vram_mb",
    "network_bytes","net_gather_s","net_reduce_s",
    "temp_instab","flow_err"
]

df_wifi = pd.read_csv(wifi_csv, names=cols)
df_wifi["network"] = "wifi"
df_eth  = pd.read_csv(eth_csv,  names=cols)
df_eth["network"]  = "ethernet"
print (df_wifi["mode"].unique())
print (df_eth["mode"].unique())
df = pd.concat([df_wifi, df_eth], ignore_index=True)
agg = (
    df
    .groupby(["network","mode","world_size"])
    .agg({
        "latency_s": "mean",
        "throughput_fps": "mean",
        "peak_vram_mb": "mean",
        "end_vram_mb": "mean",
        "temp_instab": "mean",
        "flow_err": "mean",
        "network_bytes": "mean",
        "net_gather_s": "mean",
        "net_reduce_s": "mean"
    })
    .reset_index()
)


metrics = [
    ("peak_vram_mb",    "Peak VRAM (MB)"),
    ("end_vram_mb",     "End VRAM (MB)"),
    ("latency_s",       "Latency (s)"),
    ("throughput_fps",  "Throughput (frames/s)"),
    ("temp_instab",     "Boundary Instability (L1)"),
    ("flow_err",        "Optical-Flow Warp Error"),
    ("network_bytes",   "Network Bytes Transferred"),
    ("net_gather_s",    "All-Gather Time (s)"),
    ("net_reduce_s",    "All-Reduce Time (s)")
]

for metric, ylabel in metrics:
    plt.figure()
    for network in ["wifi", "ethernet"]:
        for mode in ["fsdp", "chunk", "hybrid", "hybrid_ctx"]:
            sub = agg[(agg.network == network)& (agg["mode"] == mode)]
            print (sub)
            plt.plot(sub.world_size, sub[metric], "-o", label=f"{mode} ({network})")
    plt.title(f"{ylabel} vs World Size")
    plt.xlabel("World Size")
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    plt.savefig(f"ZeroscopeXL/{metric}.png")

    