#!/bin/bash

# Check for Hugging Face token
if [ -z "$HF_TOKEN" ]; then
    echo "Error: HF_TOKEN environment variable is not set"
    echo "You can get a token from https://huggingface.co/settings/tokens"
    exit 1
fi

# First download the model locally
echo "Downloading model locally first..."
python3 download_model.py

# Check if download was successful
if [ $? -ne 0 ]; then
    echo "Failed to download model. Please check your token and try again."
    exit 1
fi


# FSDP + Latent Chunking + Coherence Optimisations. World-sizes:1-6.
## Wi-Fi
# torchrun --nproc_per_node=1 fsdp_chunked_coherent.py --emu_bw_mbps 500 --emu_rtt_ms 3.01 --emu_jitter_ms 3.53 --model_id "cerspense/zeroscope_v2_XL" --out_csv "ZeroscopeXL/wifi_results.csv"

torchrun --nproc_per_node=1 fsdp_chunked.py --emu_bw_mbps 500 --emu_rtt_ms 3.01 --emu_jitter_ms 3.53 --model_id "cerspense/zeroscope_v2_XL" --out_csv "ZeroscopeXL/wifi_results.csv"
# # torchrun --nproc_per_node=2 fsdp_chunked_coherent.py --emu_bw_mbps 500 --emu_rtt_ms 3.01 --emu_jitter_ms 3.53 --model_id "cerspense/zeroscope_v2_XL" --out_csv "ZeroscopeXL/wifi_results.csv"
# torchrun --nproc_per_node=3 fsdp_chunked_coherent.py --emu_bw_mbps 500 --emu_rtt_ms 3.01 --emu_jitter_ms 3.53 --model_id "cerspense/zeroscope_v2_XL" --out_csv "ZeroscopeXL/wifi_results.csv"
# torchrun --nproc_per_node=4 fsdp_chunked_coherent.py --emu_bw_mbps 500 --emu_rtt_ms 3.01 --emu_jitter_ms 3.53 --model_id "cerspense/zeroscope_v2_XL" --out_csv "ZeroscopeXL/wifi_results.csv"
# torchrun --nproc_per_node=5 fsdp_chunked_coherent.py --emu_bw_mbps 500 --emu_rtt_ms 3.01 --emu_jitter_ms 3.53 --model_id "cerspense/zeroscope_v2_XL" --out_csv "ZeroscopeXL/wifi_results.csv"
# torchrun --nproc_per_node=6 fsdp_chunked_coherent.py --emu_bw_mbps 500 --emu_rtt_ms 3.01 --emu_jitter_ms 3.53 --model_id "cerspense/zeroscope_v2_XL" --out_csv "ZeroscopeXL/wifi_results.csv"

# ## Gigabit Ethernet
torchrun --nproc_per_node=1 fsdp_chunked_coherent.py --emu_bw_mbps 1000 --emu_rtt_ms 0.12 --emu_jitter_ms 0.06 --model_id "cerspense/zeroscope_v2_XL" --out_csv "ZeroscopeXL/ethernet_results.csv"
# torchrun --nproc_per_node=2 fsdp_chunked_coherent.py --emu_bw_mbps 1000 --emu_rtt_ms 0.12 --emu_jitter_ms 0.06 --model_id "cerspense/zeroscope_v2_XL" --out_csv "ZeroscopeXL/ethernet_results.csv"
# torchrun --nproc_per_node=3 fsdp_chunked_coherent.py --emu_bw_mbps 1000 --emu_rtt_ms 0.12 --emu_jitter_ms 0.06 --model_id "cerspense/zeroscope_v2_XL" --out_csv "ZeroscopeXL/ethernet_results.csv"
# torchrun --nproc_per_node=4 fsdp_chunked_coherent.py --emu_bw_mbps 1000 --emu_rtt_ms 0.12 --emu_jitter_ms 0.06 --model_id "cerspense/zeroscope_v2_XL" --out_csv "ZeroscopeXL/ethernet_results.csv"
# torchrun --nproc_per_node=5 fsdp_chunked_coherent.py --emu_bw_mbps 1000 --emu_rtt_ms 0.12 --emu_jitter_ms 0.06 --model_id "cerspense/zeroscope_v2_XL" --out_csv "ZeroscopeXL/ethernet_results.csv"
# torchrun --nproc_per_node=6 fsdp_chunked_coherent.py --emu_bw_mbps 1000 --emu_rtt_ms 0.12 --emu_jitter_ms 0.06 --model_id "cerspense/zeroscope_v2_XL" --out_csv "ZeroscopeXL/ethernet_results.csv"


# # FSDP + Latent Chunking. World-sizes:1-6.
# ## Wi-Fi
torchrun --nproc_per_node=1 fsdp_chunked.py --emu_bw_mbps 500 --emu_rtt_ms 3.01 --emu_jitter_ms 3.53 --model_id "cerspense/zeroscope_v2_XL" --out_csv "ZeroscopeXL/wifi_results.csv"
# # torchrun --nproc_per_node=2 fsdp_chunked.py --emu_bw_mbps 500 --emu_rtt_ms 3.01 --emu_jitter_ms 3.53 --model_id "cerspense/zeroscope_v2_XL" --out_csv "ZeroscopeXL/wifi_results.csv"
# torchrun --nproc_per_node=3 fsdp_chunked.py --emu_bw_mbps 500 --emu_rtt_ms 3.01 --emu_jitter_ms 3.53 --model_id "cerspense/zeroscope_v2_XL" --out_csv "ZeroscopeXL/wifi_results.csv"
# torchrun --nproc_per_node=4 fsdp_chunked.py --emu_bw_mbps 500 --emu_rtt_ms 3.01 --emu_jitter_ms 3.53 --model_id "cerspense/zeroscope_v2_XL" --out_csv "ZeroscopeXL/wifi_results.csv"
# torchrun --nproc_per_node=5 fsdp_chunked.py --emu_bw_mbps 500 --emu_rtt_ms 3.01 --emu_jitter_ms 3.53 --model_id "cerspense/zeroscope_v2_XL" --out_csv "ZeroscopeXL/wifi_results.csv"
# torchrun --nproc_per_node=6 fsdp_chunked.py --emu_bw_mbps 500 --emu_rtt_ms 3.01 --emu_jitter_ms 3.53 --model_id "cerspense/zeroscope_v2_XL" --out_csv "ZeroscopeXL/wifi_results.csv"

# ## Gigabit Ethernet
torchrun --nproc_per_node=1 fsdp_chunked.py --emu_bw_mbps 1000 --emu_rtt_ms 0.12 --emu_jitter_ms 0.06 --model_id "cerspense/zeroscope_v2_XL" --out_csv "ZeroscopeXL/ethernet_results.csv"
# # torchrun --nproc_per_node=2 fsdp_chunked.py --emu_bw_mbps 1000 --emu_rtt_ms 0.12 --emu_jitter_ms 0.06 --model_id "cerspense/zeroscope_v2_XL" --out_csv "ZeroscopeXL/ethernet_results.csv"
# torchrun --nproc_per_node=3 fsdp_chunked.py --emu_bw_mbps 1000 --emu_rtt_ms 0.12 --emu_jitter_ms 0.06 --model_id "cerspense/zeroscope_v2_XL" --out_csv "ZeroscopeXL/ethernet_results.csv"
# torchrun --nproc_per_node=4 fsdp_chunked.py --emu_bw_mbps 1000 --emu_rtt_ms 0.12 --emu_jitter_ms 0.06 --model_id "cerspense/zeroscope_v2_XL" --out_csv "ZeroscopeXL/ethernet_results.csv"
# torchrun --nproc_per_node=5 fsdp_chunked.py --emu_bw_mbps 1000 --emu_rtt_ms 0.12 --emu_jitter_ms 0.06 --model_id "cerspense/zeroscope_v2_XL" --out_csv "ZeroscopeXL/ethernet_results.csv"
# torchrun --nproc_per_node=6 fsdp_chunked.py --emu_bw_mbps 1000 --emu_rtt_ms 0.12 --emu_jitter_ms 0.06 --model_id "cerspense/zeroscope_v2_XL" --out_csv "ZeroscopeXL/ethernet_results.csv"

# # FSDP. World-sizes:1-6.
# ## Wi-Fi

torchrun --nproc_per_node=1 fsdp.py --emu_bw_mbps 500 --emu_rtt_ms 3.01 --emu_jitter_ms 3.53 --model_id "cerspense/zeroscope_v2_XL" --out_csv "ZeroscopeXL/wifi_results.csv"
# # torchrun --nproc_per_node=2 fsdp.py --emu_bw_mbps 500 --emu_rtt_ms 3.01 --emu_jitter_ms 3.53 --model_id "cerspense/zeroscope_v2_XL" --out_csv "ZeroscopeXL/wifi_results.csv"
# torchrun --nproc_per_node=3 fsdp.py --emu_bw_mbps 500 --emu_rtt_ms 3.01 --emu_jitter_ms 3.53 --model_id "cerspense/zeroscope_v2_XL" --out_csv "ZeroscopeXL/wifi_results.csv"
# torchrun --nproc_per_node=4 fsdp.py --emu_bw_mbps 500 --emu_rtt_ms 3.01 --emu_jitter_ms 3.53 --model_id "cerspense/zeroscope_v2_XL" --out_csv "ZeroscopeXL/wifi_results.csv"
# torchrun --nproc_per_node=5 fsdp.py --emu_bw_mbps 500 --emu_rtt_ms 3.01 --emu_jitter_ms 3.53 --model_id "cerspense/zeroscope_v2_XL" --out_csv "ZeroscopeXL/wifi_results.csv"
# torchrun --nproc_per_node=6 fsdp.py --emu_bw_mbps 500 --emu_rtt_ms 3.01 --emu_jitter_ms 3.53 --model_id "cerspense/zeroscope_v2_XL" --out_csv "ZeroscopeXL/wifi_results.csv"

# ## Gigabit Ethernet
torchrun --nproc_per_node=1 fsdp.py --emu_bw_mbps 1000 --emu_rtt_ms 0.12 --emu_jitter_ms 0.06 --model_id "cerspense/zeroscope_v2_XL" --out_csv "ZeroscopeXL/ethernet_results.csv"
# # torchrun --nproc_per_node=2 fsdp.py --emu_bw_mbps 1000 --emu_rtt_ms 0.12 --emu_jitter_ms 0.06 --model_id "cerspense/zeroscope_v2_XL" --out_csv "ZeroscopeXL/ethernet_results.csv"
# torchrun --nproc_per_node=3 fsdp.py --emu_bw_mbps 1000 --emu_rtt_ms 0.12 --emu_jitter_ms 0.06 --model_id "cerspense/zeroscope_v2_XL" --out_csv "ZeroscopeXL/ethernet_results.csv"
# torchrun --nproc_per_node=4 fsdp.py --emu_bw_mbps 1000 --emu_rtt_ms 0.12 --emu_jitter_ms 0.06 --model_id "cerspense/zeroscope_v2_XL" --out_csv "ZeroscopeXL/ethernet_results.csv"
# torchrun --nproc_per_node=5 fsdp.py --emu_bw_mbps 1000 --emu_rtt_ms 0.12 --emu_jitter_ms 0.06 --model_id "cerspense/zeroscope_v2_XL" --out_csv "ZeroscopeXL/ethernet_results.csv"
# torchrun --nproc_per_node=6 fsdp.py --emu_bw_mbps 1000 --emu_rtt_ms 0.12 --emu_jitter_ms 0.06 --model_id "cerspense/zeroscope_v2_XL" --out_csv "ZeroscopeXL/ethernet_results.csv"


# # Latent Chunking. World-sizes:1-6.
# ## Wi-Fi
torchrun --nproc_per_node=1 chunk_only.py --emu_bw_mbps 500 --emu_rtt_ms 3.01 --emu_jitter_ms 3.53 --model_id "cerspense/zeroscope_v2_XL" --out_csv "ZeroscopeXL/wifi_results.csv"
# # torchrun --nproc_per_node=2 chunk_only.py --emu_bw_mbps 500 --emu_rtt_ms 3.01 --emu_jitter_ms 3.53 --model_id "cerspense/zeroscope_v2_XL" --out_csv "ZeroscopeXL/wifi_results.csv"
# torchrun --nproc_per_node=3 chunk_only.py --emu_bw_mbps 500 --emu_rtt_ms 3.01 --emu_jitter_ms 3.53 --model_id "cerspense/zeroscope_v2_XL" --out_csv "ZeroscopeXL/wifi_results.csv"
# torchrun --nproc_per_node=4 chunk_only.py --emu_bw_mbps 500 --emu_rtt_ms 3.01 --emu_jitter_ms 3.53 --model_id "cerspense/zeroscope_v2_XL" --out_csv "ZeroscopeXL/wifi_results.csv"
# torchrun --nproc_per_node=5 chunk_only.py --emu_bw_mbps 500 --emu_rtt_ms 3.01 --emu_jitter_ms 3.53 --model_id "cerspense/zeroscope_v2_XL" --out_csv "ZeroscopeXL/wifi_results.csv"
# torchrun --nproc_per_node=6 chunk_only.py --emu_bw_mbps 500 --emu_rtt_ms 3.01 --emu_jitter_ms 3.53 --model_id "cerspense/zeroscope_v2_XL" --out_csv "ZeroscopeXL/wifi_results.csv"

# # ## Gigabit Ethernet
torchrun --nproc_per_node=1 chunk_only.py --emu_bw_mbps 1000 --emu_rtt_ms 0.12 --emu_jitter_ms 0.06 --model_id "cerspense/zeroscope_v2_XL" --out_csv "ZeroscopeXL/ethernet_results.csv"
# torchrun --nproc_per_node=2 chunk_only.py --emu_bw_mbps 1000 --emu_rtt_ms 0.12 --emu_jitter_ms 0.06 --model_id "cerspense/zeroscope_v2_XL" --out_csv "ZeroscopeXL/ethernet_results.csv"
# torchrun --nproc_per_node=3 chunk_only.py --emu_bw_mbps 1000 --emu_rtt_ms 0.12 --emu_jitter_ms 0.06 --model_id "cerspense/zeroscope_v2_XL" --out_csv "ZeroscopeXL/ethernet_results.csv"
# torchrun --nproc_per_node=4 chunk_only.py --emu_bw_mbps 1000 --emu_rtt_ms 0.12 --emu_jitter_ms 0.06 --model_id "cerspense/zeroscope_v2_XL" --out_csv "ZeroscopeXL/ethernet_results.csv"
# torchrun --nproc_per_node=5 chunk_only.py --emu_bw_mbps 1000 --emu_rtt_ms 0.12 --emu_jitter_ms 0.06 --model_id "cerspense/zeroscope_v2_XL" --out_csv "ZeroscopeXL/ethernet_results.csv"
# torchrun --nproc_per_node=6 chunk_only.py --emu_bw_mbps 1000 --emu_rtt_ms 0.12 --emu_jitter_ms 0.06 --model_id "cerspense/zeroscope_v2_XL" --out_csv "ZeroscopeXL/ethernet_results.csv"