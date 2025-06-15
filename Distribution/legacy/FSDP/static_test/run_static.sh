#!/usr/bin/env bash
set -e


MODEL_ID="cerspense/zeroscope_v2_XL"
OUT_CSV="static_results.csv"


echo "world_size,static_vram_before_mb,static_vram_after_mb" > $OUT_CSV

for NGPU in 1 2 4; do
  echo "=== Running static FSDP on ${NGPU} GPU(s) ==="

  # Run the static benchmark and grab only the line starting with "world_size"
  LINE=$(torchrun \
    --standalone \
    --nproc_per_node=$NGPU \
    FSDP_static_only.py \
      --model_id $MODEL_ID \
    |& grep "^world_size")

  


  echo "$LINE" \
    | sed -E 's/world_size=([0-9]+), static_vram_before_mb=([0-9]+), static_vram_after_mb=([0-9]+)/\1,\2,\3/' \
    >> $OUT_CSV

  echo "-> Recorded: $LINE"
done


