#!/usr/bin/env bash
set -e


pip install torch torchvision diffusers transformers pynvml opencv-python accelerate


PROMPT="A rocket launching into space, cinematic, detailed, 4K"
NUM_FRAMES=8
STEPS=25
MODEL_ID="cerspense/zeroscope_v2_576w"



for NGPU in 1 2 4; do
  OUTFILE="fsdp_${NGPU}gpus.npy"
  echo " Running FSDP benchmark with ${NGPU} GPU(s)"
  torchrun --nproc_per_node=${NGPU} FSDP_experiment.py \
    --prompt "${PROMPT}" \
    --num_frames ${NUM_FRAMES} \
    --steps ${STEPS} \
    --model_id ${MODEL_ID} \
    --out ${OUTFILE} 2>&1 | tee fsdp_debug.log

  echo "-> Results saved to ${OUTFILE}"
done

echo
echo "All FSDP static-memory benchmarks complete."
