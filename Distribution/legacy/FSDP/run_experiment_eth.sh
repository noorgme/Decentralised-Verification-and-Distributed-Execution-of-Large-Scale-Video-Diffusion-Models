HOST=$(hostname -I | awk '{print $1}')

# Launch 4 worker servers, one per GPU, on ports 8000-8003
for i in 0 1 2 3; do
  export CUDA_VISIBLE_DEVICES=$i
  python distributed_strategy_c.py \
    --role worker \
    --host $HOST \
    --port $((8000 + i)) \
    --workers $(for p in {0..3}; do echo -n "127.0.0.1:$((8000+p)),"; done | sed 's/,$//') \
    --device cuda &  
done


sleep 5


export CUDA_VISIBLE_DEVICES=0
python distributed_strategy_c.py \
  --role coordinator \
  --host $HOST \
  --port 8000 \
  --workers $(for p in {0..3}; do echo -n "127.0.0.1:$((8000+p)),"; done | sed 's/,$//') \
  --prompt "A rocket launching into space" \
  --num_frames 16 \
  --chunk_size 8 \
  --overlap 2 \
  --pre_steps 3 \
  --steps 25 \
  --device cuda
