# fsdp_static.py
import os, argparse, torch, logging, pynvml
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, CPUOffload, MixedPrecision
from torch.distributed.fsdp.wrap import wrap
from diffusers import DiffusionPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("FSDPStatic")

# bind each rank to its own GPU
local_rank = int(os.environ.get("LOCAL_RANK", 0))
torch.cuda.set_device(local_rank)
device = torch.device(f"cuda:{local_rank}")

def measure_vram_mb():
    handle = pynvml.nvmlDeviceGetHandleByIndex(local_rank)
    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    return info.used // 1024**2

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, required=True)
    args = parser.parse_args()

    # init and NVML
    dist.init_process_group("nccl")
    pynvml.nvmlInit()

    # measure before loading
    before = measure_vram_mb()

    # load with low_cpu_mem_usage to reduce peak
    pipe = DiffusionPipeline.from_pretrained(
        args.model_id,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map=None  # we’ll shard manually
    )

    # wrap UNet & text encoder
    mp = MixedPrecision(param_dtype=torch.float16, reduce_dtype=torch.float16, buffer_dtype=torch.float16)
    unet = FSDP(wrap(pipe.unet), cpu_offload=CPUOffload(offload_params=False), mixed_precision=mp)
    te   = FSDP(wrap(pipe.text_encoder), cpu_offload=CPUOffload(offload_params=False), mixed_precision=mp)

    # move shards to GPU
    unet = unet.to(device)
    te   = te.to(device)
    torch.cuda.synchronize()

    # measure after loading
    after = measure_vram_mb()

    # only rank0 prints/saves
    if dist.get_rank() == 0:
        print(f"world_size={dist.get_world_size()}, static_vram_before_mb={before}, static_vram_after_mb={after}")
