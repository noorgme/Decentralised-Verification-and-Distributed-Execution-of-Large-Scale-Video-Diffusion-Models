import os
os.environ["HF_HOME"] = "/vol/bitbucket/ne221/huggingface"
os.environ["TRANSFORMERS_CACHE"] = "/vol/bitbucket/ne221/huggingface/transformers"
os.environ["HF_HUB_CACHE"] = "/vol/bitbucket/ne221/huggingface/hub"




import torch
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.utils import export_to_video

pipe = DiffusionPipeline.from_pretrained("cerspense/zeroscope_v2_576w", torch_dtype=torch.float16)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_model_cpu_offload()

prompt = "Luke Skywalker is on a pink hot air balloon"
video_frames = pipe(prompt, num_inference_steps=40, height=320, width=576, num_frames=24).frames

video_frames = video_frames[0]

video_path = export_to_video(video_frames, output_video_path="./output.mp4")
print(f"Saved video at: {video_path}")

