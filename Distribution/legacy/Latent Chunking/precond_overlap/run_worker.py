#!/usr/bin/env python3
import os, sys, time, pickle, argparse, threading, socket
import numpy as np
import torch, torch.nn.functional as F
import xmlrpc.client
from xmlrpc.server import SimpleXMLRPCServer
from diffusers import DiffusionPipeline
from transformers import CLIPModel, CLIPProcessor
import pynvml, psutil, cv2

class WorkerService:
    def __init__(self, model_id, device):
        self.device = device
        
        self.pipe = DiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if device.startswith("cuda") else torch.float32,
            use_safetensors=False
        ).to(device)
        self.unet = self.pipe.unet
        self.scheduler = self.pipe.scheduler
        self.unet.eval()

    def process_chunk(self, args_pkl):
        
        args = pickle.loads(args_pkl.data)
        lat = torch.from_numpy(args['latents']).to(self.device)
        txt = torch.from_numpy(args['text_emb']).to(self.device)
        pre = args['pre_steps']
        total = args['num_steps']

        # Pre-denoise full latent for 'pre' steps (approx overlap-only)
        self.scheduler.set_timesteps(total, device=self.device)
        for t in self.scheduler.timesteps[:pre]:
            inp = torch.cat([lat]*2)
            inp = self.scheduler.scale_model_input(inp, t)
            with torch.no_grad():
                noise = self.unet(inp, t, encoder_hidden_states=txt).sample
            uncond, cond = noise.chunk(2)
            noise = uncond + 7.5*(cond-uncond)
            lat = self.scheduler.step(noise, t, lat).prev_sample

        # Finish denoising chunk
        for t in self.scheduler.timesteps[pre:]:
            inp = torch.cat([lat]*2)
            inp = self.scheduler.scale_model_input(inp, t)
            with torch.no_grad():
                noise = self.unet(inp, t, encoder_hidden_states=txt).sample
            uncond, cond = noise.chunk(2)
            noise = uncond + 7.5*(cond-uncond)
            lat = self.scheduler.step(noise, t, lat).prev_sample

        # output
        out = {
            'chunk': lat.detach().cpu().numpy(),
            'start_idx': args['start_idx'],
            'end_idx': args['end_idx']
        }
        return pickle.dumps(out)

def start_worker_server(host, port, model_id, device):
    server = SimpleXMLRPCServer((host, port), allow_none=True, logRequests=False)
    svc = WorkerService(model_id, device)
    server.register_function(svc.process_chunk, "process_chunk")
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f"[Worker] XML-RPC server listening on {host}:{port}")

class StrategyC_Coordinator:
    def __init__(self, workers, model_id, device):
        self.workers = [xmlrpc.client.ServerProxy(f"http://{h}", allow_none=True) for h in workers]
        self.device = device
        self._init_models(model_id)
        pynvml.nvmlInit()

    def _init_models(self, model_id):
        # zeroscope
        self.pipe = DiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if self.device.startswith("cuda") else torch.float32,
            use_safetensors=False
        ).to(self.device)
        self.unet = self.pipe.unet
        self.scheduler = self.pipe.scheduler
        self.tokenizer = self.pipe.tokenizer
        self.text_encoder = self.pipe.text_encoder
        self.vae = self.pipe.vae
        # CLIP
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        self.proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def _measure_net(self):
        c = psutil.net_io_counters()
        return c.bytes_sent + c.bytes_recv

    def _measure_vram(self):
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return info.used // 1024**2

    def prepare(self, prompt, num_frames, num_steps):
        # text
        tokens = self.tokenizer(prompt,
                                padding="max_length",
                                max_length=self.tokenizer.model_max_length,
                                truncation=True,
                                return_tensors="pt")
        with torch.no_grad():
            txt = self.text_encoder(tokens.input_ids.to(self.device))[0]
        # scheduler
        self.scheduler.set_timesteps(num_steps, device=self.device)
        # initial latents
        lat = torch.randn(
            (1, self.unet.config.in_channels, num_frames,
             self.vae.sample_size, self.vae.sample_size),
            device=self.device,
            dtype=torch.float16 if self.device.startswith("cuda") else torch.float32
        ) * self.scheduler.init_noise_sigma
        return lat, txt

    def _decode(self, stitched):
        # decode latent to video frames
        b,c,f,h,w = stitched.shape
        lat = stitched.permute(0,2,1,3,4).reshape(-1,c,h,w)
        lat = lat.half() if self.device.startswith("cuda") else lat
        with torch.no_grad():
            vf = self.vae.decode(lat/0.18215).sample
        vf = vf.reshape(b,f,-1,h*8,w*8).permute(0,2,1,3,4)
        return (vf/2+0.5).clamp(0,1)

    def _flow_consistency(self, video):
        # video: (1,C,F,H,W)
        frames = video[0].permute(1,2,3,0).cpu().numpy()*255
        flows=[]
        for i in range(len(frames)-1):
            g0 = cv2.cvtColor(frames[i].astype(np.uint8), cv2.COLOR_RGB2GRAY)
            g1 = cv2.cvtColor(frames[i+1].astype(np.uint8), cv2.COLOR_RGB2GRAY)
            flows.append(cv2.calcOpticalFlowFarneback(
                g0,g1,None,0.5,3,15,3,5,1.2,0))
        flows = torch.from_numpy(np.stack(flows)).to(self.device)
        dif = torch.norm(torch.diff(flows,dim=0),dim=-1)
        return float(1.0/(1.0+dif.mean()))

    def _prompt_fidelity(self, video, prompt):
        # take middle frame
        frame = (video[0,:,video.shape[2]//2].permute(1,2,0)*255).cpu().numpy().astype(np.uint8)
        inputs = self.proc(text=[prompt], images=[frame], return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            out = self.clip(**inputs)
        return float(F.cosine_similarity(out.text_embeds, out.image_embeds).item())

    def _overlap_mse(self, chunks, overlap):
        ms=[]
        for a,b in zip(chunks,chunks[1:]):
            x=a['chunk']; y=b['chunk']
            ms.append(F.mse_loss(
                torch.from_numpy(x)[:,:,-overlap:].to(self.device),
                torch.from_numpy(y)[:,:,:overlap].to(self.device)
            ).item())
        return float(np.mean(ms))

    def run(self, prompt, num_frames, chunk_size, overlap, pre_steps, num_steps):
        net0 = self._measure_net(); v0=self._measure_vram(); t0=time.time()
        lat, txt = self.prepare(prompt, num_frames, num_steps)

        # Pre-denoise full latent for pre_steps (approx overlap-only)
        for t in self.scheduler.timesteps[:pre_steps]:
            inp = torch.cat([lat]*2)
            inp = self.scheduler.scale_model_input(inp, t)
            with torch.no_grad():
                noise = self.unet(inp, t, encoder_hidden_states=txt).sample
            u,cnd = noise.chunk(2)
            noise = u + 7.5*(cnd-u)
            lat = self.scheduler.step(noise, t, lat).prev_sample

        # Chunk & dispatch
        results = []
        for i in range(0, num_frames, chunk_size-overlap):
            st,ed = i, min(i+chunk_size, num_frames)
            chunk = lat[:,:,st:ed].clone()
            args = {
              'latents': chunk.cpu().numpy(),
              'text_emb': txt.cpu().numpy(),
              'start_idx': st,'end_idx': ed,
              'pre_steps': 0,'num_steps': num_steps
            }
            pkt = xmlrpc.client.Binary(pickle.dumps(args))
            widx = (i//(chunk_size-overlap)) % len(self.workers)
            w = self.workers[widx]
            send = time.time()
            res_pkl = w.process_chunk(pkt)
            recv = time.time()
            out = pickle.loads(res_pkl.data)
            results.append((out, recv-send))

        # Stitch
        full = np.zeros_like(lat.cpu().numpy())
        weight = np.zeros((num_frames,),dtype=np.float32)
        for out,_ in results:
            c = out['chunk']; st,out_e=out['start_idx'],out['end_idx']
            full[:,:,st:out_e] += c
            weight[st:out_e] += 1
        full = full / weight.reshape(1,1,-1,1,1)
        vid = self._decode(torch.from_numpy(full).to(self.device).float())

        t1=time.time(); v1=self._measure_vram(); net1=self._measure_net()
        m = {
          'end2end_s': t1-t0,
          'per_chunk_net_s':[lat for _,lat in results],
          'vram_start_mb':v0,'vram_end_mb':v1,
          'net_bytes':net1-net0,
          'flow_consistency':self._flow_consistency(vid),
          'prompt_fidelity':self._prompt_fidelity(vid,prompt),
          'overlap_mse':self._overlap_mse(
               [o for o,_ in results], overlap)
        }
        return m



if __name__=="__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--role", choices=["worker","coordinator","both"], default="both")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--workers", type=str, required=True,
                   help="commasep list of ip:port for workers")
    p.add_argument("--prompt", type=str, required=True)
    p.add_argument("--num_frames", type=int, default=16)
    p.add_argument("--chunk_size", type=int, default=8)
    p.add_argument("--overlap", type=int, default=2)
    p.add_argument("--pre_steps", type=int, default=3)
    p.add_argument("--steps", type=int, default=25)
    p.add_argument("--model_id", default="cerspense/zeroscope_v2_576w")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()


    if args.role in ("worker","both"):
        start_worker_server(args.host, args.port, args.model_id, args.device)


    time.sleep(1)

    if args.role in ("coordinator","both"):
        hosts = [h.strip() for h in args.workers.split(",")]
        print("[Coord] Workers:", hosts)
        coord = StrategyC_Coordinator(hosts, args.model_id, args.device)
        metrics = coord.run(
            args.prompt, args.num_frames,
            args.chunk_size, args.overlap,
            args.pre_steps, args.steps
        )
        print("\n=== Strategy C Metrics ===")
        for k,v in metrics.items():
            print(f"{k}: {v}")
