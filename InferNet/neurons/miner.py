# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# Developer: Noor Elsheikh
# Copyright © 2025 Noor Elsheikh

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import time
import typing
import os
import io
import base64
import hmac
import hashlib
import torch
import numpy as np
import bittensor as bt
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import logging
import json
from datetime import datetime

# Bittensor Miner Template:
import template
from template.base.miner import BaseMinerNeuron
from template.protocol import InferNet

# Import for video generation
from diffusers import DiffusionPipeline
from diffusers.utils import export_to_video

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('miner_debug.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('Miner')

class Miner(BaseMinerNeuron):
    def __init__(self, config=None):
        super(Miner, self).__init__(config=config)
        
        # Initialise debug metrics
        self.metrics = {
            'total_requests': 0,
            'successful_generations': 0,
            'failed_generations': 0,
            'avg_generation_time': 0,
            'last_generation_time': 0,
            'model_load_time': 0,
            'errors': []
        }
        
        # Initialise the video generation pipeline
        logger.info("Loading Zeroscope model...")
        start_time = time.time()
        
        try:
            # Load the video generation model
            self.pipe = DiffusionPipeline.from_pretrained(
                "cerspense/zeroscope_v2_576w", 
                torch_dtype=torch.float16,
                use_safetensors=False
            )
            
            # Move model to GPU if available
            if torch.cuda.is_available():
                self.pipe = self.pipe.to("cuda")
                logger.info(f"Using CUDA for inference. Device: {torch.cuda.get_device_name()}")
                logger.info(f"CUDA memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
            else:
                logger.warning("CUDA not available, using CPU (this will be very slow)")
                
            self.metrics['model_load_time'] = time.time() - start_time
            logger.info(f"Zeroscope model loaded successfully in {self.metrics['model_load_time']:.2f} seconds")
            
        except Exception as e:
            logger.error(f"Error loading Zeroscope model: {str(e)}", exc_info=True)
            logger.error("Continuing without model - inference will fail until fixed.")
            self.pipe = None
            self.metrics['errors'].append({
                'timestamp': datetime.now().isoformat(),
                'error': str(e),
                'type': 'model_load_error'
            })

        self._leaf_data_by_request = {}  # (caller_hotkey, request_id) -> leaf_data
        # Register open_leaves as an axon RPC method
        if hasattr(self, 'axon'):
            self.axon.attach_rpc('open_leaves', self.open_leaves)

    def _log_metrics(self):
        """Log current metrics to file"""
        try:
            metrics_file = 'miner_metrics.json'
            with open(metrics_file, 'w') as f:
                json.dump(self.metrics, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving metrics: {str(e)}")

    def _build_merkle_tree(self, latents: List[torch.Tensor], noise_preds: List[torch.Tensor], timesteps: List[int]) -> Tuple[bytes, Dict[int, Tuple[bytes, bytes, List[bytes]]]]:
        """
        Build Merkle tree from intermediate latents and noise predictions.
        
        Inputs:
        - latents: List of latent tensors
        - noise_preds: List of noise predictions
        - timesteps: List of timesteps
            
        Returns Tuple[bytes, Dict] Root hash and dictionary of leaf data with proofs
        """
        logger.debug(f"Building Merkle tree for {len(timesteps)} timesteps")
        start_time = time.time()
        
        try:
            # Convert tensors to bytes
            leaves = []
            leaf_data = {}
            
            for i, (z, eps, t) in enumerate(zip(latents, noise_preds, timesteps)):
                logger.debug(f"Processing timestep {t} ({i+1}/{len(timesteps)})")
                
                # Convert tensors to bytes
                z_bytes = z.cpu().numpy().tobytes()
                eps_bytes = eps.cpu().numpy().tobytes()
                
                # Create leaf hash
                leaf = t.to_bytes(2, 'big') + z_bytes + eps_bytes
                leaf_hash = hashlib.sha256(leaf).digest()
                leaves.append(leaf_hash)
                
                # Store leaf data
                leaf_data[t] = (z_bytes, eps_bytes, [])
                
                logger.debug(f"Leaf {i} sise: {len(z_bytes) + len(eps_bytes)} bytes")
            
            # Build Merkle tree
            logger.debug("Building tree levels...")
            tree = [leaves]
            level = 0
            while len(tree[-1]) > 1:
                level += 1
                logger.debug(f"Building level {level} with {len(tree[-1])} nodes")
                new_level = []
                for i in range(0, len(tree[-1]), 2):
                    if i + 1 < len(tree[-1]):
                        new_level.append(hashlib.sha256(tree[-1][i] + tree[-1][i+1]).digest())
                    else:
                        new_level.append(hashlib.sha256(tree[-1][i] + tree[-1][i]).digest())
                tree.append(new_level)
            
            # Generate proof paths
            logger.debug("Generating proof paths...")
            for i, t in enumerate(timesteps):
                proof = []
                idx = i
                for level in tree[:-1]:
                    if idx % 2 == 0:
                        if idx + 1 < len(level):
                            proof.append(level[idx + 1])
                        else:
                            proof.append(level[idx])
                    else:
                        proof.append(level[idx - 1])
                    idx //= 2
                leaf_data[t] = (leaf_data[t][0], leaf_data[t][1], proof)
            
            build_time = time.time() - start_time
            logger.info(f"Merkle tree built in {build_time:.2f} seconds")
            logger.debug(f"Tree height: {len(tree)}, Root hash: {tree[-1][0].hex()}")
            
            return tree[-1][0], leaf_data
            
        except Exception as e:
            logger.error(f"Error building Merkle tree: {str(e)}", exc_info=True)
            self.metrics['errors'].append({
                'timestamp': datetime.now().isoformat(),
                'error': str(e),
                'type': 'merkle_tree_error'
            })
            raise

    def _sign_proof(self, challenge: bytes, seed: int, video_bytes: bytes, merkle_root: bytes) -> bytes:
        """
        Sign the proof of inference.
        
        Args:
            challenge: Challenge bytes
            seed: Derived seed
            video_bytes: Generated video data
            merkle_root: Root of Merkle tree
            
        Returns:
            bytes: Signature
        """
        logger.debug("Signing proof of inference")
        try:
            # Concatenate all components
            # message = challenge + str(seed).encode() + hashlib.sha256(video_bytes).digest() + merkle_root
            seed_little_endian = seed.to_bytes(8, byteorder='little', signed=False)
            message = challenge + seed_little_endian + hashlib.sha256(video_bytes).digest() + merkle_root
            
            # Sign with miner's keypair
            signature = self.wallet.hotkey.sign(message)
            logger.debug(f"Proof signed successfully. Signature length: {len(signature)} bytes")
            return signature
            
        except Exception as e:
            logger.error(f"Error signing proof: {str(e)}", exc_info=True)
            self.metrics['errors'].append({
                'timestamp': datetime.now().isoformat(),
                'error': str(e),
                'type': 'signature_error'
            })
            raise

    async def forward(
        self, synapse: template.protocol.InferNet
    ) -> template.protocol.InferNet:
        """
        Processes a video generation request with proof-of-inference.
        
        Args:
            synapse: Synapse object containing the request parameters
            
        Returns:
            InferNet: The synapse with video data and proof
        """
        self.metrics['total_requests'] += 1
        request_id = f"req_{self.metrics['total_requests']}"
        logger.info(f"[{request_id}] Received video generation request with prompt: {synapse.text_prompt}")

        try:
            # Check if model is loaded
            if self.pipe is None:
                raise ValueError("Zeroscope model was not properly loaded during initialisation")
                
            # Extract parameters
            assert hasattr(synapse, "text_prompt"), "Missing text prompt"
            fps = int(getattr(synapse, "fps", 8))
            height = int(getattr(synapse, "height", 320))
            width = int(getattr(synapse, "width", 576))
            num_frames = int(getattr(synapse, "num_frames", 16))
            
            logger.debug(f"[{request_id}] Parameters: fps={fps}, height={height}, width={width}, num_frames={num_frames}")
            
            # Get challenge and seed
            challenge = synapse.challenge
            seed = synapse.seed
            logger.debug(f"[{request_id}] Using seed: {seed}")
            
            # Set random seed for deterministic generation
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
            
            # Generate video frames with intermediate latents
            logger.info(f"[{request_id}] Starting video generation...")
            start = time.time()
            
            # Run inference with intermediate latents
            latents = []
            noise_preds = []
            timesteps = []
            
            # Get scheduler timesteps
            scheduler = self.pipe.scheduler
            scheduler.set_timesteps(25)  # Can be adjusted
            logger.debug(f"[{request_id}] Using {len(scheduler.timesteps)} denoising steps")
            
            # Initial noise
            z = torch.randn(
                (1, 4, num_frames, height // 8, width // 8),
                device=self.pipe.device,
                dtype=torch.float16
            )
            logger.debug(f"[{request_id}] Initial noise shape: {z.shape}")
            
            # Run denoising steps
            for i, t in enumerate(scheduler.timesteps):
                logger.debug(f"[{request_id}] Step {i+1}/{len(scheduler.timesteps)}, timestep {t}")
                step_start = time.time()
                
                # Store latent and timestep
                latents.append(z.detach())
                timesteps.append(t.item())
                
                # Predict noise
                eps = self.pipe.unet(z, t, encoder_hidden_states=None).sample
                noise_preds.append(eps.detach())
                
                # Update latent
                z = scheduler.step(eps, t, z).prev_sample
                
                step_time = time.time() - step_start
                logger.debug(f"[{request_id}] Step {i+1} completed in {step_time:.2f}s")
            
            # Decode final latents to frames
            logger.debug(f"[{request_id}] Decoding final latents")
            frames = self.pipe.vae.decode(z / self.pipe.vae.config.scaling_factor).sample
            frames = (frames / 2 + 0.5).clamp(0, 1)
            frames = frames.cpu().permute(0, 2, 3, 4, 1).numpy()
            
            generation_time = time.time() - start
            self.metrics['last_generation_time'] = generation_time
            self.metrics['avg_generation_time'] = (
                (self.metrics['avg_generation_time'] * (self.metrics['successful_generations']) + generation_time) /
                (self.metrics['successful_generations'] + 1)
            )
            
            logger.info(f"[{request_id}] Video generation completed in {generation_time:.2f} seconds")
            
            # Export frames to video
            logger.debug(f"[{request_id}] Exporting frames to video")
            video_path = export_to_video(frames[0], fps=fps)
            
            # Read video bytes
            with open(video_path, "rb") as f:
                video_bytes = f.read()
                
            # Clean up
            if os.path.exists(video_path):
                os.remove(video_path)
            
            # Build Merkle tree
            logger.debug(f"[{request_id}] Building Merkle tree")
            merkle_root, leaf_data = self._build_merkle_tree(latents, noise_preds, timesteps)
            # Store leaf_data for this request, keyed by (caller_hotkey, request_id)
            caller_hotkey = getattr(synapse.dendrite, 'hotkey', 'unknown') if hasattr(synapse, 'dendrite') else 'unknown'
            self._leaf_data_by_request[(caller_hotkey, request_id)] = leaf_data
            
            # Sign proof
            logger.debug(f"[{request_id}] Signing proof")
            signature = self._sign_proof(challenge, seed, video_bytes, merkle_root)
            
            # Set response (commitment only, no leaf_data)
            synapse.video_data_b64 = base64.b64encode(video_bytes).decode('ascii')
            synapse.merkle_root = merkle_root
            synapse.timesteps = timesteps
            synapse.latents = [z.cpu().numpy().tobytes() for z in latents]
            synapse.noise_preds = [eps.cpu().numpy().tobytes() for eps in noise_preds]
            synapse.signature = signature
            synapse.proof = {
                'merkle_root': merkle_root,
                'signature': signature,
                'seed': seed,
                'challenge': challenge,
            }
            
            self.metrics['successful_generations'] += 1
            logger.info(f"[{request_id}] Video generated successfully, size: {len(video_bytes)} bytes")
            
        except Exception as e:
            self.metrics['failed_generations'] += 1
            self.metrics['errors'].append({
                'timestamp': datetime.now().isoformat(),
                'error': str(e),
                'type': 'generation_error',
                'request_id': request_id
            })
            logger.error(f"[{request_id}] Error generating video: {str(e)}", exc_info=True)
            synapse.video_data_b64 = ""
            
        # Log metrics
        self._log_metrics()
        return synapse

    async def blacklist(
        self, synapse: template.protocol.InferNet
    ) -> typing.Tuple[bool, str]:
        """
        Determines whether an incoming request should be blacklisted.
        """
        if synapse.dendrite is None or synapse.dendrite.hotkey is None:
            logger.warning("Received a request without a dendrite or hotkey.")
            return True, "Missing dendrite or hotkey"

        try:
            uid = self.metagraph.hotkeys.index(synapse.dendrite.hotkey)
            logger.debug(f"Request from UID {uid} with hotkey {synapse.dendrite.hotkey}")
        except ValueError:
            logger.warning(f"Hotkey {synapse.dendrite.hotkey} not found in metagraph")
            if self.config.blacklist.allow_non_registered:
                return False, "Non-registered hotkey allowed"
            return True, "Hotkey not in metagraph"

        if (
            not self.config.blacklist.allow_non_registered
            and synapse.dendrite.hotkey not in self.metagraph.hotkeys
        ):
            logger.trace(f"Blacklisting un-registered hotkey {synapse.dendrite.hotkey}")
            return True, "Unrecognised hotkey"

        if self.config.blacklist.force_validator_permit:
            if not self.metagraph.validator_permit[uid]:
                logger.warning(f"Blacklisting a request from non-validator hotkey {synapse.dendrite.hotkey}")
                return True, "Non-validator hotkey"

        logger.trace(f"Not Blacklisting recognised hotkey {synapse.dendrite.hotkey}")
        return False, "Hotkey recognised!"

    async def priority(self, synapse: template.protocol.InferNet) -> float:
        """
        The priority function determines the order in which requests are handled.
        """
        if synapse.dendrite is None or synapse.dendrite.hotkey is None:
            logger.warning("Received a request without a dendrite or hotkey.")
            return 0.0

        try:
            caller_uid = self.metagraph.hotkeys.index(synapse.dendrite.hotkey)
            priority = float(self.metagraph.S[caller_uid])
            logger.debug(f"Priority for UID {caller_uid}: {priority}")
        except (ValueError, IndexError):
            logger.warning(f"Could not get priority for hotkey {synapse.dendrite.hotkey}")
            return 0.0
            
        logger.trace(f"Prioritising {synapse.dendrite.hotkey} with value: {priority}")
        return priority

    async def open_leaves(self, indices: list, request_id: str = None, caller_hotkey: str = None):
        """
        RPC method - Given a list of indices, return only the requested (z_bytes, eps_bytes, proof_path) for each index.
        Uses the leaf_data for the given (caller_hotkey, request_id).
        """
        indices = [int(i) for i in indices]
        if caller_hotkey is None:
            caller_hotkey = 'unknown'
        if request_id is None:
            return {i: None for i in indices}
        key = (caller_hotkey, request_id)
        leaf_data = self._leaf_data_by_request.get(key)
        if not leaf_data:
            return {i: None for i in indices}
        result = {i: leaf_data[i] for i in indices if i in leaf_data}
   
        del self._leaf_data_by_request[key]
        return result

if __name__ == "__main__":
    with Miner() as miner:
        while True:
            logger.info(f"Miner running... {time.time()}")
            time.sleep(5)
