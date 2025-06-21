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

import os
import sys
import asyncio
import uuid

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import typing
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
    level=logging.INFO,
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
                # Enable memory efficient attention if available 
                try:
                    if hasattr(self.pipe.unet, 'enable_xformers_memory_efficient_attention'):
                        self.pipe.unet.enable_xformers_memory_efficient_attention()
                        logger.info("Enabled xformers memory efficient attention")
                except Exception as e:
                    logger.warning(f"Could not enable xformers memory optimization: {e}")
                
                # Enable gradient checkpointing to save memory
                try:
                    self.pipe.unet.enable_gradient_checkpointing()
                    logger.info("Enabled gradient checkpointing for memory efficiency")
                except Exception as e:
                    logger.warning(f"Could not enable gradient checkpointing: {e}")
                
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

        # Verify model is loaded
        if self.pipe is None:
            logger.error("Zeroscope model failed to load. Miner will not be able to generate videos.")
        else:
            logger.info("Zeroscope model loaded successfully and ready for inference")

        self._leaf_data_by_request = {}  # (caller_hotkey, request_id) -> leaf_data
        
        # Register open_leaves as an axon RPC method after axon is created
        self._attach_rpc_methods()

    def _attach_rpc_methods(self):
        """Attach RPC methods to the axon after it's created"""
        if hasattr(self, 'axon'):
            try:
                # Try to attach RPC methods using the axon's internal mechanism
                if hasattr(self.axon, '_attach_rpc'):
                    self.axon._attach_rpc('open_leaves', self.open_leaves)
                    self.axon._attach_rpc('forward_reveal_leaves', self.forward_reveal_leaves)
                    logger.info("Attached RPC methods 'open_leaves' and 'forward_reveal_leaves' using _attach_rpc")
                elif hasattr(self.axon, 'attach'):
                    # Store RPC method for manual handling
                    if not hasattr(self, '_rpc_methods'):
                        self._rpc_methods = {}
                    self._rpc_methods['open_leaves'] = self.open_leaves
                    self._rpc_methods['forward_reveal_leaves'] = self.forward_reveal_leaves
                    logger.info("Stored RPC methods 'open_leaves' and 'forward_reveal_leaves' for manual handling")
                else:
                    logger.warning("Could not attach RPC methods - axon method not found")
            except Exception as e:
                logger.warning(f"Failed to attach RPC methods: {e}")
        else:
            logger.warning("Axon not available for RPC attachment")
            
        # Also ensure the axon is properly configured for synapse handling
        if hasattr(self, 'axon'):
            try:
                # Ensure axon has proper serialization settings
                if hasattr(self.axon, 'forward'):
                    logger.info("Axon forward method is available")
                else:
                    logger.warning("Axon forward method not found")
                    
            except Exception as e:
                logger.warning(f"Error checking axon configuration: {e}")

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
            
        Returns tuple of bytes, dict - Root hash and dictionary of leaf data with proofs
        """
        logger.info(f"Building Merkle tree for {len(timesteps)} timesteps")
        start_time = time.time()
        
        try:
            # Convert tensors to bytes
            leaves = []
            leaf_data = {}
            
            for i, (z, eps, t) in enumerate(zip(latents, noise_preds, timesteps)):
                logger.info(f"Processing timestep {t} ({i+1}/{len(timesteps)})")
                
                # Convert tensors to bytes
                z_bytes = z.cpu().numpy().tobytes()
                eps_bytes = eps.cpu().numpy().tobytes()
                
                # Create leaf hash
                leaf = t.to_bytes(2, 'big') + z_bytes + eps_bytes
                leaf_hash = hashlib.sha256(leaf).digest()
                leaves.append(leaf_hash)
                
                # Store leaf data
                leaf_data[t] = (z_bytes, eps_bytes, [])
                
                logger.info(f"Leaf {i} sise: {len(z_bytes) + len(eps_bytes)} bytes")
            
            # Build Merkle tree
            logger.info("Building tree levels...")
            tree = [leaves]
            level = 0
            while len(tree[-1]) > 1:
                level += 1
                logger.info(f"Building level {level} with {len(tree[-1])} nodes")
                new_level = []
                for i in range(0, len(tree[-1]), 2):
                    left = tree[-1][i]
                    right = tree[-1][i+1] if i+1 < len(tree[-1]) else left
                    # Sort for consistency with the verifier
                    if left < right:
                        combined = left + right
                    else:
                        combined = right + left
                    new_level.append(hashlib.sha256(combined).digest())
                tree.append(new_level)
            
            # Generate proof paths
            logger.info("Generating proof paths...")
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
            logger.info(f"Tree height: {len(tree)}, Root hash: {tree[-1][0].hex()}")
            
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
        
        Input:
            challenge: Challenge bytes
            seed: Derived seed
            video_bytes: Generated video data
            merkle_root: Root of Merkle tree
            
        Returns bytes (Signature)
        """
        logger.info("Signing proof of inference")
        try:
            logger.debug(f"=== MINER SIGNING DEBUG START ===")
            logger.debug(f"Input parameters:")
            logger.debug(f"  challenge: {challenge.hex()}")
            logger.debug(f"  seed: {seed}")
            logger.debug(f"  video_bytes length: {len(video_bytes)}")
            logger.debug(f"  merkle_root: {merkle_root.hex()}")
            
            # Concatenate all components
            seed_little_endian = seed.to_bytes(8, byteorder='little', signed=False)
            message = challenge + seed_little_endian + hashlib.sha256(video_bytes).digest() + merkle_root
            
            logger.debug(f"Message components:")
            logger.debug(f"  challenge: {challenge.hex()}")
            logger.debug(f"  seed: {seed}")
            logger.debug(f"  seed_little_endian: {seed_little_endian.hex()}")
            logger.debug(f"  video_hash: {hashlib.sha256(video_bytes).hexdigest()}")
            logger.debug(f"  merkle_root: {merkle_root.hex()}")
            logger.debug(f"  full_message: {message.hex()}")
            logger.debug(f"  message_length: {len(message)} bytes")
            
            # Debug keypair info
            logger.debug(f"Keypair info:")
            logger.debug(f"  hotkey ss58: {self.wallet.hotkey.ss58_address}")
            logger.debug(f"  public key: {self.wallet.hotkey.public_key.hex()}")
            logger.debug(f"  crypto type: {self.wallet.hotkey.crypto_type}")
            
            # Sign with miner's keypair
            logger.debug(f"Calling self.wallet.hotkey.sign(message)...")
            signature = self.wallet.hotkey.sign(message)
            logger.info(f"Proof signed successfully. Signature length: {len(signature)} bytes")
            logger.debug(f"raw_signature: {signature.hex()}")
            
            # Verify the signature immediately to catch any issues
            logger.debug(f"Self-verifying signature...")
            try:
                is_valid = self.wallet.hotkey.verify(message, signature)
                logger.debug(f"✅ Self-verification result: {is_valid}")
                if not is_valid:
                    logger.error("❌ Self-verification failed! This indicates a signing issue.")
            except Exception as verify_error:
                logger.error(f"❌ Error in self-verification: {verify_error}")
            
            logger.debug(f"=== MINER SIGNING DEBUG END ===")
            return signature
            
        except Exception as e:
            logger.error(f"❌ Error signing proof: {str(e)}", exc_info=True)
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
        
        Input:
            synapse: Synapse object containing the request parameters
            
        Returns InferNet Object (The synapse with video data and proof)
        """
        # Check if this is a spot-check request first before any other processing
        # Spot-check requests are identified by seed=0 and empty challenge
        is_spot_check = (
            hasattr(synapse, 'seed') and synapse.seed == 0 and
            hasattr(synapse, 'challenge') and (not synapse.challenge or synapse.challenge == "")
        )
        
        if is_spot_check:
            request_id = getattr(synapse, 'request_id', str(uuid.uuid4()))
            logger.info(f"[{request_id}] 🔍 SPOT-CHECK REQUEST detected (seed=0, empty challenge)")
            
            # For spot-check requests, we need to get the caller hotkey from the dendrite
            caller_hotkey = None
            if hasattr(synapse, 'dendrite') and synapse.dendrite and hasattr(synapse.dendrite, 'hotkey'):
                caller_hotkey = synapse.dendrite.hotkey
            else:
                logger.warning(f"[{request_id}] No caller hotkey available for spot-check")
                # Return empty response for spot-check
                synapse.video_data_b64 = ""
                synapse.merkle_root = ""
                synapse.timesteps = []
                synapse.latents = []
                synapse.noise_preds = []
                synapse.signature = ""
                synapse.proof = {
                    'merkle_root': "",
                    'signature': "",
                    'seed': 0,
                    'challenge': "",
                    'leaf_data': {},
                }
                return synapse
            
            logger.info(f"[{request_id}] Caller hotkey: {caller_hotkey}")
            
            # Handle spot-check request
            try:
                key = (caller_hotkey, request_id)
                logger.debug(f"[{request_id}] Looking for leaf data with key: {key}")
                logger.debug(f"[{request_id}] Available leaf data keys: {list(self._leaf_data_by_request.keys())}")
                leaf_data = self._leaf_data_by_request.get(key)
                
                if not leaf_data:
                    logger.warning(f"[{request_id}] No leaf data found for spot-check key: {key}")
                    # Return empty response for spot-check
                    synapse.video_data_b64 = ""
                    synapse.merkle_root = ""
                    synapse.timesteps = []
                    synapse.latents = []
                    synapse.noise_preds = []
                    synapse.signature = ""
                    synapse.proof = {
                        'merkle_root': "",
                        'signature': "",
                        'seed': 0,
                        'challenge': "",
                        'leaf_data': {},
                    }
                    return synapse
                
                # For spot-check requests, we need to know which indices to return
                # Since we can't get the indices from the synapse (they get lost in serialization),
                # we'll return all available leaves
                logger.info(f"[{request_id}] Returning all available leaves for spot-check")
                
                # Build result dictionary with all available leaves
                result = {}
                for timestep, (z_bytes, eps_bytes, proof_path) in leaf_data.items():
                    # Encode as base64 strings for JSON serialization
                    z_b64 = base64.b64encode(z_bytes).decode('ascii')
                    eps_b64 = base64.b64encode(eps_bytes).decode('ascii')
                    proof_path_b64 = [base64.b64encode(p).decode('ascii') for p in proof_path]
                    result[timestep] = (z_b64, eps_b64, proof_path_b64)
                
                # Clean up the leaf data
                del self._leaf_data_by_request[key]
                
                # Store the leaves in the proof.leaf_data field
                synapse.video_data_b64 = ""  # No video for spot-check
                synapse.merkle_root = ""
                synapse.timesteps = []
                synapse.latents = []
                synapse.noise_preds = []
                synapse.signature = ""
                synapse.proof = {
                    'merkle_root': "",
                    'signature': "",
                    'seed': 0,
                    'challenge': "",
                    'leaf_data': result,  # Store all leaves here
                }
                
                logger.info(f"[{request_id}] ✅ SPOT-CHECK: Returning {len(result)} leaves")
                return synapse
                
            except Exception as e:
                logger.error(f"[{request_id}] ❌ Error in spot-check handling: {str(e)}", exc_info=True)
                # Return empty response for spot-check
                synapse.video_data_b64 = ""
                synapse.merkle_root = ""
                synapse.timesteps = []
                synapse.latents = []
                synapse.noise_preds = []
                synapse.signature = ""
                synapse.proof = {
                    'merkle_root': "",
                    'signature': "",
                    'seed': 0,
                    'challenge': "",
                    'leaf_data': {},
                }
                return synapse
        
        # Check if this is an RPC call for open_leaves
        if hasattr(synapse, '_rpc_method') and synapse._rpc_method == 'open_leaves':
            # Handle RPC call manually
            if hasattr(synapse, '_rpc_args') and hasattr(synapse, '_rpc_kwargs'):
                indices = synapse._rpc_args[0] if synapse._rpc_args else []
                request_id = synapse._rpc_kwargs.get('request_id')
                caller_hotkey = synapse._rpc_kwargs.get('caller_hotkey')
                result = await self.open_leaves(indices, request_id, caller_hotkey)
                # Set the result in the synapse
                synapse._rpc_result = result
                return synapse
        
        # Normal video generation request
        self.metrics['total_requests'] += 1
        # Use the request_id from the synapse if provided, otherwise generate a fallback
        request_id = getattr(synapse, 'request_id', None)
        if request_id is None:
            request_id = f"req_{self.metrics['total_requests']}"
            logger.warning(f"[{request_id}] No request_id provided in synapse, using fallback")
        else:
            logger.info(f"[{request_id}] Using provided request_id from synapse")
            
        logger.info(f"[{request_id}] Received video generation request with prompt: {synapse.text_prompt}")

        try:
            # Check if model is loaded
            if self.pipe is None:
                logger.error(f"[{request_id}] Zeroscope model was not properly loaded during initialisation")
                raise ValueError("Zeroscope model was not properly loaded during initialisation. Check the miner logs for model loading errors.")
                
            # Extract parameters
            assert hasattr(synapse, "text_prompt"), "Missing text prompt"
            fps = int(getattr(synapse, "fps", 1))  # Minimal FPS
            height = int(getattr(synapse, "height", 128))  # Minimal height

            width = int(getattr(synapse, "width", 128))  # Minimal width
            num_frames = int(getattr(synapse, "num_frames", 3))  # Minimal frames
            
            logger.info(f"[{request_id}] Parameters: fps={fps}, height={height}, width={width}, num_frames={num_frames}")
            
            # Get challenge and seed
            challenge = synapse.challenge
            seed = synapse.seed
            logger.info(f"[{request_id}] Using seed: {seed}")
            
            # Decode challenge from base64 string to bytes for signing
            challenge_bytes = base64.b64decode(challenge) if challenge else b""
            logger.info(f"[{request_id}] Decoded challenge from base64, length: {len(challenge_bytes)} bytes")
            
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
            # Get num_steps from config with fallback
            num_steps = getattr(self.config, 'diffusion', None)
            if num_steps is None or not hasattr(num_steps, 'num_steps'):
                # Fallback to default value if diffusion config is missing
                num_steps = 10
                logger.warning(f"[{request_id}] diffusion.num_steps not found in config, using default: {num_steps}")
            else:
                num_steps = num_steps.num_steps
            scheduler.set_timesteps(num_steps)
            logger.info(f"[{request_id}] Using {len(scheduler.timesteps)} denoising steps")
            
            # Initial noise
            z = torch.randn(
                (1, 4, num_frames, height // 8, width // 8),
                device=self.pipe.device,
                dtype=torch.float16
            )
            logger.info(f"[{request_id}] Initial noise shape: {z.shape}")
            
            # Encode text prompt once before the loop
            logger.info(f"[{request_id}] Encoding text prompt")
            text_inputs = self.pipe.tokenizer(
                synapse.text_prompt,
                padding="max_length",
                max_length=self.pipe.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt"
            )
            text_inputs = {k: v.to(self.pipe.device) for k, v in text_inputs.items()}
            with torch.no_grad():
                encoder_hidden_states = self.pipe.text_encoder(**text_inputs)[0]
            logger.info(f"[{request_id}] Text encoding completed")
            
            # Run denoising steps
            for i, t in enumerate(scheduler.timesteps):
                logger.info(f"[{request_id}] Step {i+1}/{len(scheduler.timesteps)}, timestep {t}")
                step_start = time.time()
                
                # Store latent and timestep
                latents.append(z.detach())
                timesteps.append(t.item())
                
                # Predict noise with pre-encoded text conditioning
                with torch.no_grad():
                    eps = self.pipe.unet(z, t, encoder_hidden_states=encoder_hidden_states).sample
                
                noise_preds.append(eps.detach())
                
                # Update latent
                z = scheduler.step(eps, t, z).prev_sample
                
                step_time = time.time() - step_start
                logger.info(f"[{request_id}] Step {i+1} completed in {step_time:.2f}s")
            
            # Decode final latents to frames
            logger.info(f"[{request_id}] Decoding final latents")
        
            B, C, T, H, W = z.shape
            flat_latents = z.view(B * T, C, H, W)
            decoded = self.pipe.vae.decode(flat_latents / self.pipe.vae.config.scaling_factor).sample
            # decoded: (B*T, 3, H_out, W_out)
            # reshape back into (B, T, 3, H_out, W_out)
            decoded = decoded.view(B, T, *decoded.shape[-3:])
            frames = decoded[0].permute(0, 2, 3, 1).cpu().detach().numpy()
            
            
            generation_time = time.time() - start
            self.metrics['last_generation_time'] = generation_time
            self.metrics['avg_generation_time'] = (
                (self.metrics['avg_generation_time'] * (self.metrics['successful_generations']) + generation_time) /
                (self.metrics['successful_generations'] + 1)
            )
            
            logger.info(f"[{request_id}] Video generation completed in {generation_time:.2f} seconds")
            
            # Export frames to video
            logger.info(f"[{request_id}] Exporting frames to video")
            video_path = export_to_video(frames, fps=fps)
            
            # Read video bytes
            with open(video_path, "rb") as f:
                video_bytes = f.read()
                
            # Clean up
            if os.path.exists(video_path):
                os.remove(video_path)
            
            # Clear GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                import gc
                gc.collect()
                logger.info(f"[{request_id}] GPU memory cleared. Current usage: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
            
            # Build Merkle tree
            logger.info(f"[{request_id}] Building Merkle tree")
            merkle_root, leaf_data = self._build_merkle_tree(latents, noise_preds, timesteps)
            # Store leaf_data for this request, keyed by (caller_hotkey, request_id)
            caller_hotkey = getattr(synapse.dendrite, 'hotkey', 'unknown') if hasattr(synapse, 'dendrite') else 'unknown'
            self._leaf_data_by_request[(caller_hotkey, request_id)] = leaf_data
            
            # Sign proof
            logger.info(f"[{request_id}] Signing proof")
            signature = self._sign_proof(challenge_bytes, seed, video_bytes, merkle_root)
            
            # Encode every leaf (z, eps, path) into proof.leaf_data
            leaf_data_b64 = {
                t: (
                    base64.b64encode(z_bytes).decode('ascii'),
                    base64.b64encode(eps_bytes).decode('ascii'),
                    [base64.b64encode(p).decode('ascii') for p in proof_path]
                )
                for t, (z_bytes, eps_bytes, proof_path) in leaf_data.items()
            }

            # Assign base64-encoded strings for network transmission
            synapse.video_data_b64 = base64.b64encode(video_bytes).decode('ascii')
            synapse.merkle_root = base64.b64encode(merkle_root).decode('ascii')
            synapse.timesteps = timesteps
            synapse.latents = [base64.b64encode(z.cpu().numpy().tobytes()).decode('ascii') for z in latents]
            synapse.noise_preds = [base64.b64encode(eps.cpu().numpy().tobytes()).decode('ascii') for eps in noise_preds]
            synapse.signature = base64.b64encode(signature).decode('ascii')
            synapse.proof = {
                'merkle_root': base64.b64encode(merkle_root).decode('ascii'),
                'signature': base64.b64encode(signature).decode('ascii'),
                'seed': seed,
                'challenge': challenge,
                'leaf_data': leaf_data_b64,
            }
            
            # Debug: Log what we're setting
            logger.info(f"[{request_id}] MINER DEBUG: Setting synapse fields")
            logger.info(f"[{request_id}]   - timesteps: {timesteps} (type: {type(timesteps)}, length: {len(timesteps)})")
            logger.info(f"[{request_id}]   - video_data_b64 length: {len(synapse.video_data_b64)}")
            logger.info(f"[{request_id}]   - merkle_root length: {len(synapse.merkle_root)}")
            logger.info(f"[{request_id}]   - signature length: {len(synapse.signature)}")
            logger.info(f"[{request_id}]   - latents length: {len(synapse.latents)}")
            logger.info(f"[{request_id}]   - noise_preds length: {len(synapse.noise_preds)}")
            logger.info(f"[{request_id}]   - proof keys: {list(synapse.proof.keys())}")
            
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
            # Ensure synapse object is properly initialized even on error
            synapse.video_data_b64 = ""
            synapse.merkle_root = ""
            synapse.timesteps = []
            synapse.latents = []
            synapse.noise_preds = []
            synapse.signature = ""
            synapse.proof = {
                'merkle_root': "",
                'signature': "",
                'seed': 0,
                'challenge': "",
                'leaf_data': {},
            }
            
        # Log metrics
        self._log_metrics()
        
        # Ensure all fields are properly set before returning
        if not hasattr(synapse, 'video_data_b64') or synapse.video_data_b64 is None:
            synapse.video_data_b64 = ""
        if not hasattr(synapse, 'merkle_root') or synapse.merkle_root is None:
            synapse.merkle_root = ""
        if not hasattr(synapse, 'timesteps') or synapse.timesteps is None:
            synapse.timesteps = []
        if not hasattr(synapse, 'latents') or synapse.latents is None:
            synapse.latents = []
        if not hasattr(synapse, 'noise_preds') or synapse.noise_preds is None:
            synapse.noise_preds = []
        if not hasattr(synapse, 'signature') or synapse.signature is None:
            synapse.signature = ""
        if not hasattr(synapse, 'proof') or synapse.proof is None:
            synapse.proof = {
                'merkle_root': "",
                'signature': "",
                'seed': 0,
                'challenge': "",
                'leaf_data': {},
            }
            
        # Debug: Log final synapse state
        logger.info(f"[{request_id}] FINAL MINER DEBUG: Returning timesteps")
        logger.info(f"[{request_id}]   - synapse.timesteps: {synapse.timesteps}")
        logger.info(f"[{request_id}]   - synapse.timesteps type: {type(synapse.timesteps)}")
        logger.info(f"[{request_id}]   - synapse.timesteps length: {len(synapse.timesteps)}")
        
        # Debug: Log synapse structure
        logger.debug(f"[{request_id}] Synapse structure:")
        logger.debug(f"  - video_data_b64: {type(synapse.video_data_b64)} (length: {len(synapse.video_data_b64)})")
        logger.debug(f"  - merkle_root: {type(synapse.merkle_root)} (length: {len(synapse.merkle_root)})")
        logger.debug(f"  - signature: {type(synapse.signature)} (length: {len(synapse.signature)})")
        logger.debug(f"  - timesteps: {type(synapse.timesteps)} (length: {len(synapse.timesteps)})")
        logger.debug(f"  - latents: {type(synapse.latents)} (length: {len(synapse.latents)})")
        logger.debug(f"  - noise_preds: {type(synapse.noise_preds)} (length: {len(synapse.noise_preds)})")
        logger.debug(f"  - proof: {type(synapse.proof)}")
        
        # Ensure synapse is JSON serialisable
        try:
            import json
            json.dumps(synapse.dict())
            logger.debug(f"[{request_id}] Synapse is JSON serialisable")
        except Exception as e:
            logger.error(f"[{request_id}] Synapse is NOT JSON serialisable: {e}")
            
        logger.debug(f"[{request_id}] Synapse object ID: {id(synapse)}")
        logger.debug(f"[{request_id}] Synapse has video_data_b64 attribute: {hasattr(synapse, 'video_data_b64')}")
        logger.debug(f"[{request_id}] Synapse video_data_b64 is None: {synapse.video_data_b64 is None}")
        
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
            logger.info(f"Request from UID {uid} with hotkey {synapse.dendrite.hotkey}")
        except ValueError:
            logger.warning(f"Hotkey {synapse.dendrite.hotkey} not found in metagraph")
            if self.config.blacklist.allow_non_registered:
                return False, "Non-registered hotkey allowed"
            return True, "Hotkey not in metagraph"

        if (
            not self.config.blacklist.allow_non_registered
            and synapse.dendrite.hotkey not in self.metagraph.hotkeys
        ):
            logger.info(f"Blacklisting un-registered hotkey {synapse.dendrite.hotkey}")
            return True, "Unrecognised hotkey"

        if self.config.blacklist.force_validator_permit:
            if not self.metagraph.validator_permit[uid]:
                logger.warning(f"Blacklisting a request from non-validator hotkey {synapse.dendrite.hotkey}")
                return True, "Non-validator hotkey"

        logger.info(f"Not Blacklisting recognised hotkey {synapse.dendrite.hotkey}")
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
            logger.info(f"Priority for UID {caller_uid}: {priority}")
        except (ValueError, IndexError):
            logger.warning(f"Could not get priority for hotkey {synapse.dendrite.hotkey}")
            return 0.0
            
        logger.info(f"Prioritising {synapse.dendrite.hotkey} with value: {priority}")
        return priority

    async def open_leaves(self, indices: list, request_id: str = None, caller_hotkey: str = None):
        """
        RPC method - Given a list of indices, return only the requested (z_bytes, eps_bytes, proof_path) for each index.
        Uses the leaf_data for the given (caller_hotkey, request_id).
        Returns base64-encoded strings for JSON serialisation.
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
        
        result = {}
        for i in indices:
            if i in leaf_data:
                z_bytes, eps_bytes, proof_path = leaf_data[i]
                # Encode as base64 strings for JSON serialisation
                z_b64 = base64.b64encode(z_bytes).decode('ascii')
                eps_b64 = base64.b64encode(eps_bytes).decode('ascii')
                proof_path_b64 = [base64.b64encode(p).decode('ascii') for p in proof_path]
                result[i] = (z_b64, eps_b64, proof_path_b64)
            else:
                result[i] = None
   
        del self._leaf_data_by_request[key]
        return result

    async def forward_reveal_leaves(self, synapse: template.protocol.RevealLeavesSynapse) -> template.protocol.RevealLeavesSynapse:
        """
        Handle RevealLeavesSynapse for spot-checking.
        This method is called when the validator requests specific leaves for verification.
        """
        try:
            logger.info(f"Received RevealLeavesSynapse request for indices: {synapse.indices}")
            logger.info(f"Request ID: {synapse.request_id}, Caller: {synapse.caller_hotkey}")
            
            # Use the existing open_leaves logic
            indices = [int(i) for i in synapse.indices]
            key = (synapse.caller_hotkey, synapse.request_id)
            leaf_data = self._leaf_data_by_request.get(key)
            
            if not leaf_data:
                logger.warning(f"No leaf data found for key: {key}")
                synapse.leaves = {i: None for i in indices}
                return synapse
            
            # Build result dictionary
            result = {}
            for i in indices:
                if i in leaf_data:
                    z_bytes, eps_bytes, proof_path = leaf_data[i]
                    # Encode as base64 strings for JSON serialization
                    z_b64 = base64.b64encode(z_bytes).decode('ascii')
                    eps_b64 = base64.b64encode(eps_bytes).decode('ascii')
                    proof_path_b64 = [base64.b64encode(p).decode('ascii') for p in proof_path]
                    result[i] = (z_b64, eps_b64, proof_path_b64)
                else:
                    result[i] = None
            
            # Clean up the leaf data
            del self._leaf_data_by_request[key]
            
            # Set the leaves in the synapse
            synapse.leaves = result
            logger.info(f"Returning {len(result)} leaves for spot-checking")
            
            return synapse
            
        except Exception as e:
            logger.error(f"Error in forward_reveal_leaves: {str(e)}", exc_info=True)
            synapse.leaves = {i: None for i in synapse.indices}
            return synapse

if __name__ == "__main__":
    # Create and run the miner
    miner = Miner()
    miner.run()
