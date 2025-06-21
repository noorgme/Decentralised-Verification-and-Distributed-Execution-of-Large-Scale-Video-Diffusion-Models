# Validator implementation combining proof-of-inference, spot-check Merkle commitment,
# cheat-detection, scoring, and reward updates.

import os
import sys
import time
import asyncio
import copy
import random
import base64
import hmac
import hashlib
import json
import logging
import numpy as np
import bittensor as bt
from typing import List, Dict, Tuple, Optional, Union
from datetime import datetime
import threading
from pathlib import Path
from web3 import Web3

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eth_utils import keccak
from flask import Flask, request, jsonify
from api.prompt_api import create_prompt_api
from events.deposit_listener import start_deposit_listener

from template.base.validator import BaseValidatorNeuron
from template.protocol import InferNet
from template.validator.proof import verify_proof_of_inference, verify_proof_signature, verify_merkle_leaf, run_unet_step, derive_seed
from template.validator.scoring import compute_quality_score_clip, verify_video_authenticity_clip, MDVQS_scorer
from template.utils.uids import get_random_uids

from dotenv import load_dotenv

load_dotenv()


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('validator_debug.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('Validator')

# Directory to store generated videos
OUTPUT_DIR = Path("./generated_videos")
OUTPUT_DIR.mkdir(exist_ok=True)

# --- On-chain contract integration config ---
ETH_NODE_URL = os.environ.get('INFERNET_ETH_NODE_URL', 'http://localhost:8545')
CONTRACT_ABI_PATH = os.environ.get('INFERNET_CONTRACT_ABI', 'out/InferNetRewards.sol/InferNetRewards.json')
CONTRACT_ADDRESS = os.environ.get('INFERNET_CONTRACT_ADDRESS', '0xYourContractAddress')
VALIDATOR_ETH_ADDRESS = os.environ.get('INFERNET_VALIDATOR_ADDRESS', '0xYourValidatorAddress')
VALIDATOR_PRIVATE_KEY = os.environ.get('INFERNET_VALIDATOR_PRIVATE_KEY', '0xYourValidatorPrivateKey')

print("On-chain Contract integration environment variables:")
print(f"ETH_NODE_URL: {ETH_NODE_URL}")
print(f"CONTRACT_ABI_PATH: {CONTRACT_ABI_PATH}")
print(f"CONTRACT_ADDRESS: {CONTRACT_ADDRESS}")
print(f"VALIDATOR_ETH_ADDRESS: {VALIDATOR_ETH_ADDRESS}")
print(f"VALIDATOR_PRIVATE_KEY: {VALIDATOR_PRIVATE_KEY}")


# --- Load contract and web3 ---
with open(CONTRACT_ABI_PATH) as f:
    contract_json = json.load(f)
    contract_abi = contract_json['abi']

w3 = Web3(Web3.HTTPProvider(ETH_NODE_URL))
w3.ens = None

# Convert contract address to checksum format
CONTRACT_ADDRESS_CHECKSUM = w3.to_checksum_address(CONTRACT_ADDRESS)
print(f"Original contract address: {CONTRACT_ADDRESS}")
print(f"Checksum contract address: {CONTRACT_ADDRESS_CHECKSUM}")

contract = w3.eth.contract(address=CONTRACT_ADDRESS_CHECKSUM, abi=contract_abi)

class ValidatorNeuron(BaseValidatorNeuron):
    """
    Full Validator neuron with enhanced logging and debugging.
    """

    def __init__(self, config: Optional[bt.config] = None):
        logger.info("Starting validator initialisation...")
        start_time = time.time()
        
        logger.info("Initialising base validator...")
        super().__init__(config)
        logger.info(f"Base validator initialised in {time.time() - start_time:.2f}s")
        
        # Ensure UNet config is available for proof verification
        if not hasattr(self.config, 'unet_config') or not self.config.unet_config:
            logger.info("Initialising UNet config...")
            self.config.unet_config = {
                'latent_channels': getattr(self.config, 'unet_config.latent_channels', 4),
                'latent_height': getattr(self.config, 'unet_config.latent_height', 16),  # 128 // 8 for minimal height
                'latent_width': getattr(self.config, 'unet_config.latent_width', 16),   # 128 // 8 for minimal width
                'alphas': [0.9999, 0.9998, 0.9997, 0.9996]  # 4 steps for ultra-fast diffusion demo
            }
            logger.info(f"UNet config initialised: {self.config.unet_config}")
        
        # Add clean_start configuration
        self.clean_start = getattr(config, 'clean_start', True)
        logger.info(f"Clean start mode: {self.clean_start}")
        
        # Initialise trust weights
        logger.info("Initialising trust weights...")
        self.trust_weights = {hotkey: 1.0 for hotkey in self.metagraph.hotkeys}
        logger.info(f"Trust weights initialised in {time.time() - start_time:.2f}s")
        
        # Load quality scorer
        logger.info("Loading quality scorer...")
        self.quality_scorer = MDVQS_scorer
        logger.info(f"Quality scorer loaded in {time.time() - start_time:.2f}s")
        
        # Initialise metrics
        logger.info("Initialising metrics...")
        self.metrics = {
            'slashes': [],
            'trust_updates': [],
            'quality_scores': [],
            'total_requests': 0,
            'successful_validations': 0,
            'failed_validations': 0,
            'spot_checks_performed': 0,
            'spot_checks_failed': 0,
            'avg_validation_time': 0,
            'last_validation_time': 0,
            'errors': [],
            'miner_scores': {}
        }
        logger.info(f"Metrics initialised in {time.time() - start_time:.2f}s")

        # Initialise scores
        logger.info("Initialising scores...")
        self.scores = {}
        logger.info(f"Scores initialised in {time.time() - start_time:.2f}s")
        
        # Track miner checkpoints for spot-checking
        logger.info("Initialising miner checkpoints...")
        self.miner_checkpoints: Dict[int, List[Dict]] = {}
        logger.info(f"Miner checkpoints initialsed in {time.time() - start_time:.2f}s")
        
        # Track last spot-check time for each miner
        logger.info("Initialising spot check tracking...")
        self.last_spot_check: Dict[int, float] = {}
        logger.info(f"Spot check tracking initialised in {time.time() - start_time:.2f}s")
        
        logger.info(f"Initialised ValidatorNeuron with weights: α={config.validator.alpha}, β={config.validator.beta}, γ={config.validator.gamma}")

        logger.info("Initialising video cache and request tracking...")
        self.video_cache = {}  # Cache for storing video generation results
        self.spot_check_interval = self.config.validator.spot_check_interval
        self.last_spot_check = time.time()
        self.active_requests = {}  # request_id -> dict with prompt/user/etc.
        self._pending_proofs = {}  # request_id -> {uid: (merkle_root, signature)}
        self._pending_prompt_queue = set()  # request_ids waiting for prompt
        logger.info(f"Video cache and request tracking initialised in {time.time() - start_time:.2f}s")
        
        # Start background thread to listen for Deposit events
        logger.info("Starting deposit listener thread...")
        listener_start = time.time()
        self._event_listener_thread = threading.Thread(
            target=start_deposit_listener, args=(self, contract, w3, self.config), daemon=True)
        self._event_listener_thread.start()
        logger.info(f"Deposit listener thread started successfully in {time.time() - listener_start:.2f}s")
        
        # Start REST API for prompt delivery in a background thread
        logger.info("Starting prompt API thread...")
        api_start = time.time()
        self._api_thread = threading.Thread(target=self._start_prompt_api, daemon=True)
        self._api_thread.start()
        logger.info(f"Prompt API thread started successfully in {time.time() - api_start:.2f}s")
        
        total_init_time = time.time() - start_time
        logger.info(f"Validator initialisation completed in {total_init_time:.2f} seconds")

    def _log_metrics(self):
        """Log current metrics to file"""
        try:
            metrics_file = 'validator_metrics.json'
            with open(metrics_file, 'w') as f:
                json.dump(self.metrics, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving metrics: {str(e)}")

    async def validate_video_generation(
        self,
        synapse: InferNet,
        response: InferNet,
        uid: int
    ) -> Tuple[float, str]:
        """
        Validates a video generation response using two-stage proof verification.
        
        Inputs:
        - synapse: Original request synapse
        - response: Miner's response
        - uid: Miner's UID
            
        Returns Tuple of (score, reason)
        """
        try:
            # Extract miner hotkey
            miner_hotkey = self.metagraph.hotkeys[uid]
            logger.info(f"Validating response from miner {uid} ({miner_hotkey})")
            
            # Decode video data
            video_bytes = response.deserialize()
            if not video_bytes:
                return 0.0, "No video data received"
            
            # Decode proof components
            challenge_bytes = base64.b64decode(response.challenge) if response.challenge else b""
            merkle_root = base64.b64decode(response.merkle_root) if response.merkle_root else b""
            signature = base64.b64decode(response.signature) if response.signature else b""
            
            # Stage 1: Verify signature and basic proof structure
            logger.info(f"Stage 1: Verifying signature for miner {uid}")
            if not verify_proof_signature(
                miner_hotkey,
                challenge_bytes,
                response.seed,
                video_bytes,
                merkle_root,
                signature
            ):
                return 0.0, "Invalid proof signature"
            
            # Stage 2: Perform Merkle spot-checking
            logger.info(f"Stage 2: Performing Merkle spot-checking for miner {uid}")
            spot_check_passed = await self._perform_merkle_spot_check(
                response, miner_hotkey, uid
            )
            if not spot_check_passed:
                return 0.0, "Failed Merkle spot-check verification"
            
            # Stage 3: Verify video authenticity
            logger.info(f"Stage 3: Verifying video authenticity for miner {uid}")
            
            # Create temporary file from Base64 data for video authenticity check
            video_bytes = base64.b64decode(response.video_data_b64)
            temp_video_path = f"temp_video_{uid}_{int(time.time())}.mp4"
            try:
                with open(temp_video_path, "wb") as f:
                    f.write(video_bytes)
                
                if not verify_video_authenticity_clip(temp_video_path):
                    return 0.0, "Video authenticity check failed"
                    
            finally:
                # Clean up temporary file
                if os.path.exists(temp_video_path):
                    os.remove(temp_video_path)
            
            # Stage 4: Compute quality score
            logger.info(f"Stage 4: Computing quality score for miner {uid}")
            
            # Create temporary file from Base64 data for quality scoring
            video_bytes = base64.b64decode(response.video_data_b64)
            temp_video_path = f"temp_video_{uid}_{int(time.time())}.mp4"
            try:
                with open(temp_video_path, "wb") as f:
                    f.write(video_bytes)
                
                quality_score = compute_quality_score_clip(
                    temp_video_path,
                    synapse.text_prompt
                )
                
            finally:
                # Clean up temporary file
                if os.path.exists(temp_video_path):
                    os.remove(temp_video_path)
            
            return quality_score, "Valid video generation"
            
        except Exception as e:
            logger.error(f"Validation error for miner {uid}: {str(e)}", exc_info=True)
            return 0.0, f"Validation error: {str(e)}"

    async def _perform_merkle_spot_check(
        self, 
        response: InferNet, 
        miner_hotkey: str, 
        uid: int
    ) -> bool:
        """
        Performs Merkle spot-checking using the data already provided in the response.
        
        Input:
            response: Miner's response containing proof
            miner_hotkey: Miner's hotkey address
            uid: Miner's UID
            
            Returns bool (True if spot-check passes, False otherwise)
        """
        try:
            # For now, implement a simplified spot-check that doesn't require RPC
            # This is a temporary solution until RPC methods are properly supported
            
            # Check if we have the basic proof components
            if not response.merkle_root or not response.signature or not response.timesteps:
                logger.error(f"Miner {uid} missing basic proof components")
                return False
            
            # Verify the signature first
            challenge_bytes = base64.b64decode(response.challenge) if response.challenge else b""
            merkle_root = base64.b64decode(response.merkle_root) if response.merkle_root else b""
            signature = base64.b64decode(response.signature) if response.signature else b""
            video_bytes = response.deserialize()
            
            if not verify_proof_signature(
                miner_hotkey,
                challenge_bytes,
                response.seed,
                video_bytes,
                merkle_root,
                signature
            ):
                logger.error(f"Miner {uid} signature verification failed")
                return False
            
            # For now, accept the proof if signature is valid
            # TODO: Implement full Merkle spot-checking when RPC is supported
            logger.info(f"Miner {uid} passed simplified spot-check (signature verified)")
            return True
            
        except Exception as e:
            logger.error(f"Error in Merkle spot-checking for miner {uid}: {str(e)}", exc_info=True)
            return False

    def _validate_request(self, synapse: InferNet) -> bool:
        """Validates the video generation request."""
        try:
            # Check required fields
            if not all([
                synapse.text_prompt,
                synapse.width,
                synapse.height,
                synapse.num_frames,
                synapse.fps
            ]):
                return False
            
            # Validate dimensions
            if not (0 < synapse.width <= 1024 and 0 < synapse.height <= 1024):
                return False
            
            # Validate frame count
            if not (0 < synapse.num_frames <= 100):
                return False
            
            # Validate FPS
            if not (0 < synapse.fps <= 60):
                return False
            
            return True
            
        except Exception:
            return False
    
    async def _get_miner_response(self, synapse: InferNet) -> InferNet:
        """Gets response from a miner."""
        
        return synapse

    async def forward(self):
        self.metrics['total_requests'] += 1
        request_id = f"req_{self.metrics['total_requests']}"
        logger.info(f"[{request_id}] Starting validation cycle")

        try:
            # --- Find requests with prompts available ---
            requests_with_prompts = [
                rid for rid, info in self.active_requests.items() 
                if info.get('status') == 'pending' and info.get('prompt')
            ]
            
            # Debug: Log available requests
            logger.debug(f"Active requests: {list(self.active_requests.keys())}")
            logger.debug(f"Requests with prompts: {requests_with_prompts}")
            
            # If no requests with prompts, check for any pending requests
            if not requests_with_prompts:
                pending_requests = [rid for rid, info in self.active_requests.items() if info.get('status') == 'pending']
                logger.debug(f"Pending requests without prompts: {pending_requests}")
                if not pending_requests:
                    logger.info("No pending on-chain requests to process.")
                    await asyncio.sleep(self.config.validator.poll_interval)
                    return
                # Use the first pending request (might not have prompt yet)
                real_request_id = pending_requests[0]
                logger.info(f"No requests with prompts available, using pending request: {real_request_id}")
            else:
                # Use the first request that has a prompt
                real_request_id = requests_with_prompts[0]
                logger.info(f"Found request with prompt: {real_request_id}")
                
            self.current_request_id = real_request_id
            self.active_requests[real_request_id]['status'] = 'processing'
            logger.info(f"Processing on-chain requestId: {real_request_id}")
            
            # --- Initialise per-miner proof map for this request ---
            self._pending_proofs[real_request_id] = {}
            
            # --- Use the verified prompt for this request ---
            prompt = self.active_requests[real_request_id].get('prompt')
            logger.info(f"[{request_id}] Retrieved prompt from active_requests: '{prompt}'")
            logger.info(f"[{request_id}] Active request data: {self.active_requests[real_request_id]}")
            
            if not prompt:
                logger.info(f"Prompt not yet delivered for request {real_request_id}, queuing and retrying later.")
                self.active_requests[real_request_id]['status'] = 'pending'
                self._pending_prompt_queue.add(real_request_id)
                await asyncio.sleep(self.config.validator.poll_interval)
                return

            # 1. Generate per-request randomness
            C = os.urandom(self.config.validator.challenge_bytes)
            # Convert challenge to base64 for JSON serialisation
            C_b64 = base64.b64encode(C).decode('utf-8')
            # Use the correct wallet attribute for the hotkey address
            hotkey_address = self.wallet.hotkey.ss58_address
            seed_bytes = hmac.new(hotkey_address.encode(), C, hashlib.sha256).digest()
            seed = int.from_bytes(seed_bytes[:8], 'big')
            logger.debug(f"[{request_id}] Generated challenge and seed: {seed}")

            # 2. Choose spot-check timesteps
            total_steps = self.config.diffusion.num_steps
            k = self.config.validator.num_checkpoints
            
            # Safety check: ensure k doesn't exceed total_steps
            if k > total_steps:
                logger.warning(f"[{request_id}]  WARNING: num_checkpoints ({k}) > num_steps ({total_steps}). Reducing k to {total_steps}")
                k = total_steps
            
            logger.debug(f"[{request_id}] Using k={k} checkpoints from {total_steps} total steps")
            
            # Do NOT pick k timesteps here or send them to the miner
            # Instead, miner will return all leaves, and we will sample k after receiving the response
            # 3. Build Merkle commitment

            # 4. Select miners
            uids = get_random_uids(self, k=self.config.neuron.sample_size)
            if len(uids) == 0:
                logger.warning(f"[{request_id}] No miners available")
                await asyncio.sleep(self.config.validator.poll_interval)
                return

            axons = [self.metagraph.axons[uid] for uid in uids]
            logger.info(f"[{request_id}] Selected miners: {uids}")

            # 5. Dispatch requests
            syn = InferNet(
                text_prompt=prompt,
                width=self.config.validator.width,
                height=self.config.validator.height,
                num_frames=self.config.validator.num_frames,
                fps=self.config.validator.fps,
                seed=seed,
                challenge=C_b64,  # Use base64-encoded challenge
                request_id=str(real_request_id),  # Pass the on-chain request ID to miners
                # Do NOT send timesteps
            )
            logger.info(f"[{request_id}] Using minimal parameters: width={self.config.validator.width}, height={self.config.validator.height}, frames={self.config.validator.num_frames}, fps={self.config.validator.fps}")
            logger.debug(f"[{request_id}] Dispatching requests to {len(axons)} miners")
            start_time = time.time()
            
            responses = await self.dendrite(
                axons=axons,
                synapse=syn,
                deserialize=False,
                timeout=self.config.validator.timeout
            )

            # 6. Process responses
            rewards = []
            result_details = []  # Collect per-miner result details
            video_paths = []     # Collect video paths for all miners
            for resp, uid in zip(responses, uids):
                try:
                    logger.debug(f"[{request_id}] Processing response from miner {uid}")
                    
                    # Handle different response types
                    if isinstance(resp, bytes):
                        logger.warning(f"Miner {uid} returned bytes instead of synapse object")
                        rewards.append(0.0)
                        result_details.append({
                            'uid': uid,
                            'status': 'bytes_response',
                            'score': 0.0,
                            'video_path': None,
                            'error': 'Miner returned bytes instead of synapse object',
                        })
                        continue
                    
                    # Unpack response using attribute access
                    logger.debug(f"[{request_id}] Miner {uid} response object attributes: {[attr for attr in dir(resp) if not attr.startswith('_')]}")
                    V_bytes_b64 = resp.video_data_b64
                    logger.debug(f"[{request_id}] Miner {uid} response video_data_b64: {type(V_bytes_b64)}, length: {len(V_bytes_b64) if V_bytes_b64 else 0}")
                    if not V_bytes_b64:
                        bt.logging.warning(f"Miner {uid} returned empty video")
                        rewards.append(0.0)
                        result_details.append({
                            'uid': uid,
                            'status': 'empty_video',
                            'score': 0.0,
                            'video_path': None,
                            'error': 'Empty video returned',
                        })
                        continue
                    V_bytes = base64.b64decode(V_bytes_b64)

                    proof = resp.proof 
                    if not proof:
                        bt.logging.warning(f"Miner {uid} returned no proof")
                        rewards.append(0.0)
                        result_details.append({
                            'uid': uid,
                            'status': 'no_proof',
                            'score': 0.0,
                            'video_path': None,
                            'error': 'No proof returned',
                        })
                        continue
                    merkle_root = proof['merkle_root']
                    signature = proof['signature']
                    seed_ret = proof['seed']
                    C_ret = proof['challenge']

                    # Decode base64 fields back to bytes
                    merkle_root = base64.b64decode(merkle_root) if isinstance(merkle_root, str) else merkle_root
                    signature = base64.b64decode(signature) if isinstance(signature, str) else signature

                    # Basic proof checks
                    if C_ret != C_b64 or seed_ret != seed:
                        logger.warning(f"[{request_id}] Bad challenge/seed from miner {uid}")
                        rewards.append(0.0)
                        result_details.append({
                            'uid': uid,
                            'status': 'bad_challenge_or_seed',
                            'score': 0.0,
                            'video_path': None,
                            'error': 'Bad challenge or seed',
                        })
                        continue

                    # Verify signature
                    logger.debug(f"[{request_id}] Miner {uid} signature verification:")
                    logger.debug(f"  - Original challenge C: {syn.challenge[:16]}... (length: {len(syn.challenge)})")
                    logger.debug(f"  - Response challenge C_ret: {resp.challenge}")
                    logger.debug(f"  - Decoded challenge: {base64.b64decode(resp.challenge).hex()[:16]}... (length: {len(base64.b64decode(resp.challenge))})")
                    logger.debug(f"  - Signature: {resp.signature[:16]}... (length: {len(resp.signature)})")
                    logger.debug(f"  - Video hash: {hashlib.sha256(V_bytes).hexdigest()[:16]}...")
                    logger.debug(f"  - Merkle root: {resp.merkle_root[:16]}...")
                    
                    # Use the decoded challenge from the response for verification
                    challenge_bytes = base64.b64decode(resp.challenge)
                    signature_bytes = base64.b64decode(resp.signature)
                    merkle_root_bytes = base64.b64decode(resp.merkle_root)
                    
                    logger.debug(f"[{request_id}] Miner {uid} decoded components:")
                    logger.debug(f"  - challenge_bytes: {challenge_bytes.hex()}")
                    logger.debug(f"  - signature_bytes: {signature_bytes.hex()}")
                    logger.debug(f"  - merkle_root_bytes: {merkle_root_bytes.hex()}")
                    
                    # Get the miner's hotkey from the metagraph
                    miner_hotkey_ss58 = self.metagraph.hotkeys[uid]
                    logger.debug(f"[{request_id}] Miner {uid} hotkey from metagraph: {miner_hotkey_ss58}")
                    logger.debug(f"[{request_id}] Miner {uid} axon.hotkey: {resp.axon.hotkey}")
                    
                    # Try to get the miner's public key from the axon
                    logger.debug(f"[{request_id}] Miner {uid} axon object attributes: {[attr for attr in dir(resp.axon) if not attr.startswith('_')]}")
                    try:
                        axon_public_key = resp.axon.public_key
                        logger.debug(f"[{request_id}] Miner {uid} axon.public_key: {axon_public_key.hex() if axon_public_key else 'None'}")
                    except Exception as e:
                        logger.debug(f"[{request_id}] Could not get miner public key from axon: {e}")
                        axon_public_key = None
                    
                    # Try alternative ways to get the public key
                    try:
                        axon_pubkey_alt = getattr(resp.axon, 'pubkey', None)
                        logger.debug(f"[{request_id}] Miner {uid} axon.pubkey (alt): {axon_pubkey_alt.hex() if axon_pubkey_alt else 'None'}")
                    except Exception as e:
                        logger.debug(f"[{request_id}] Could not get axon.pubkey (alt): {e}")
                        axon_pubkey_alt = None
                    
                    # Try to create a keypair from the hotkey to get the public key
                    try:
                        miner_keypair = bt.Keypair(ss58_address=miner_hotkey_ss58)
                        metagraph_public_key = miner_keypair.public_key
                        logger.debug(f"[{request_id}] Miner {uid} public key from keypair: {metagraph_public_key.hex()}")
                    except Exception as e:
                        logger.debug(f"[{request_id}] Could not create keypair from hotkey: {e}")
                        metagraph_public_key = None
                    
                    # Use the public key from the axon if available, otherwise from keypair
                    verification_public_key = axon_public_key if axon_public_key is not None else (axon_pubkey_alt if axon_pubkey_alt is not None else metagraph_public_key)
                    logger.debug(f"[{request_id}] Miner {uid} using public key for verification: {verification_public_key.hex() if verification_public_key else 'None'}")
                    
                    if verification_public_key is None:
                        logger.error(f"[{request_id}] No public key available for miner {uid} verification!")
                        logger.error(f"  axon public key: {axon_public_key.hex() if axon_public_key else 'None'}")
                        logger.error(f"  axon pubkey alt: {axon_pubkey_alt.hex() if axon_pubkey_alt else 'None'}")
                        logger.error(f"  metagraph public key: {metagraph_public_key.hex() if metagraph_public_key else 'None'}")
                    
                    if not verify_proof_signature(
                        miner_hotkey_ss58=miner_hotkey_ss58,
                        miner_public_key=verification_public_key,
                        challenge=challenge_bytes,
                        seed=resp.seed,
                        video_bytes=V_bytes,
                        merkle_root=merkle_root_bytes,
                        signature=signature_bytes,
                    ):
                        logger.warning(f"[{request_id}] Invalid signature from miner {uid}")
                        rewards.append(0.0)
                        result_details.append({
                            'uid': uid,
                            'status': 'invalid_signature',
                            'score': 0.0,
                            'video_path': None,
                            'error': 'Invalid signature',
                        })
                        continue

                    # --- Store per-miner proof for this request ---
                    self._pending_proofs[real_request_id][uid] = (merkle_root, signature)

                    # --- TRUE COMMIT-THEN-REVEAL: Request k leaves after commitment ---
                    logger.debug(f"[{request_id}] STARTING SPOT-CHECK for miner {uid}")
                    logger.debug(f"[{request_id}] k (num_checkpoints): {k}")
                    logger.debug(f"[{request_id}] Response object type: {type(resp)}")
                    logger.debug(f"[{request_id}] Response object attributes: {[attr for attr in dir(resp) if not attr.startswith('_')]}")
                    
                    # Check if timesteps attribute exists
                    if hasattr(resp, 'timesteps'):
                        logger.debug(f"[{request_id}] timesteps attribute found")
                        all_timesteps = resp.timesteps
                        logger.debug(f"[{request_id}] all_timesteps type: {type(all_timesteps)}")
                        logger.debug(f"[{request_id}] all_timesteps value: {all_timesteps}")
                        logger.debug(f"[{request_id}] all_timesteps length: {len(all_timesteps) if all_timesteps else 0}")
                    else:
                        logger.debug(f"[{request_id}] timesteps attribute NOT found")
                        all_timesteps = None
                    
                    # Check other potential attributes that might contain timesteps
                    for attr_name in ['timestep_list', 'step_list', 'steps', 'diffusion_steps', 'denoising_steps']:
                        if hasattr(resp, attr_name):
                            logger.debug(f"[{request_id}] Found alternative attribute: {attr_name} = {getattr(resp, attr_name)}")
                    
                    # Check if proof contains timesteps
                    if hasattr(resp, 'proof') and resp.proof:
                        logger.debug(f"[{request_id}] Proof keys: {list(resp.proof.keys()) if isinstance(resp.proof, dict) else 'Not a dict'}")
                        if isinstance(resp.proof, dict) and 'timesteps' in resp.proof:
                            logger.debug(f"[{request_id}] Found timesteps in proof: {resp.proof['timesteps']}")
                    
                    self.metrics['spot_checks_performed'] += 1
                    
                    # Get the full set of timesteps from the response (should be T long)
                    if not all_timesteps or len(all_timesteps) < k:
                        logger.warning(f"[{request_id}] SPOT-CHECK FAILED: Not enough timesteps")
                        logger.warning(f"[{request_id}]   - all_timesteps: {all_timesteps}")
                        logger.warning(f"[{request_id}]   - len(all_timesteps): {len(all_timesteps) if all_timesteps else 0}")
                        logger.warning(f"[{request_id}]   - required k: {k}")
                        logger.warning(f"[{request_id}]   - config.diffusion.num_steps: {self.config.diffusion.num_steps}")
                        rewards.append(0.0)
                        result_details.append({
                            'uid': uid,
                            'status': 'not_enough_timesteps',
                            'score': 0.0,
                            'video_path': None,
                            'error': f'Not enough timesteps for spot-checking. Got {len(all_timesteps) if all_timesteps else 0}, need {k}',
                        })
                        continue
                    
                    logger.debug(f"[{request_id}] SUFFICIENT TIMESTEPS: {len(all_timesteps)} >= {k}")
                    spot_check_indices = random.sample(all_timesteps, k)
                    logger.debug(f"[{request_id}] Selected spot-check indices: {spot_check_indices}")
                    
                    # Call miner's open_leaves RPC to get only the requested leaves
                    axon = self.metagraph.axons[uid]
                    logger.debug(f"[{request_id}] Miner {uid} axon: {axon}")
                    
                    # Defensive: ensure open_leaves is available
                    try:
                        # Pass request_id and caller_hotkey to tie the reveal to the specific request
                        caller_hotkey = self.wallet.hotkey.ss58_address
                        logger.debug(f"[{request_id}] Caller hotkey: {caller_hotkey}")
                        logger.debug(f"[{request_id}] Request ID for RPC: {request_id}")
                        
                        # Try different RPC call methods for compatibility
                        try:
                            logger.debug(f"[{request_id}] Attempting spot-check via main forward method")
                            # Use the main forward method with a special flag for spot-checking
                            spot_check_synapse = InferNet(
                                text_prompt="",  # Required field
                                width=1, height=1, num_frames=1, fps=1,  # Minimal values
                                seed=0,  # Use seed=0 to indicate this is a spot-check request
                                challenge="",  # Empty challenge for spot-check
                                request_id=str(real_request_id)  # Use the same real_request_id as the original request
                            )
                            
                            spot_check_response = await self.dendrite(
                                axons=[axon],
                                synapse=spot_check_synapse,
                                deserialize=False,
                                timeout=self.config.validator.timeout
                            )
                            logger.debug(f"[{request_id}] Spot-check response: {type(spot_check_response)}, length: {len(spot_check_response) if spot_check_response else 0}")
                            if spot_check_response and len(spot_check_response) > 0:
                                # The miner now returns all leaves in proof.leaf_data for spot-check requests
                                spot_check_syn = spot_check_response[0]
                                if hasattr(spot_check_syn, 'proof') and spot_check_syn.proof and 'leaf_data' in spot_check_syn.proof:
                                    leaves_result = spot_check_syn.proof['leaf_data']
                                    logger.debug(f"[{request_id}] ✅ Spot-check succeeded, leaves_result: {type(leaves_result)}")
                                else:
                                    leaves_result = None
                                    logger.debug(f"[{request_id}] Spot-check failed: no leaf_data in proof")
                            else:
                                leaves_result = None
                                logger.debug(f"[{request_id}] Spot-check failed: no response")
                        except Exception as e:
                            logger.warning(f"[{request_id}] SPOT-CHECK FAILED: {e}")
                            leaves_result = None
                            logger.debug(f"[{request_id}] Exception details:", exc_info=True)
                    except Exception as e:
                        logger.warning(f"[{request_id}] RPC CALL FAILED: Failed to call open_leaves on miner {uid}: {e}")
                        logger.debug(f"[{request_id}] Exception details:", exc_info=True)
                        rewards.append(0.0)
                        result_details.append({
                            'uid': uid,
                            'status': 'open_leaves_failed',
                            'score': 0.0,
                            'video_path': None,
                            'error': f'open_leaves failed: {e}',
                        })
                        continue
                    
                    logger.debug(f"[{request_id}] leaves_result type: {type(leaves_result)}")
                    logger.debug(f"[{request_id}] leaves_result: {leaves_result}")
                    
                    if not leaves_result or not isinstance(leaves_result, dict):
                        logger.warning(f"[{request_id}] NO LEAVES DATA: open_leaves returned no data for miner {uid}")
                        logger.warning(f"[{request_id}]   - leaves_result: {leaves_result}")
                        logger.warning(f"[{request_id}]   - type: {type(leaves_result)}")
                        rewards.append(0.0)
                        result_details.append({
                            'uid': uid,
                            'status': 'open_leaves_no_data',
                            'score': 0.0,
                            'video_path': None,
                            'error': f'open_leaves returned no data. Type: {type(leaves_result)}, Value: {leaves_result}',
                        })
                        continue
                    
                    logger.debug(f"[{request_id}] LEAVES DATA RECEIVED: {len(leaves_result)} leaves")
                    logger.debug(f"[{request_id}] Leaves keys: {list(leaves_result.keys())}")
                    
                    ok = True
                    for i, t in enumerate(spot_check_indices):
                        logger.debug(f"[{request_id}] Verifying leaf for timestep {t}")
                        # Convert timestep to string since miner returns string keys
                        t_str = str(t)
                        leaf = leaves_result.get(t_str)
                        logger.debug(f"[{request_id}] Leaf for t={t} (key={t_str}): {type(leaf)}, value: {leaf}")
                        
                        if not leaf or len(leaf) != 3:
                            logger.warning(f"[{request_id}] MALFORMED LEAF: Missing or malformed leaf for t={t} from miner {uid}")
                            logger.warning(f"[{request_id}]   - leaf: {leaf}")
                            logger.warning(f"[{request_id}]   - len(leaf): {len(leaf) if leaf else 0}")
                            ok = False
                            break
                        
                        z_b, eps_b, proof_path = leaf
                        logger.debug(f"[{request_id}] Leaf components for t={t}:")
                        logger.debug(f"[{request_id}]   - z_b type: {type(z_b)}, length: {len(z_b) if z_b else 0}")
                        logger.debug(f"[{request_id}]   - eps_b type: {type(eps_b)}, length: {len(eps_b) if eps_b else 0}")
                        logger.debug(f"[{request_id}]   - proof_path type: {type(proof_path)}, length: {len(proof_path) if proof_path else 0}")
                        
                        # Decode base64 strings back to bytes
                        z_bytes = base64.b64decode(z_b)
                        eps_bytes = base64.b64decode(eps_b)
                        proof_path_bytes = [base64.b64decode(p) for p in proof_path]
                        
                        # Verify Merkle path
                        leaf_hash = hashlib.sha256(t.to_bytes(2,'big') + z_bytes + eps_bytes).digest()
                        logger.debug(f"[{request_id}] Leaf hash for t={t}: {leaf_hash.hex()[:16]}...")
                        
                        if not verify_merkle_leaf(leaf_hash, proof_path_bytes, merkle_root):
                            logger.warning(f"[{request_id}] MERKLE VERIFICATION FAILED: Merkle verification failed for miner {uid} at timestep {t}")
                            ok = False
                            break
                        
                        logger.debug(f"[{request_id}] Merkle verification passed for t={t}")
                        
                        # Run UNet step - use step index i instead of timestep t
                        if not run_unet_step(z_bytes, eps_bytes, self.config.unet_config, i):
                            logger.warning(f"[{request_id}] UNET VERIFICATION FAILED: UNet step verification failed for miner {uid} at timestep {t} (step {i})")
                            ok = False
                            break
                        
                        logger.debug(f"[{request_id}] UNet verification passed for t={t} (step {i})")
                    
                    if not ok:
                        self.metrics['spot_checks_failed'] += 1
                        logger.warning(f"[{request_id}] SPOT-CHECK FAILED: Spot-check failed for miner {uid}")
                        rewards.append(0.0)
                        result_details.append({
                            'uid': uid,
                            'status': 'spot_check_failed',
                            'score': 0.0,
                            'video_path': None,
                            'error': 'Spot-check failed',
                        })
                        continue
                    
                    logger.debug(f"[{request_id}] SPOT-CHECK PASSED: All verifications successful for miner {uid}")

                    # Save video for analysis
                    video_path = OUTPUT_DIR / f"video_{uid}_{int(time.time())}.mp4"
                    with open(video_path, 'wb') as f:
                        f.write(V_bytes)
                    logger.debug(f"[{request_id}] Saved video from miner {uid} to {video_path}")
                    video_paths.append(str(video_path))

                    # Verify video authenticity
                    if not verify_video_authenticity_clip(str(video_path)):
                        logger.warning(f"[{request_id}] Cheat detected in video from miner {uid}")
                        rewards.append(0.0)
                        result_details.append({
                            'uid': uid,
                            'status': 'cheat_detected',
                            'score': 0.0,
                            'video_path': str(video_path),
                            'error': 'Cheat detected in video',
                        })
                        continue

                    # Compute CLIP quality score
                    logger.debug(f"[{request_id}] Computing CLIP quality score for miner {uid}")
                    logger.debug(f"[{request_id}] Using prompt for CLIP scoring: '{prompt}'")
                    logger.debug(f"[{request_id}] Prompt type: {type(prompt)}")
                    logger.debug(f"[{request_id}] Prompt length: {len(prompt) if prompt else 'None'}")
                    logger.debug(f"[{request_id}] Prompt is None: {prompt is None}")
                    logger.debug(f"[{request_id}] Prompt is empty: {prompt == ''}")
                    
                    # Check if prompt is valid before calling CLIP
                    if not prompt or prompt.strip() == "":
                        logger.warning(f"[{request_id}] WARNING: Prompt is empty or None, using default prompt")
                        prompt = "a video"
                        logger.debug(f"[{request_id}] Using fallback prompt: '{prompt}'")
                    
                    score = compute_quality_score_clip(str(video_path), prompt)
                    logger.info(f"[{request_id}] Miner {uid} CLIP score: {score:.3f}")
                    rewards.append(score)
                    result_details.append({
                        'uid': uid,
                        'status': 'success',
                        'score': score,
                        'video_path': str(video_path),
                        'error': None,
                    })
                    # Update miner metrics
                    if uid not in self.metrics['miner_scores']:
                        self.metrics['miner_scores'][uid] = []
                    self.metrics['miner_scores'][uid].append({
                        'timestamp': datetime.now().isoformat(),
                        'score': score,
                    })
                except Exception as e:
                    logger.error(f"[{request_id}] Error processing miner {uid}: {str(e)}", exc_info=True)
                    self.metrics['errors'].append({
                        'timestamp': datetime.now().isoformat(),
                        'error': str(e),
                        'type': 'miner_processing_error',
                        'miner_uid': uid,
                        'request_id': request_id
                    })
                    rewards.append(0.0)
                    result_details.append({
                        'uid': uid,
                        'status': 'exception',
                        'score': 0.0,
                        'video_path': None,
                        'error': str(e),
                    })

            # 7. Update scores
            if rewards:
                validation_time = time.time() - start_time
                self.metrics['last_validation_time'] = validation_time
                self.metrics['avg_validation_time'] = (
                    (self.metrics['avg_validation_time'] * self.metrics['successful_validations'] + validation_time) /
                    (self.metrics['successful_validations'] + 1)
                )
                self.metrics['successful_validations'] += 1
                
                logger.info(f"[{request_id}] Updating scores for {len(rewards)} miners")
                self.update_scores(rewards, uids)
                logger.info(f"[{request_id}] Scores updated successfully")
                
                # Store result for retrieval via API
                if real_request_id in self.active_requests:
                    logger.info(f"[{request_id}] Storing result in active_requests")
                    self.active_requests[real_request_id]['result'] = {
                        'miners': result_details,
                        'uids': uids,
                        'scores': rewards,
                        'video_paths': video_paths,
                        'timestamp': datetime.now().isoformat(),
                        'status': self.active_requests[real_request_id].get('status'),
                        'prompt': prompt,
                    }
                    logger.info(f"[{request_id}] Result stored in active_requests")
                else:
                    logger.warning(f"[{request_id}] real_request_id {real_request_id} not found in active_requests")
            else:
                logger.warning(f"[{request_id}] No rewards to update")
            
            # Log metrics
            logger.info(f"[{request_id}] Logging metrics")
            self._log_metrics()
            logger.info(f"[{request_id}] Metrics logged")

            # Save results to file for API access
            logger.info(f"[{request_id}] Starting result file saving process")
            result_file = f"results_{real_request_id}.json"
            result_data = {
                'request_id': real_request_id,
                'status': 'completed',
                'result': {
                    'miners': result_details,
                    'total_miners': len(result_details),
                    'successful_miners': len([m for m in result_details if m['status'] == 'success']),
                    'avg_score': np.mean([m['score'] for m in result_details if m['status'] == 'success' and m['score'] > 0]) if any(m['status'] == 'success' and m['score'] > 0 for m in result_details) else 0.0
                }
            }
            
            logger.info(f"[{request_id}] Saving results to file: {result_file}")
            logger.info(f"[{request_id}] Result data structure: {list(result_data.keys())}")
            logger.info(f"[{request_id}] Number of miners in result: {len(result_details)}")
            
            for i, miner in enumerate(result_details):
                logger.info(f"[{request_id}] Miner {i}: uid={miner.get('uid')}, status={miner.get('status')}, score={miner.get('score')}")
                logger.info(f"[{request_id}] Miner {i}: video_path={miner.get('video_path')}")
                
                # Check if video file exists
                if miner.get('video_path'):
                    video_exists = os.path.exists(miner['video_path'])
                    logger.info(f"[{request_id}] Miner {i}: video file exists: {video_exists}")
                    if not video_exists:
                        logger.warning(f"[{request_id}] Miner {i}: video file missing: {miner['video_path']}")
            
            try:
                # Convert NumPy types to native Python types for JSON serialisation
                def convert_numpy_types(obj):
                    if hasattr(obj, 'item'):  # NumPy scalar
                        return obj.item()
                    elif isinstance(obj, dict):
                        return {key: convert_numpy_types(value) for key, value in obj.items()}
                    elif isinstance(obj, list):
                        return [convert_numpy_types(item) for item in obj]
                    else:
                        return obj
                
                # Convert the result data to JSON-serialisable types
                result_data = convert_numpy_types(result_data)
                
                with open(result_file, 'w') as f:
                    json.dump(result_data, f, indent=2)
                logger.info(f"[{request_id}] ✅ Results saved successfully to {result_file}")
                
                # Verify the file was written correctly
                if os.path.exists(result_file):
                    file_size = os.path.getsize(result_file)
                    logger.info(f"[{request_id}] Result file size: {file_size} bytes")
                    
                    # Test reading the file back
                    with open(result_file, 'r') as f:
                        test_data = json.load(f)
                    logger.info(f"[{request_id}] Result file can be read back successfully")
                    logger.info(f"[{request_id}] Test read - miners count: {len(test_data.get('result', {}).get('miners', []))}")
                else:
                    logger.error(f"[{request_id}] Result file was not created: {result_file}")
                    
            except Exception as e:
                logger.error(f"[{request_id}] Error saving results to file: {str(e)}")
                logger.error(f"[{request_id}] Error details: {type(e).__name__}")
                import traceback
                logger.error(f"[{request_id}] Traceback: {traceback.format_exc()}")

        except Exception as e:
            self.metrics['failed_validations'] += 1
            self.metrics['errors'].append({
                'timestamp': datetime.now().isoformat(),
                'error': str(e),
                'type': 'validation_cycle_error',
                'request_id': request_id
            })
            logger.error(f"[{request_id}] Error in validation cycle: {str(e)}", exc_info=True)
            logger.error(f"[{request_id}] This error prevented result saving from executing")

        # 8. Throttle
        logger.info(f"[{request_id}] Throttling for {self.config.validator.poll_interval} seconds")
        await asyncio.sleep(self.config.validator.poll_interval)
        logger.info(f"[{request_id}] Throttling complete, continuing to next iteration")

    def trigger_refund(self, request_id):
        """
        Triggers a refundUnused transaction for the given request_id.
        Returns the transaction hash.
        """
        logger.info(f"Attempting to trigger refund for request {request_id}")
        try:
            tx_data = contract.functions.refundUnused(int(request_id))
            tx_dict = {
                'from': VALIDATOR_ETH_ADDRESS,
                'nonce': w3.eth.get_transaction_count(VALIDATOR_ETH_ADDRESS),
            }
            try:
                tx_dict['gas'] = tx_data.estimate_gas(tx_dict)
            except Exception as e:
                logger.warning(f"Gas estimation failed for refundUnused, using fallback: {e}")
                tx_dict['gas'] = 100000
            try:
                tx_dict['gasPrice'] = w3.eth.gas_price
            except Exception as e:
                logger.warning(f"Failed to fetch gas price, using fallback: {e}")
                tx_dict['gasPrice'] = w3.to_wei('5', 'gwei')
            tx = tx_data.build_transaction(tx_dict)
            signed_tx = w3.eth.account.sign_transaction(tx, private_key=VALIDATOR_PRIVATE_KEY)
            tx_hash = w3.eth.send_raw_transaction(signed_tx.raw_transaction)
            logger.info(f"refundUnused tx: {tx_hash.hex()}")
            return tx_hash.hex()
        except Exception as e:
            logger.error(f"Failed to call refundUnused for request {request_id}: {e}")
            raise

    def receive_user_prompt(self, request_id, user_prompt):
        """
        Called when the user delivers their prompt off-chain via REST
        Verifies the prompt hash matches the on-chain commitment, then stores it for use.
        """
        if request_id not in self.active_requests:
            raise Exception(f"Unknown request_id {request_id}")
        prompt_hash_onchain = self.active_requests[request_id]['promptHash']
        # Use the same hash computation as the frontend: keccak256(stringToBytes(prompt))
        prompt_hash_local = keccak(user_prompt.encode('utf-8'))
        logger.info(f"Hash comparison for request {request_id}:")
        logger.info(f"  On-chain hash: {prompt_hash_onchain.hex()}")
        logger.info(f"  Local hash: {prompt_hash_local.hex()}")
        logger.info(f"  Prompt: '{user_prompt}'")
        if prompt_hash_local != prompt_hash_onchain:
            raise Exception("Prompt mismatch: user did not commit to this prompt")
        self.active_requests[request_id]['prompt'] = user_prompt
        logger.info(f"Received and verified prompt for request {request_id}")

    def update_scores(self, rewards: List[float], uids: List[int]):
        """
        Update miner scores with validation results.
        Also record submissions and distribute rewards on-chain.
        Only processes the current request being validated.
        Input:
            rewards: List of reward scores
            uids: List of miner UIDs
        """
        try:
            # Only process the current request
            request_id = getattr(self, 'current_request_id', None)
            if request_id is None or request_id not in self.active_requests:
                logger.warning("No current request to pay out.")
                return
            req_info = self.active_requests[request_id]
            if req_info.get('status') != 'processing':
                logger.warning(f"Request {request_id} not in processing state.")
                return
            # --- Nonce management: fetch base nonce once ---
            base_nonce = w3.eth.get_transaction_count(VALIDATOR_ETH_ADDRESS)
            num_miners = len(rewards)
            for i, (reward, uid) in enumerate(zip(rewards, uids)):
                # Get current scores - ensure scores is a dictionary
                if not hasattr(self, 'scores') or not isinstance(self.scores, dict):
                    self.scores = {}
                current_scores = self.scores.get(uid, 0.0)
                # Update with moving average
                new_score = (
                    (1 - self.config.neuron.moving_average_alpha) * current_scores +
                    self.config.neuron.moving_average_alpha * reward
                )
                self.scores[uid] = new_score
                logger.info(
                    f"Updated scores for miner {uid}: "
                    f"reward={reward:.3f}, "
                    f"moving_avg={new_score:.3f}"
                )
                # --- On-chain: record submission ---
                try:
                    miner_eth_address = self.metagraph.axons[uid].eth_address if hasattr(self.metagraph.axons[uid], 'eth_address') else '0xMinerEthAddress'  # TODO: update
                    # Look up the per-miner proof for this request
                    if request_id not in self._pending_proofs or uid not in self._pending_proofs[request_id]:
                        logger.error(f"No proof found for miner {uid} on request {request_id}")
                        continue
                    merkle_root, signature = self._pending_proofs[request_id][uid]
                    # proof_bytes = encode_abi(['bytes32', 'bytes'], [merkle_root, signature])
                    proof_bytes = merkle_root + signature
                    tx_data = contract.functions.recordSubmission(
                        int(request_id),
                        miner_eth_address,
                        int(reward * 1e6),
                        proof_bytes
                    )
                    tx_dict = {
                        'from': VALIDATOR_ETH_ADDRESS,
                        'nonce': base_nonce + i,
                    }
                    # --- Dynamic gas estimation and pricing ---
                    try:
                        tx_dict['gas'] = tx_data.estimate_gas(tx_dict)
                    except Exception as e:
                        logger.warning(f"Gas estimation failed for recordSubmission, using fallback: {e}")
                        tx_dict['gas'] = 300000
                    try:
                        tx_dict['gasPrice'] = w3.eth.gas_price
                    except Exception as e:
                        logger.warning(f"Failed to fetch gas price, using fallback: {e}")
                        tx_dict['gasPrice'] = w3.to_wei('5', 'gwei')
                    tx = tx_data.build_transaction(tx_dict)
                    signed_tx = w3.eth.account.sign_transaction(tx, private_key=VALIDATOR_PRIVATE_KEY)
                    tx_hash = w3.eth.send_raw_transaction(signed_tx.raw_transaction)
                    logger.info(f"recordSubmission tx: {tx_hash.hex()}")
                except Exception as e:
                    logger.error(f"On-chain recordSubmission failed for miner {uid}: {e}")
            # --- On-chain: distribute rewards after all submissions ---
            try:
                tx_data = contract.functions.distributeRewards(
                    int(request_id)
                )
                tx_dict = {
                    'from': VALIDATOR_ETH_ADDRESS,
                    'nonce': base_nonce + num_miners,
                }
                # --- Dynamic gas estimation and pricing ---
                try:
                    tx_dict['gas'] = tx_data.estimate_gas(tx_dict)
                except Exception as e:
                    logger.warning(f"Gas estimation failed for distributeRewards, using fallback: {e}")
                    tx_dict['gas'] = 200000
                try:
                    tx_dict['gasPrice'] = w3.eth.gas_price
                except Exception as e:
                    logger.warning(f"Failed to fetch gas price, using fallback: {e}")
                    tx_dict['gasPrice'] = w3.to_wei('5', 'gwei')
                tx = tx_data.build_transaction(tx_dict)
                signed_tx = w3.eth.account.sign_transaction(tx, private_key=VALIDATOR_PRIVATE_KEY)
                tx_hash = w3.eth.send_raw_transaction(signed_tx.raw_transaction)
                logger.info(f"distributeRewards tx: {tx_hash.hex()}")
                # Mark request as completed and clear current_request_id
                req_info['status'] = 'completed'
                self.current_request_id = None
                # Clean up proofs for this request
                if request_id in self._pending_proofs:
                    del self._pending_proofs[request_id]
            except Exception as e:
                logger.error(f"On-chain distributeRewards failed: {e}")
        except Exception as e:
            logger.error(f"Error updating scores: {str(e)}", exc_info=True)
            self.metrics['errors'].append({
                'timestamp': datetime.now().isoformat(),
                'error': str(e),
                'type': 'score_update_error'
            })

    def _start_prompt_api(self):
        app = create_prompt_api(self)
        app.run(host='0.0.0.0', port=8080)

    def update_trust_weight(self, miner_hotkey: str, detected_fraud: bool):
        """Updates miner trust weight based on fraud detection."""
        try:
            if detected_fraud:
                # Apply trust decay on fraud
                self.trust_weights[miner_hotkey] *= (1 - self.config.trust_decay)
                logger.info(f"Trust weight decayed for {miner_hotkey} to {self.trust_weights[miner_hotkey]:.4f}")
            else:
                # Honest drift towards equilibrium
                self.trust_weights[miner_hotkey] += self.config.honest_drift * (
                    1/len(self.trust_weights) - self.trust_weights[miner_hotkey]
                )
                logger.info(f"Trust weight updated for {miner_hotkey} to {self.trust_weights[miner_hotkey]:.4f}")
            
            # Log update
            self.metrics['trust_updates'].append({
                'timestamp': datetime.now().isoformat(),
                'miner': miner_hotkey,
                'new_weight': self.trust_weights[miner_hotkey],
                'detected_fraud': detected_fraud
            })
            
        except Exception as e:
            logger.error(f"Error updating trust weight for {miner_hotkey}: {str(e)}")

    def slash_miner(self, miner_hotkey: str):
        """Slash miner's stake for detected fraud."""
        try:
            # Get current stake
            current_stake = self.metagraph.S[self.metagraph.hotkeys.index(miner_hotkey)]
            
            # Calculate slash amount
            slash_amount = current_stake * self.config.slash_fraction
            
            # Update stake
            self.metagraph.S[self.metagraph.hotkeys.index(miner_hotkey)] -= slash_amount
            
            # Log slash event
            self.metrics['slashes'].append({
                'timestamp': datetime.now().isoformat(),
                'miner': miner_hotkey,
                'amount': slash_amount,
                'remaining_stake': current_stake - slash_amount
            })
            
            logger.info(f"Slashed {slash_amount:.4f} TAO from {miner_hotkey}")
            
        except Exception as e:
            logger.error(f"Error slashing miner {miner_hotkey}: {str(e)}")

    def verify_and_score(self, miner_hotkey: str, video_path: str, prompt: str, proof: Dict) -> Tuple[bool, float]:
        """Verify proof and score video quality."""
        try:
            # Load bytes if given a path
            if isinstance(video_path, str):
                with open(video_path, "rb") as _f:
                    video_bytes = _f.read()
            else:
                video_bytes = video_path

            # Verify proof of inference
            if not verify_proof_of_inference(
                miner_hotkey,
                video_bytes,
                proof,
                self.config.unet_config
            ):
                logger.warning(f"Proof verification failed for {miner_hotkey}")
                self.slash_miner(miner_hotkey)
                self.update_trust_weight(miner_hotkey, True)
                return False, 0.0
            
            # Score video quality using frame-wise CLIP similarity
            quality_score = self.quality_scorer.compute_quality_score(video_path, prompt)
            
            # Log quality score
            self.metrics['quality_scores'].append({
                'timestamp': datetime.now().isoformat(),
                'miner': miner_hotkey,
                'quality_score': quality_score
            })
            
            # Update trust weight based on quality
            if quality_score < self.config.quality_threshold:
                self.update_trust_weight(miner_hotkey, True)
                return False, quality_score
            
            # Update trust weight for honest execution
            self.update_trust_weight(miner_hotkey, False)
            return True, quality_score
            
        except Exception as e:
            logger.error(f"Error in verify_and_score for {miner_hotkey}: {str(e)}")
            return False, 0.0

    def clear_state(self):
        """Clear all validator state to start fresh."""
        logger.info("Clearing validator state...")
        self.active_requests = {}
        self._pending_proofs = {}
        self._pending_prompt_queue = set()
        self.current_request_id = None
        self.video_cache = {}
        self.miner_checkpoints = {}
        self.last_spot_check = {}
        logger.info("Validator state cleared successfully")

# Entry point
if __name__ == '__main__':
    # Use Bittensor's default config but override with our minimal settings
    cfg = ValidatorNeuron.config()
    
    # Override with our minimal settings
    cfg.validator.width = 128
    cfg.validator.height = 128
    cfg.validator.num_frames = 3
    cfg.validator.fps = 1
    cfg.diffusion.num_steps = 4
    cfg.unet_config = {
        'latent_channels': 4,
        'latent_height': 16,  # 128 // 8
        'latent_width': 16,   # 128 // 8
        'alphas': [0.9999, 0.9998, 0.9997, 0.9996]  # 4 steps
    }
    
    neuron = ValidatorNeuron(config=cfg)
    neuron.run()
