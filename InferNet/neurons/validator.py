# Validator implementation combining proof-of-inference, spot-check Merkle commitment,
# cheat-detection, scoring, and reward updates.

import os
import sys
import time
import random
import hmac
import hashlib
import asyncio
import bittensor as bt
import logging
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from web3 import Web3
import threading

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eth_utils import keccak
from flask import Flask, request, jsonify
from api.prompt_api import create_prompt_api
from events.deposit_listener import start_deposit_listener



from template.base.validator import BaseValidatorNeuron
from template.protocol import InferNet
from template.validator.proof import verify_proof_of_inference
from template.validator.scoring import compute_md_vqs, verify_video_authenticity
from template.utils.uids import get_random_uids
from template.validator.proof import (
    verify_proof_signature,
    build_merkle_root,
    verify_merkle_leaf,
    run_unet_step
)
from template.validator.proof import derive_seed
from template.validator.scoring import MDVQS

from dotenv import load_dotenv

load_dotenv()


logging.basicConfig(
    level=logging.DEBUG,
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
contract = w3.eth.contract(address=CONTRACT_ADDRESS, abi=contract_abi)

class ValidatorNeuron(BaseValidatorNeuron):
    """
    Full Validator neuron with enhanced logging and debugging.
    """

    def __init__(self, config: Optional[bt.config] = None):
        super().__init__(config)
        
        # Initialise trust weights
        self.trust_weights = {hotkey: 1.0 for hotkey in self.metagraph.hotkeys}
        
        # Load quality scorer
        self.quality_scorer = VideoScorer()
        
        # Initialise metrics
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

        # Initialise scores
        self.scores = {}
        
        # Track miner checkpoints for spot-checking
        self.miner_checkpoints: Dict[int, List[Dict]] = {}
        
        # Track last spot-check time for each miner
        self.last_spot_check: Dict[int, float] = {}
        
        logger.info(f"Initialised ValidatorNeuron with weights: α={config.validator.alpha}, β={config.validator.beta}, γ={config.validator.gamma}")

        self.video_cache = {}  # Cache for storing video generation results
        self.spot_check_interval = self.config.validator.spot_check_interval
        self.last_spot_check = time.time()
        self.active_requests = {}  # request_id -> dict with prompt/user/etc.
        self._pending_proofs = {}  # request_id -> {uid: (merkle_root, signature)}
        self._pending_prompt_queue = set()  # request_ids waiting for prompt
        # Start background thread to listen for Deposit events
        self._event_listener_thread = threading.Thread(
            target=start_deposit_listener, args=(self, contract, w3, self.config), daemon=True)
        self._event_listener_thread.start()
        # Start REST API for prompt delivery in a background thread
        self._api_thread = threading.Thread(target=self._start_prompt_api, daemon=True)
        self._api_thread.start()

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
        Validates a video generation response.
        
        Inputs:
        - synapse: Original request synapse
        - response: Miner's response
        - uid: Miner's UID
            
        Returns Tuple of (score, reason)
        """
        try:
            # 1. Verify proof-of-inference
            if not verify_proof_of_inference(
                response.seed,
                response.challenge,
                response.merkle_root,
                response.timesteps
            ):
                return 0.0, "Invalid proof of inference"
            
            # 2. Verify video authenticity
            if not verify_video_authenticity(
                response.video_data_b64,
                response.seed,
                response.timesteps
            ):
                return 0.0, "Video authenticity check failed"
            
            # 3. Compute multi-dimensional video quality score
            quality_score = compute_md_vqs(
                response.video_data_b64,
                synapse.text_prompt,
                synapse.width,
                synapse.height,
                synapse.num_frames,
                synapse.fps
            )
            
            # 4. Perform spot-checking
            
            spot_check_score = await self._perform_spot_check(response)
            if spot_check_score < 0.8:  # 80% threshold
                return 0.0, "Failed spot check verification"
            
            return quality_score, "Valid video generation"
            
        except Exception as e:
            logger.error(f"Validation error for miner {uid}: {str(e)}")
            return 0.0, f"Validation error: {str(e)}"
    
    
    async def _perform_spot_check(self, response: InferNet) -> float:
        """
        Performs a spot check on the video generation.
        
        Inputs:
        - response: Miner's response to check
            
        Returns Spot check score between 0 and 1
        """
        try:
            # 1. Verify timesteps match the video
            if not self._verify_timesteps(response):
                return 0.0
            
            # 2. Check for temporal consistency
            if not self._check_temporal_consistency(response):
                return 0.0
            
            # 3. Verify prompt adherence
            if not self._verify_prompt_adherence(response):
                return 0.0
            
            return 1.0
            
        except Exception as e:
            logger.error(f"Spot check error: {str(e)}")
            return 0.0
    

    # Optional spot-check functions for future use
    def _verify_timesteps(self, response: InferNet) -> bool:
        """Verifies that the timesteps match the video frames."""
        
        return True
    
    def _check_temporal_consistency(self, response: InferNet) -> bool:
        """Checks for temporal consistency in the video."""
        
        return True
    
    def _verify_prompt_adherence(self, response: InferNet) -> bool:
        """Verifies that the video adheres to the prompt."""
        
        return True
    
    
    
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
            # --- Pop a single pending request from active_requests ---
            pending_requests = [rid for rid, info in self.active_requests.items() if info.get('status') == 'pending']
            if not pending_requests:
                logger.info("No pending on-chain requests to process.")
                await asyncio.sleep(self.config.validator.poll_interval)
                return
            # Use the first pending request
            real_request_id = pending_requests[0]
            self.current_request_id = real_request_id
            self.active_requests[real_request_id]['status'] = 'processing'
            logger.info(f"Processing on-chain requestId: {real_request_id}")
            # --- Initialise per-miner proof map for this request ---
            self._pending_proofs[real_request_id] = {}
            # --- Use the verified prompt for this request ---
            prompt = self.active_requests[real_request_id].get('prompt')
            if not prompt:
                logger.info(f"Prompt not yet delivered for request {real_request_id}, queuing and retrying later.")
                self.active_requests[real_request_id]['status'] = 'pending'
                self._pending_prompt_queue.add(real_request_id)
                await asyncio.sleep(self.config.validator.poll_interval)
                return

            # 1. Generate per-request randomness
            C = os.urandom(self.config.validator.challenge_bytes)
            seed_bytes = hmac.new(self.wallet.ss58_address.encode(), C, hashlib.sha256).digest()
            seed = int.from_bytes(seed_bytes[:8], 'big')
            logger.debug(f"[{request_id}] Generated challenge and seed: {seed}")

            # 2. Choose spot-check timesteps
            total_steps = self.config.diffusion.num_steps
            k = self.config.validator.num_checkpoints
            # Do NOT pick k timesteps here or send them to the miner
            # Instead, miner will return all leaves, and we will sample k after receiving the response
            # 3. Build Merkle commitment

            # 4. Select miners
            uids = get_random_uids(self, k=self.config.neuron.sample_size)
            if not uids:
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
                challenge=C,
                # Do NOT send timesteps
            )
            logger.debug(f"[{request_id}] Dispatching requests to {len(axons)} miners")
            start_time = time.time()
            
            responses = await self.dendrite(
                axons=axons,
                synapse=syn,
                deserialize=True,
                timeout=self.config.validator.timeout
            )

            # 6. Process responses
            rewards = []
            result_details = []  # Collect per-miner result details
            video_paths = []     # Collect video paths for all miners
            for resp, uid in zip(responses, uids):
                try:
                    logger.debug(f"[{request_id}] Processing response from miner {uid}")
                    
                    # Unpack response using attribute access
                    V_bytes_b64 = resp.video_data_b64
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
                    import base64
                    V_bytes = base64.b64decode(V_bytes_b64)

                    proof = resp.proof  # This is still a dict inside the synapse
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

                    # Basic proof checks
                    if C_ret != C or seed_ret != seed:
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

                    # Signature verification
                    if not verify_proof_signature(uid, C_ret, seed_ret, V_bytes, merkle_root, signature):
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
                    logger.debug(f"[{request_id}] Requesting k random leaves for spot-check from miner {uid}")
                    self.metrics['spot_checks_performed'] += 1
                    # Get the full set of timesteps from the response (should be T long)
                    all_timesteps = resp.timesteps if hasattr(resp, 'timesteps') else None
                    if not all_timesteps or len(all_timesteps) < k:
                        logger.warning(f"[{request_id}] Not enough timesteps in response for spot-checking")
                        rewards.append(0.0)
                        result_details.append({
                            'uid': uid,
                            'status': 'not_enough_timesteps',
                            'score': 0.0,
                            'video_path': None,
                            'error': 'Not enough timesteps for spot-checking',
                        })
                        continue
                    spot_check_indices = random.sample(all_timesteps, k)
                    # Call miner's open_leaves RPC to get only the requested leaves
                    axon = self.metagraph.axons[uid]
                    # Defensive: ensure open_leaves is available
                    try:
                        # Pass request_id and caller_hotkey to tie the reveal to the specific request
                        caller_hotkey = self.wallet.hotkey.ss58_address if hasattr(self.wallet, 'hotkey') else 'unknown'
                        leaves_result = await self.dendrite.call(
                            axon=axon,
                            method='open_leaves',
                            args=[spot_check_indices],
                            kwargs={'request_id': request_id, 'caller_hotkey': caller_hotkey},
                            timeout=self.config.validator.timeout
                        )
                    except Exception as e:
                        logger.warning(f"[{request_id}] Failed to call open_leaves on miner {uid}: {e}")
                        rewards.append(0.0)
                        result_details.append({
                            'uid': uid,
                            'status': 'open_leaves_failed',
                            'score': 0.0,
                            'video_path': None,
                            'error': f'open_leaves failed: {e}',
                        })
                        continue
                    if not leaves_result or not isinstance(leaves_result, dict):
                        logger.warning(f"[{request_id}] open_leaves returned no data for miner {uid}")
                        rewards.append(0.0)
                        result_details.append({
                            'uid': uid,
                            'status': 'open_leaves_no_data',
                            'score': 0.0,
                            'video_path': None,
                            'error': 'open_leaves returned no data',
                        })
                        continue
                    ok = True
                    for t in spot_check_indices:
                        leaf = leaves_result.get(t)
                        if not leaf or len(leaf) != 3:
                            logger.warning(f"[{request_id}] Missing or malformed leaf for t={t} from miner {uid}")
                            ok = False
                            break
                        z_b, eps_b, proof_path = leaf
                        # Verify Merkle path
                        leaf_hash = hashlib.sha256(t.to_bytes(2,'big') + z_b + eps_b).digest()
                        if not verify_merkle_leaf(leaf_hash, proof_path, merkle_root):
                            logger.warning(f"[{request_id}] Merkle verification failed for miner {uid} at timestep {t}")
                            ok = False
                            break
                        # Run UNet step
                        if not run_unet_step(z_b, eps_b, self.config, t):
                            logger.warning(f"[{request_id}] UNet step verification failed for miner {uid} at timestep {t}")
                            ok = False
                            break
                    if not ok:
                        self.metrics['spot_checks_failed'] += 1
                        logger.warning(f"[{request_id}] Spot-check failed for miner {uid}")
                        rewards.append(0.0)
                        result_details.append({
                            'uid': uid,
                            'status': 'spot_check_failed',
                            'score': 0.0,
                            'video_path': None,
                            'error': 'Spot-check failed',
                        })
                        continue

                    # Save video for analysis
                    video_path = OUTPUT_DIR / f"video_{uid}_{int(time.time())}.mp4"
                    with open(video_path, 'wb') as f:
                        f.write(V_bytes)
                    logger.debug(f"[{request_id}] Saved video from miner {uid} to {video_path}")
                    video_paths.append(str(video_path))

                    # Verify video authenticity
                    if not verify_video_authenticity(str(video_path)):
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

                    # Compute MD-VQS
                    logger.debug(f"[{request_id}] Computing MD-VQS for miner {uid}")
                    pf, vq, tc = compute_md_vqs(str(video_path), self.config.validator.default_prompt)
                    score = (self.config.validator.alpha * pf +
                            self.config.validator.beta  * vq +
                            self.config.validator.gamma * tc)
                    logger.info(f"[{request_id}] Miner {uid} MD-VQS: {score:.3f} (pf={pf:.3f}, vq={vq:.3f}, tc={tc:.3f})")
                    rewards.append(score)
                    result_details.append({
                        'uid': uid,
                        'status': 'success',
                        'score': score,
                        'video_path': str(video_path),
                        'pf': pf,
                        'vq': vq,
                        'tc': tc,
                        'error': None,
                    })
                    # Update miner metrics
                    if uid not in self.metrics['miner_scores']:
                        self.metrics['miner_scores'][uid] = []
                    self.metrics['miner_scores'][uid].append({
                        'timestamp': datetime.now().isoformat(),
                        'score': score,
                        'pf': pf,
                        'vq': vq,
                        'tc': tc
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
                # Store result for retrieval via API
                if real_request_id in self.active_requests:
                    self.active_requests[real_request_id]['result'] = {
                        'miners': result_details,
                        'uids': uids,
                        'scores': rewards,
                        'video_paths': video_paths,
                        'timestamp': datetime.now().isoformat(),
                        'status': self.active_requests[real_request_id].get('status'),
                        'prompt': prompt,
                    }
                
            # Log metrics
            self._log_metrics()

        except Exception as e:
            self.metrics['failed_validations'] += 1
            self.metrics['errors'].append({
                'timestamp': datetime.now().isoformat(),
                'error': str(e),
                'type': 'validation_cycle_error',
                'request_id': request_id
            })
            logger.error(f"[{request_id}] Error in validation cycle: {str(e)}", exc_info=True)

        # 8. Throttle
        await asyncio.sleep(self.config.validator.poll_interval)

    def trigger_refund(self, request_id):
        """
        Triggers a refundUnused transaction for the given request_id.
        Returns the transaction hash.
        """
        self.logger.info(f"Attempting to trigger refund for request {request_id}")
        try:
            tx_data = contract.functions.refundUnused(int(request_id))
            tx_dict = {
                'from': VALIDATOR_ETH_ADDRESS,
                'nonce': w3.eth.get_transaction_count(VALIDATOR_ETH_ADDRESS),
            }
            try:
                tx_dict['gas'] = tx_data.estimate_gas(tx_dict)
            except Exception as e:
                self.logger.warning(f"Gas estimation failed for refundUnused, using fallback: {e}")
                tx_dict['gas'] = 100000
            try:
                tx_dict['gasPrice'] = w3.eth.gas_price
            except Exception as e:
                self.logger.warning(f"Failed to fetch gas price, using fallback: {e}")
                tx_dict['gasPrice'] = w3.to_wei('5', 'gwei')
            tx = tx_data.build_transaction(tx_dict)
            signed_tx = w3.eth.account.sign_transaction(tx, private_key=VALIDATOR_PRIVATE_KEY)
            tx_hash = w3.eth.send_raw_transaction(signed_tx.rawTransaction)
            self.logger.info(f"refundUnused tx: {tx_hash.hex()}")
            return tx_hash.hex()
        except Exception as e:
            self.logger.error(f"Failed to call refundUnused for request {request_id}: {e}")
            raise

    def receive_user_prompt(self, request_id, user_prompt):
        """
        Called when the user delivers their prompt off-chain (e.g., via REST).
        Verifies the prompt hash matches the on-chain commitment, then stores it for use.
        """
        if request_id not in self.active_requests:
            raise Exception(f"Unknown request_id {request_id}")
        prompt_hash_onchain = self.active_requests[request_id]['promptHash']
        prompt_hash_local = keccak(text=user_prompt)
        if prompt_hash_local != prompt_hash_onchain:
            raise Exception("Prompt mismatch: user did not commit to this prompt")
        self.active_requests[request_id]['prompt'] = user_prompt
        logger.info(f"Received and verified prompt for request {request_id}")

    def update_scores(self, rewards: List[float], uids: List[int]):
        """
        Update miner scores with validation results.
        Also record submissions and distribute rewards on-chain.
        Only processes the current request being validated.
        Args:
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
                # Get current scores
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
                    tx_hash = w3.eth.send_raw_transaction(signed_tx.rawTransaction)
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
                tx_hash = w3.eth.send_raw_transaction(signed_tx.rawTransaction)
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
            # Verify proof of inference
            if not verify_proof_of_inference(
                miner_hotkey,
                video_path,
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

# Entry point
if __name__ == '__main__':
    cfg = ValidatorNeuron.config()  
    neuron = ValidatorNeuron(config=cfg)
    neuron.run()
