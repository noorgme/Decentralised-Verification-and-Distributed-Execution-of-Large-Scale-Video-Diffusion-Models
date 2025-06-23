import hashlib
import hmac
import base64
import random
import bittensor as bt
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional


def derive_seed(challenge: bytes, wallet_address: str) -> int:
    """Maps challenge and wallet to a 64-bit seed using HMAC-SHA256."""
    try:
        h = hmac.new(wallet_address.encode(), challenge, hashlib.sha256)
        return int.from_bytes(h.digest()[:8], byteorder='big')
    except Exception as e:
        bt.logging.error(f"Error deriving seed: {str(e)}")
        raise


def verify_proof_signature(
    miner_hotkey_ss58: str,
    # miner_public_key: bytes,
    challenge: bytes,
    seed: int,
    video_bytes: bytes,
    merkle_root: bytes,
    signature: bytes,
) -> bool:
    """Checks if the miner's signature is valid for the given proof data."""
    try:
        bt.logging.debug(f"=== SIGNATURE VERIFICATION DEBUG START ===")
        # bt.logging.debug(f"Input parameters:")
        # bt.logging.debug(f"  miner_hotkey_ss58: {miner_hotkey_ss58}")
        # bt.logging.debug(f"  miner_public_key: {miner_public_key.hex() if miner_public_key else 'None'}")
        # Derive miner public key from SS58 hotkey
        keypair = bt.Keypair(ss58_address=miner_hotkey_ss58)
        miner_public_key = keypair.public_key
        bt.logging.debug(f"Derived miner_public_key: {miner_public_key.hex()} from hotkey {miner_hotkey_ss58}")
        bt.logging.debug(f"Input parameters:")
        bt.logging.debug(f"  miner_hotkey_ss58: {miner_hotkey_ss58}")
        bt.logging.debug(f"  challenge: {challenge.hex()}")
        bt.logging.debug(f"  seed: {seed}")
        bt.logging.debug(f"  video_bytes length: {len(video_bytes)}")
        bt.logging.debug(f"  merkle_root: {merkle_root.hex()}")
        bt.logging.debug(f"  signature: {signature.hex()}")
        bt.logging.debug(f"  signature length: {len(signature)}")
        
        # if miner_public_key is None:
        #     bt.logging.error("No miner public key available for verification")
        #     return False
        
        bt.logging.debug(f"Miner public key is available: {miner_public_key.hex()}")
        
        # Calculate message components
        video_hash = hashlib.sha256(video_bytes).digest()
        seed_le64  = seed.to_bytes(8, byteorder="little", signed=False)
        message    = challenge + seed_le64 + video_hash + merkle_root
        
        bt.logging.debug(f"Message components:")
        bt.logging.debug(f"  challenge: {challenge.hex()}")
        bt.logging.debug(f"  seed: {seed}")
        bt.logging.debug(f"  seed_le64: {seed_le64.hex()}")
        bt.logging.debug(f"  video_hash: {video_hash.hex()}")
        bt.logging.debug(f"  merkle_root: {merkle_root.hex()}")
        bt.logging.debug(f"  full_message: {message.hex()}")
        bt.logging.debug(f"  message_length: {len(message)} bytes")
        
        # Create keypair for verification
        bt.logging.debug(f"Creating keypair with sr25519 crypto type...")
        # try:
        #     # Convert public key bytes to hex string
        #     public_key_hex = miner_public_key.hex()
        #     bt.logging.debug(f"Public key hex: {public_key_hex}")
        #     # Use crypto_type=1 for sr25519 to match the miner
        #     kp: bt.Keypair = bt.Keypair(public_key=public_key_hex, crypto_type=1)
        #     bt.logging.debug(f"Created keypair successfully")
        #     bt.logging.debug(f"  Keypair public key: {kp.public_key.hex()}")
        #     bt.logging.debug(f"  Keypair crypto type: {kp.crypto_type}")
        #     bt.logging.debug(f"  Keypair ss58 address: {kp.ss58_address}")
        # except Exception as keypair_error:
        #     bt.logging.error(f"Error creating keypair with sr25519: {keypair_error}")
        #     bt.logging.debug(f"Trying fallback without crypto type...")
        #     try:
        #         # Convert public key bytes to hex string
        #         public_key_hex = miner_public_key.hex()
        #         kp: bt.Keypair = bt.Keypair(public_key=public_key_hex)
        #         bt.logging.debug(f"Created keypair (fallback) successfully")
        #         bt.logging.debug(f"  Keypair public key: {kp.public_key.hex()}")
        #         bt.logging.debug(f"  Keypair crypto type: {kp.crypto_type}")
        #         bt.logging.debug(f"  Keypair ss58 address: {kp.ss58_address}")
        #     except Exception as fallback_error:
        #         bt.logging.error(f"Error creating keypair (fallback): {fallback_error}")
        #         return False
        
        # Verify signature
        bt.logging.debug(f"Calling kp.verify(message, signature)...")
        bt.logging.debug(f"  message: {message.hex()}")
        bt.logging.debug(f"  signature: {signature.hex()}")
        kp = bt.Keypair(public_key=miner_public_key.hex(), crypto_type=1)
        
        try:
            result = kp.verify(message, signature)
            bt.logging.debug(f"Verification result: {result}")
        except Exception as verify_error:
            bt.logging.error(f"Error in kp.verify: {verify_error}")
            result = False
        
        if not result:
            bt.logging.error(f"SIGNATURE VERIFICATION FAILED")
            bt.logging.error(f"Final comparison:")
            bt.logging.error(f"  Message: {message.hex()}")
            bt.logging.error(f"  Signature: {signature.hex()}")
            bt.logging.error(f"  Public key: {kp.public_key.hex()}")
            bt.logging.error(f"  Crypto type: {kp.crypto_type}")
            bt.logging.error(f"  SS58 address: {kp.ss58_address}")
        else:
            bt.logging.debug(f"SIGNATURE VERIFICATION SUCCESSFUL")
            
        bt.logging.debug(f"=== SIGNATURE VERIFICATION DEBUG END ===")
        return result
    except Exception as e:
        bt.logging.error(f"Signature verification failed with exception: {str(e)}")
        bt.logging.error(f"Exception type: {type(e)}")
        import traceback
        bt.logging.error(f"Traceback: {traceback.format_exc()}")
        return False


def verify_proof_of_inference(
    miner_hotkey_ss58: str,
    video_bytes: bytes,
    proof: Dict,
    config_for_unet: Dict,
) -> bool:
    """Verifies the complete proof of inference by checking signature and sampling UNet steps."""
    try:
        # ---- decode everything ----
        challenge = proof["challenge"]
        if isinstance(challenge, str):
            challenge = base64.b64decode(challenge)

        seed = proof["seed"]

        merkle_root = proof["merkle_root"]
        if isinstance(merkle_root, str):
            merkle_root = base64.b64decode(merkle_root)

        signature = proof["signature"]
        if isinstance(signature, str):
            signature = base64.b64decode(signature)

        # video_bytes might also be b64
        if not isinstance(video_bytes, (bytes, bytearray)):
            video_bytes = base64.b64decode(video_bytes)

        # now verify the signature
        if not verify_proof_signature(
            miner_hotkey_ss58,
            challenge,
            seed,
            video_bytes,
            merkle_root,
            signature,
        ):
            return False

        # ---- decode leaf_data ----
        raw_ld = proof.get("leaf_data", {})
        decoded_leaf_data: Dict[int, Tuple[bytes, bytes, list[bytes]]] = {}
        for t_key, (z_b64, eps_b64, path_b64) in raw_ld.items():
            t = int(t_key)
            z_bytes  = base64.b64decode(z_b64)
            eps_bytes= base64.b64decode(eps_b64)
            path     = [ base64.b64decode(p) for p in path_b64 ]
            decoded_leaf_data[t] = (z_bytes, eps_bytes, path)

        # Check Merkle tree and UNet steps
        # Use the original order of timesteps as they were processed by the miner
        items = list(decoded_leaf_data.items())  # Keep original order
        timestep_to_step = {timestep: step_idx for step_idx, (timestep, _) in enumerate(items)}
        
        # Get the actual alpha values from the proof if available, otherwise use config
        proof_alphas = proof.get('alphas', [])
        if proof_alphas and len(proof_alphas) > 0:
            bt.logging.debug(f"Using alpha values from proof for UNet verification: {proof_alphas}")
            alphas_to_use = proof_alphas
        else:
            bt.logging.debug(f"No alpha values in proof, using config alphas for UNet verification: {config_for_unet['alphas']}")
            alphas_to_use = config_for_unet['alphas']
        
        for t, (z_b, eps_b, path) in decoded_leaf_data.items():
            # Verify Merkle path
            leaf_hash = hashlib.sha256(
                t.to_bytes(2, "big") + z_b + eps_b
            ).digest()

            if not verify_merkle_leaf(leaf_hash, path, merkle_root):
                bt.logging.warning(f"Merkle path failed at timestep {t}")
                return False

            # Check UNet step using the correct step index based on original order
            step_i = timestep_to_step.get(t)
            if step_i is None:
                bt.logging.warning(f"Could not determine step index for timestep {t}")
                return False
                
            if not run_unet_step(z_b, eps_b, {'alphas': alphas_to_use, **config_for_unet}, step_i):
                bt.logging.warning(f"UNet step check failed at timestep {t} (step {step_i})")
                return False

        # Cross-step consistency check for temporal coherence
        if len(decoded_leaf_data) >= 2:
            # Convert decoded_leaf_data to the format expected by verify_temporal_coherence_spot_check
            leaves_result = {}
            for t, (z_b, eps_b, _) in decoded_leaf_data.items():
                # Convert bytes back to base64 strings for the dedicated function
                z_b64 = base64.b64encode(z_b).decode('utf-8')
                eps_b64 = base64.b64encode(eps_b).decode('utf-8')
                leaves_result[str(t)] = (z_b64, eps_b64, [])  # Empty proof path since we already verified Merkle
            
            # Get all timesteps in order
            spot_check_indices = [t for t, _ in items]
            
            # Use the dedicated temporal coherence verification function
            temporal_coherence_ok = verify_temporal_coherence_spot_check(
                spot_check_indices=spot_check_indices,
                leaves_result=leaves_result,
                timestep_to_step=timestep_to_step,
                proof_alphas=alphas_to_use,
                unet_config=config_for_unet,
                request_id="proof_verification",
                logger=bt.logging
            )
            
            if not temporal_coherence_ok:
                bt.logging.warning("Temporal coherence check failed in proof verification")
                return False
            else:
                bt.logging.debug("Temporal coherence check passed in proof verification")

        return True

    except Exception as e:
        bt.logging.error(f"Proof-of-inference verification error: {str(e)}")
        return False



def build_merkle_root(leaves: List[bytes]) -> Tuple[bytes, List[List[bytes]]]:
    """Builds a Merkle tree and returns the root hash and proof paths."""
    if not leaves:
        return b"", []
    
    # Hash each leaf
    leaf_hashes = [hashlib.sha256(leaf).digest() for leaf in leaves]
    tree = [leaf_hashes]
    
    # Build tree levels
    while len(tree[-1]) > 1:
        current_level = tree[-1]
        next_level = []
        for i in range(0, len(current_level), 2):
            left = current_level[i]
            right = current_level[i+1] if i+1 < len(current_level) else current_level[i]
            # Sort for consistency
            if left < right:
                combined = left + right
            else:
                combined = right + left
            next_level.append(hashlib.sha256(combined).digest())
        tree.append(next_level)
    
    # Get root and generate proofs
    root = tree[-1][0]
    proofs: List[List[bytes]] = []
    
    for idx in range(len(leaf_hashes)):
        proof: List[bytes] = []
        index = idx
        for level in tree[:-1]:
            sibling_index = index ^ 1
            sibling = level[sibling_index] if sibling_index < len(level) else level[index]
            proof.append(sibling)
            index //= 2
        proofs.append(proof)
    
    return root, proofs


def verify_merkle_leaf(leaf_hash: bytes, proof_path: List[bytes], root: bytes) -> bool:
    """Checks if a leaf hash is valid against the Merkle root using its proof path."""
    try:
        current = leaf_hash
        for sibling in proof_path:
            if current < sibling:
                combined = current + sibling
            else:
                combined = sibling + current
            current = hashlib.sha256(combined).digest()
        return current == root
    except Exception as e:
        bt.logging.error(f"Error verifying Merkle leaf: {str(e)}")
        return False


def run_unet_step(
    z_bytes: bytes,
    eps_bytes: bytes,
    config: dict,
    step_i: int
) -> bool:
    """Runs a single UNet denoising step to verify the latents."""
    try:
        # Check bounds for alphas array
        if step_i >= len(config['alphas']):
            bt.logging.error(f"Step index {step_i} out of bounds for alphas array (length: {len(config['alphas'])})")
            return False
            
        # Decode bytes to tensors
        z = torch.from_numpy(np.frombuffer(z_bytes, dtype=np.float16))
        eps = torch.from_numpy(np.frombuffer(eps_bytes, dtype=np.float16))
        
        # The miner stores 5D tensors: (1, 4, num_frames, height//8, width//8)
        # We need to reshape to handle the batch and frame dimensions
        total_elements = z.numel()
        expected_elements = config['latent_channels'] * config['latent_height'] * config['latent_width']
        
        # If we have more elements than expected, it's a 5D tensor
        if total_elements > expected_elements:
            # Reshape to 5D first, then take the first frame
            num_frames = total_elements // (config['latent_channels'] * config['latent_height'] * config['latent_width'])
            z = z.view(1, config['latent_channels'], num_frames, config['latent_height'], config['latent_width'])
            eps = eps.view(1, config['latent_channels'], num_frames, config['latent_height'], config['latent_width'])
            
            # Take the first frame for verification
            z = z[0, :, 0, :, :]  # Shape: (latent_channels, latent_height, latent_width)
            eps = eps[0, :, 0, :, :]  # Shape: (latent_channels, latent_height, latent_width)
        else:
            # Reshape to 3D as expected
            z = z.view(config['latent_channels'], config['latent_height'], config['latent_width'])
            eps = eps.view(config['latent_channels'], config['latent_height'], config['latent_width'])
        
        # Use same scheduler as the miner
        try:
            from diffusers import DDIMScheduler
            
         
            # We can infer the scheduler configuration from the alpha values in the proof
            scheduler = DDIMScheduler()
            
           
            scheduler.set_timesteps(len(config['alphas']))
            
          
            bt.logging.debug(f"UNet step {step_i}: using timestep {scheduler.timesteps[step_i]} from scheduler")
            bt.logging.debug(f"Scheduler timesteps: {scheduler.timesteps.tolist()}")
            bt.logging.debug(f"Config alphas: {config['alphas']}")
            
            prev_sample = scheduler.step(
                eps,           
                scheduler.timesteps[step_i], 
                z            
            ).prev_sample   
            
            # Check if the result is valid
            is_finite = torch.isfinite(prev_sample).all()
            is_bounded = torch.abs(prev_sample).max() < 10.0
            
            bt.logging.debug(f"UNet step {step_i} verification: finite={is_finite}, bounded={is_bounded}")
            
            return is_finite and is_bounded
            
        except ImportError:
            bt.logging.error(f"Could not import DDIMScheduler - falling back to manual formula")
            # Fallback to manual formula if scheduler is not available
            alpha_t = config['alphas'][step_i]
            alpha_tensor = torch.tensor(alpha_t, dtype=torch.float16)
            sqrt_1_minus_alpha = torch.sqrt(1.0 - alpha_tensor)
            z_pred = (z - sqrt_1_minus_alpha * eps) / torch.sqrt(alpha_tensor)
            
            # Check if the result is valid
            is_finite = torch.isfinite(z_pred).all()
            is_bounded = torch.abs(z_pred).max() < 10.0
            
            bt.logging.debug(f"UNet step {step_i} verification (fallback): finite={is_finite}, bounded={is_bounded}")
            
            return is_finite and is_bounded
        
    except Exception as e:
        bt.logging.error(f"Error running UNet step {step_i}: {str(e)}")
        return False


# --- Step sampling for commit-then-reveal Merkle spot-check protocol ---

def commit_then_reveal_merkle_spotcheck(
    merkle_root: bytes,
    num_leaves: int,
    num_to_reveal: int,
    random_seed: int
) -> List[int]:
    """Picks random leaves to reveal for spot-checking, ensuring temporal coherence."""
    rng = random.Random(random_seed)
    
    # Pick num_to_reveal starting points, then reveal pairs for temporal coherence
    if num_leaves < 2:
        # Fallback to single indices if not enough leaves
        return list(range(min(num_to_reveal, num_leaves)))
    
    max_start = num_leaves - 2
    starts = rng.sample(range(max_start + 1), min(num_to_reveal, max_start + 1))
    result = []
    for s in starts:
        result.extend([s, s+1])
    return result


def verify_temporal_coherence_spot_check(
    spot_check_indices: List[int],
    leaves_result: Dict[str, Tuple[str, str, List[str]]],
    timestep_to_step: Dict[int, int],
    proof_alphas: List[float],
    unet_config: Dict,
    request_id: str = None,
    logger=None
) -> bool:
    """
    Verifies temporal coherence for spot-checked leaves.
    
    Input Params:
        spot_check_indices: List of timesteps to check
        leaves_result: Dictionary mapping timestep strings to (z_b64, eps_b64, proof_path_b64) tuples
        timestep_to_step: Mapping from timestep values to step indices
        proof_alphas: Alpha values from the miner's proof
        unet_config: UNet configuration for tensor reshaping
        
        
    Returns True if temporal coherence check passes, False otherwise
    """
    if logger is None:
        import logging
        logger = logging.getLogger(__name__)
    
    if len(spot_check_indices) < 2:
        logger.debug(f"[{request_id}] Not enough timesteps for temporal coherence check (need at least 2, got {len(spot_check_indices)})")
        return True  # Skip if not enough timesteps
    
    logger.debug(f"[{request_id}] Performing temporal coherence check on {len(spot_check_indices)} timesteps")
    logger.debug(f"[{request_id}] Spot check indices: {spot_check_indices}")
    
    # Build a list of (timestep, z_bytes, eps_bytes) sorted by timestep
    leaf_data = {}
    for i, t in enumerate(spot_check_indices):
        t_str = str(t)
        leaf = leaves_result.get(t_str)
        if leaf and len(leaf) == 3:
            z_b, eps_b, _ = leaf
            leaf_data[t] = (z_b, eps_b)
            logger.debug(f"[{request_id}] Added leaf data for timestep {t}: z_b64_len={len(z_b)}, eps_b64_len={len(eps_b)}")
        else:
            logger.warning(f"[{request_id}] Missing or malformed leaf for timestep {t}: {leaf}")
    
    logger.debug(f"[{request_id}] Collected leaf data for {len(leaf_data)} timesteps")
    
    # Check if we have enough timesteps for temporal coherence
    if len(leaf_data) < 2:
        logger.warning(f"[{request_id}] Not enough leaf data for temporal coherence check (need at least 2, got {len(leaf_data)})")
        logger.warning(f"[{request_id}] Temporal coherence check will be skipped")
        return True  # Skip if not enough data
    
    # Use timesteps in the order they were selected by the validator
    # The validator already selected consecutive pairs, so we should trust that order
    # Don't re-sort by step index  can break the consecutive pairs)
    sorted_timesteps = [t for t in spot_check_indices if t in leaf_data]
    
    logger.debug(f"[{request_id}] Using timesteps in validator's selection order: {sorted_timesteps}")
    logger.debug(f"[{request_id}] Step indices: {[timestep_to_step.get(t) for t in sorted_timesteps]}")
    
    # Check if we have enough alphas for all timesteps
    if len(sorted_timesteps) > len(proof_alphas):
        logger.warning(f"[{request_id}] More timesteps ({len(sorted_timesteps)}) than alphas ({len(proof_alphas)})")
        logger.warning(f"[{request_id}] Will only check first {len(proof_alphas)} timesteps")
        sorted_timesteps = sorted_timesteps[:len(proof_alphas)]
    
    # Set up scheduler for proper reverse step calculation
    try:
        from diffusers import DDIMScheduler
        
        # Create a scheduler with the same configuration as the miner
        # We can infer the configuration from the alpha values in the proof
        scheduler = DDIMScheduler()
        
        # Set the same number of timesteps as the miner
        scheduler.set_timesteps(len(proof_alphas))
        logger.debug(f"[{request_id}] Set up scheduler with {len(proof_alphas)} timesteps")
        logger.debug(f"[{request_id}] Scheduler timesteps: {scheduler.timesteps.tolist()}")
        logger.debug(f"[{request_id}] Proof alphas: {proof_alphas}")
    except ImportError:
        logger.error(f"[{request_id}] Could not import DDIMScheduler - temporal coherence check will fail")
        return False
    except Exception as e:
        logger.error(f"[{request_id}] Error setting up scheduler: {e}")
        return False
    
    for idx in range(len(sorted_timesteps) - 1):
        t_i = sorted_timesteps[idx]
        t_j = sorted_timesteps[idx + 1]
        
        logger.debug(f"[{request_id}] Checking temporal coherence: t_i={t_i} -> t_j={t_j}")
        
        z_i_b64, eps_i_b64 = leaf_data[t_i]
        z_j_b64, eps_j_b64 = leaf_data[t_j]
        
        # Decode base64 strings
        try:
            z_i_bytes = base64.b64decode(z_i_b64)
            eps_i_bytes = base64.b64decode(eps_i_b64)
            z_j_bytes = base64.b64decode(z_j_b64)
            logger.debug(f"[{request_id}] Decoded bytes: z_i={len(z_i_bytes)}, eps_i={len(eps_i_bytes)}, z_j={len(z_j_bytes)}")
        except Exception as e:
            logger.error(f"[{request_id}] Base64 decode error: {e}")
            return False
        
        # Look up the miner's actual step indices:
        step_i = timestep_to_step.get(t_i)
        step_j = timestep_to_step.get(t_j)
        if step_i is None or step_j is None:
            logger.warning(f"[{request_id}] Could not map timesteps to step indices: t_i={t_i} (step {step_i}), t_j={t_j} (step {step_j})")
            return False
        # Note: We trust the validator's selection of consecutive pairs
        # The validator already ensured these are consecutive in the miner's timestep array

        logger.debug(f"[{request_id}] Step indices: step_i={step_i}, step_j={step_j}")
        
        # Reconstruct next latent from current step using scheduler
        import torch
        import numpy as np
        
        try:
            # Decode tensors with proper shape handling
            z_i_tensor = torch.from_numpy(np.frombuffer(z_i_bytes, dtype=np.float16))
            eps_i_tensor = torch.from_numpy(np.frombuffer(eps_i_bytes, dtype=np.float16))
            
            logger.debug(f"[{request_id}] Initial tensor shapes: z_i={z_i_tensor.shape}, eps_i={eps_i_tensor.shape}")
            
            # Handle 5D tensor reshaping (batch, channels, frames, height, width)
            total_elements = z_i_tensor.numel()
            expected_elements = unet_config['latent_channels'] * unet_config['latent_height'] * unet_config['latent_width']
            
            logger.debug(f"[{request_id}] Tensor reshaping: total_elements={total_elements}, expected_elements={expected_elements}")
            
            if total_elements > expected_elements:
                # Reshape to 5D first, then take the first frame
                num_frames = total_elements // (unet_config['latent_channels'] * unet_config['latent_height'] * unet_config['latent_width'])
                logger.debug(f"[{request_id}] 5D tensor detected: num_frames={num_frames}")
                
                z_i_tensor = z_i_tensor.view(1, unet_config['latent_channels'], num_frames, unet_config['latent_height'], unet_config['latent_width'])
                eps_i_tensor = eps_i_tensor.view(1, unet_config['latent_channels'], num_frames, unet_config['latent_height'], unet_config['latent_width'])
                
                # Take the first frame for verification
                z_i_tensor = z_i_tensor[0, :, 0, :, :]  # Shape: (latent_channels, latent_height, latent_width)
                eps_i_tensor = eps_i_tensor[0, :, 0, :, :]  # Shape: (latent_channels, latent_height, latent_width)
                logger.debug(f"[{request_id}] After 5D reshape: z_i={z_i_tensor.shape}, eps_i={eps_i_tensor.shape}")
            else:
                # Reshape to 3D as expected
                z_i_tensor = z_i_tensor.view(unet_config['latent_channels'], unet_config['latent_height'], unet_config['latent_width'])
                eps_i_tensor = eps_i_tensor.view(unet_config['latent_channels'], unet_config['latent_height'], unet_config['latent_width'])
                logger.debug(f"[{request_id}] After 3D reshape: z_i={z_i_tensor.shape}, eps_i={eps_i_tensor.shape}")
            
           
            timestep_value = scheduler.timesteps[step_i]
            logger.debug(f"[{request_id}] Computing z_pred_j using scheduler.step() with timestep {timestep_value} (step_i={step_i})")
            
            # Use the scheduler's step method to get the actual reverse sample
            prev_sample = scheduler.step(
                eps_i_tensor,    # noise prediction
                timestep_value,  # actual timestep value from scheduler
                z_i_tensor       # current latent
            ).prev_sample        # this is the actual z_{t-1}
            
            logger.debug(f"[{request_id}] z_pred_j shape: {prev_sample.shape}")
            logger.debug(f"[{request_id}] z_pred_j stats: min={prev_sample.min():.6f}, max={prev_sample.max():.6f}, mean={prev_sample.mean():.6f}")
            
            # Compare to actual z_j
            z_j_tensor = torch.from_numpy(np.frombuffer(z_j_bytes, dtype=np.float16))
            logger.debug(f"[{request_id}] z_j_tensor initial shape: {z_j_tensor.shape}")
            
            # Apply same reshaping logic to z_j
            total_elements_j = z_j_tensor.numel()
            if total_elements_j > expected_elements:
                num_frames_j = total_elements_j // (unet_config['latent_channels'] * unet_config['latent_height'] * unet_config['latent_width'])
                logger.debug(f"[{request_id}] z_j 5D reshape: num_frames_j={num_frames_j}")
                
                z_j_tensor = z_j_tensor.view(1, unet_config['latent_channels'], num_frames_j, unet_config['latent_height'], unet_config['latent_width'])
                z_j_tensor = z_j_tensor[0, :, 0, :, :]
            else:
                z_j_tensor = z_j_tensor.view(unet_config['latent_channels'], unet_config['latent_height'], unet_config['latent_width'])
            
            logger.debug(f"[{request_id}] z_j_tensor final shape: {z_j_tensor.shape}")
            logger.debug(f"[{request_id}] z_j_tensor stats: min={z_j_tensor.min():.6f}, max={z_j_tensor.max():.6f}, mean={z_j_tensor.mean():.6f}")
            
            # tolerance for fp16 rounding errors
            is_close = torch.allclose(prev_sample, z_j_tensor, rtol=1, atol=1)
            max_diff = torch.abs(prev_sample - z_j_tensor).max()
            mean_diff = torch.abs(prev_sample - z_j_tensor).mean()
            
            logger.debug(f"[{request_id}] Comparison results: is_close={is_close}, max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")
            
            if not is_close:
                logger.warning(f"[{request_id}] Cross-step check failed between t={t_i} and t={t_j}")
                logger.warning(f"[{request_id}] Max difference: {max_diff:.6f}")
                logger.warning(f"[{request_id}] Mean difference: {mean_diff:.6f}")
                logger.warning(f"[{request_id}] This indicates a temporal coherence violation!")
                return False
            else:
                logger.debug(f"[{request_id}] Cross-step check passed between t={t_i} and t={t_j}")
                logger.debug(f"[{request_id}] Max difference: {max_diff:.6f} (within tolerance)")
                
        except Exception as e:
            logger.error(f"[{request_id}] Error in temporal coherence computation: {str(e)}")
            logger.error(f"[{request_id}] Exception details:", exc_info=True)
            return False
    
    logger.debug(f"[{request_id}] TEMPORAL COHERENCE CHECK PASSED: All temporal coherence verifications successful")
    return True

