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
    miner_public_key: bytes,
    challenge: bytes,
    seed: int,
    video_bytes: bytes,
    merkle_root: bytes,
    signature: bytes,
) -> bool:
    """Checks if the miner's signature is valid for the given proof data."""
    try:
        bt.logging.debug(f"=== SIGNATURE VERIFICATION DEBUG START ===")
        bt.logging.debug(f"Input parameters:")
        bt.logging.debug(f"  miner_hotkey_ss58: {miner_hotkey_ss58}")
        bt.logging.debug(f"  miner_public_key: {miner_public_key.hex() if miner_public_key else 'None'}")
        bt.logging.debug(f"  challenge: {challenge.hex()}")
        bt.logging.debug(f"  seed: {seed}")
        bt.logging.debug(f"  video_bytes length: {len(video_bytes)}")
        bt.logging.debug(f"  merkle_root: {merkle_root.hex()}")
        bt.logging.debug(f"  signature: {signature.hex()}")
        bt.logging.debug(f"  signature length: {len(signature)}")
        
        if miner_public_key is None:
            bt.logging.error("No miner public key available for verification")
            return False
        
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
        try:
            # Convert public key bytes to hex string
            public_key_hex = miner_public_key.hex()
            bt.logging.debug(f"Public key hex: {public_key_hex}")
            # Use crypto_type=1 for sr25519 to match the miner
            kp: bt.Keypair = bt.Keypair(public_key=public_key_hex, crypto_type=1)
            bt.logging.debug(f"Created keypair successfully")
            bt.logging.debug(f"  Keypair public key: {kp.public_key.hex()}")
            bt.logging.debug(f"  Keypair crypto type: {kp.crypto_type}")
            bt.logging.debug(f"  Keypair ss58 address: {kp.ss58_address}")
        except Exception as keypair_error:
            bt.logging.error(f"Error creating keypair with sr25519: {keypair_error}")
            bt.logging.debug(f"Trying fallback without crypto type...")
            try:
                # Convert public key bytes to hex string
                public_key_hex = miner_public_key.hex()
                kp: bt.Keypair = bt.Keypair(public_key=public_key_hex)
                bt.logging.debug(f"Created keypair (fallback) successfully")
                bt.logging.debug(f"  Keypair public key: {kp.public_key.hex()}")
                bt.logging.debug(f"  Keypair crypto type: {kp.crypto_type}")
                bt.logging.debug(f"  Keypair ss58 address: {kp.ss58_address}")
            except Exception as fallback_error:
                bt.logging.error(f"Error creating keypair (fallback): {fallback_error}")
                return False
        
        # Verify signature
        bt.logging.debug(f"Calling kp.verify(message, signature)...")
        bt.logging.debug(f"  message: {message.hex()}")
        bt.logging.debug(f"  signature: {signature.hex()}")
        
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
            None,
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
        for step_i, (t, (z_b, eps_b, path)) in enumerate(decoded_leaf_data.items()):
            # Verify Merkle path
            leaf_hash = hashlib.sha256(
                t.to_bytes(2, "big") + z_b + eps_b
            ).digest()

            if not verify_merkle_leaf(leaf_hash, path, merkle_root):
                bt.logging.warning(f"Merkle path failed at timestep {t}")
                return False

            # Check UNet step using step index, not raw timestep value
            if not run_unet_step(z_b, eps_b, config_for_unet, step_i):
                bt.logging.warning(f"UNet step check failed at timestep {t} (step {step_i})")
                return False

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
        
        # Get alpha for this step
        alpha_t = config['alphas'][step_i]
        
        # Perform the denoising step verification
        # Convert alpha_t to tensor for torch operations
        alpha_tensor = torch.tensor(alpha_t, dtype=torch.float16)
        z_pred = (z - (1 - alpha_tensor) * eps) / torch.sqrt(alpha_tensor)
        
        # Check if the result is valid
        is_finite = torch.isfinite(z_pred).all()
        is_bounded = torch.abs(z_pred).max() < 10.0
        
        bt.logging.debug(f"UNet step {step_i} verification: finite={is_finite}, bounded={is_bounded}")
        
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
    """Picks random leaves to reveal for spot-checking."""
    rng = random.Random(random_seed)
    indices = list(range(num_leaves))
    rng.shuffle(indices)
    return indices[:num_to_reveal]

