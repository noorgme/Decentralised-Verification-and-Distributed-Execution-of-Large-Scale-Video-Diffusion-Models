import hmac
import hashlib
import typing
import bittensor as bt
import torch
import numpy as np
from typing import List, Tuple, Optional
import random


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
    challenge: bytes,
    seed: int,
    video_bytes: bytes,
    merkle_root: bytes,
    signature: bytes,
) -> bool:
    """Checks if the miner's signature is valid for the given proof data."""
    try:
        kp: bt.Keypair = bt.Keypair(ss58_address=miner_hotkey_ss58)
        video_hash = hashlib.sha256(video_bytes).digest()
        seed_le64  = seed.to_bytes(8, byteorder="little", signed=False)
        message    = challenge + seed_le64 + video_hash + merkle_root
        return kp.verify(signature, message)
    except Exception as e:
        bt.logging.error(f"Signature verification failed: {str(e)}")
        return False


# template/validator/proof.py   (add below verify_proof_signature)

from typing import Dict, Tuple

def verify_proof_of_inference(
    miner_hotkey_ss58: str,
    video_bytes: bytes,
    proof: Dict,
    config_for_unet: Dict,
) -> bool:
    """Verifies the complete proof of inference by checking signature and sampling UNet steps."""
    try:
        # Check signature first
        if not verify_proof_signature(
            miner_hotkey_ss58,
            proof["challenge"],
            proof["seed"],
            video_bytes,
            proof["merkle_root"],
            proof["signature"],
        ):
            return False

        # Check Merkle tree and UNet steps
        merkle_root = proof["merkle_root"]
        leaf_data   = proof["leaf_data"]

        for t, (z_b, eps_b, path) in leaf_data.items():
            # Verify Merkle path
            leaf_hash = hashlib.sha256(
                t.to_bytes(2, "big") + z_b + eps_b
            ).digest()

            if not verify_merkle_leaf(leaf_hash, path, merkle_root):
                bt.logging.warning(f"Merkle path failed at timestep {t}")
                return False

            # Check UNet step
            if not run_unet_step(z_b, eps_b, config_for_unet, int(t)):
                bt.logging.warning(f"UNet step check failed at timestep {t}")
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
    t: int
) -> bool:
    """Runs a single UNet denoising step to verify the latents."""
    try:
        z = torch.from_numpy(np.frombuffer(z_bytes, dtype=np.float32))
        eps = torch.from_numpy(np.frombuffer(eps_bytes, dtype=np.float32))
        z = z.view(config['latent_channels'], config['latent_height'], config['latent_width'])
        eps = eps.view(config['latent_channels'], config['latent_height'], config['latent_width'])
        alpha_t = config['alphas'][t]
        z_pred = (z - (1 - alpha_t) * eps) / torch.sqrt(alpha_t)
        return torch.isfinite(z_pred).all() and torch.abs(z_pred).max() < 10.0
    except Exception as e:
        bt.logging.error(f"Error running UNet step: {str(e)}")
        return False


# --- Commit-then-reveal Merkle spot-check protocol ---

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

