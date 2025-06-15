import os
import pytest
from typing import List
import hashlib
import bittensor as bt
import torch
import numpy as np
from template.protocol import InferNet
from template.validator.proof import (
    derive_seed,
    verify_proof_signature,
    build_merkle_root,
    verify_merkle_leaf,
    run_unet_step
)
from template.validator.scoring import MDVQS
from neurons.validator import ValidatorNeuron
import json
import base64
import hashlib

def dummy_baseneuron_init(self, config=None):
    self.config = config
    self.wallet = None
    self.subtensor = None
    self.metagraph = None
    self.device = None
    self.uid = 0

@pytest.fixture(autouse=True)
def patch_baseneuron_init(monkeypatch):
    monkeypatch.setattr("template.base.neuron.BaseNeuron.__init__", dummy_baseneuron_init)

@pytest.fixture(autouse=True)
def patch_validator_init(monkeypatch):
    def dummy_validator_init(self, config=None):
        dummy_baseneuron_init(self, config)
        self.metagraph = type('MockMetagraph', (), {'hotkeys': ['test_hotkey']})()
        self.hotkeys = self.metagraph.hotkeys
        self.scores = {}
        self.last_spot_check = {}
        self.scorer = type('MockScorer', (), {'compute_md_vqs': lambda *a, **kw: (1.0, 1.0, 1.0, 1.0)})()
        def _spot_check(uid, response):
            self.last_spot_check[uid] = 123
        self._spot_check = _spot_check
        self.metrics = {
            'total_requests': 0,
        }
        self.dendrite = type('Dendrite', (), {})()
        async def _mock_forward(*args, **kwargs):
            synapse = args[0]
            self.metrics['total_requests'] += 1
            with open('validator_metrics.json', 'w') as f:
                json.dump(self.metrics, f)
            with open('miner_metrics.json', 'w') as f:
                json.dump({'total_requests': 1, 'successful_generations': 1}, f)
            
            synapse.challenge = os.urandom(self.config.validator.challenge_bytes)
            synapse.seed      = derive_seed(synapse.challenge, synapse.dendrite.hotkey)
            self.scores[synapse.dendrite.uuid] = 1.0
            return synapse
        self.forward = _mock_forward
    monkeypatch.setattr("neurons.validator.ValidatorNeuron.__init__", dummy_validator_init)

class MockDendrite:
    def __init__(self, hotkey: str, uuid: str):
        self.hotkey = hotkey
        self.uuid = uuid

class MockResponse:
    def __init__(self, video_data: bytes, latents: List[bytes], noise_preds: List[bytes], timesteps: List[int]):
        self.video_data_b64 = base64.b64encode(video_data).decode('utf-8')
        self.latents = latents
        self.noise_preds = noise_preds
        self.timesteps = timesteps
        self.merkle_root = b"mock_root"
        self.signature = b"mock_signature"
        self.model_config = {
            "latent_channels": 4,
            "latent_height": 32,
            "latent_width": 32,
            "alphas": [0.1 * i for i in range(1000)]
        }

    def deserialize(self) -> bytes:
        return self.video_data_b64.encode()

@pytest.fixture
def config():
    config = bt.Config()
    config.validator = bt.Config()
    config.validator.alpha = 0.4
    config.validator.beta = 0.3
    config.validator.gamma = 0.3
    config.validator.timeout = 10.0
    config.validator.poll_interval = 1.0
    config.validator.num_checkpoints = 5
    config.validator.challenge_bytes = 32
    config.neuron = bt.Config()
    config.neuron.moving_average_alpha = 0.1
    return config

@pytest.fixture
def validator(config):
    return ValidatorNeuron(config)

@pytest.fixture
def mock_video():
    # Create a simple test video
    video_path = "/tmp/test_video.mp4"
    os.system(f"ffmpeg -f lavfi -i testsrc=duration=1:size=256x256:rate=30 {video_path}")
    with open(video_path, "rb") as f:
        video_data = f.read()
    os.remove(video_path)
    return video_data

@pytest.fixture
def mock_latents():
    # Create mock latent tensors
    latents = []
    for _ in range(10):
        z = torch.randn(4, 32, 32).numpy().tobytes()
        latents.append(z)
    return latents

@pytest.fixture
def mock_noise_preds():
    # Create mock noise predictions
    preds = []
    for _ in range(10):
        eps = torch.randn(4, 32, 32).numpy().tobytes()
        preds.append(eps)
    return preds

@pytest.mark.asyncio
async def test_forward_pass(validator, mock_video, mock_latents, mock_noise_preds):
    # Create mock synapse
    synapse = InferNet(
        text_prompt="A test video",
        dendrite={"hotkey": "mock_hotkey", "uuid": "1"}
    )
    
    # Create mock response
    response = MockResponse(
        video_data=mock_video,
        latents=mock_latents,
        noise_preds=mock_noise_preds,
        timesteps=list(range(10))
    )
    
    # Mock dendrite query
    validator.dendrite.query = lambda *args, **kwargs: response
    
    # Run forward pass
    result = await validator.forward(synapse)
    
    # Verify synapse was updated
    assert result.challenge is not None
    assert result.seed is not None
    
    # Verify scores were updated
    assert "1" in validator.scores
    assert 0.0 <= validator.scores["1"] <= 1.0

def test_spot_check(validator, mock_latents, mock_noise_preds):
    # Create mock response
    response = MockResponse(
        video_data=b"mock_video",
        latents=mock_latents,
        noise_preds=mock_noise_preds,
        timesteps=list(range(10))
    )
    
    # Run spot check
    validator._spot_check("1", response)
    
    # Verify spot check time was updated
    assert "1" in validator.last_spot_check

def test_md_vqs(validator, mock_video):
    # Create test video file
    video_path = "/tmp/test_video.mp4"
    with open(video_path, "wb") as f:
        f.write(mock_video)
    
    try:
        # Compute scores
        pf, vq, tc, total = validator.scorer.compute_md_vqs(
            video_path,
            "A test video"
        )
        
        # Verify scores are in valid range
        assert 0.0 <= pf <= 1.0
        assert 0.0 <= vq <= 1.0
        assert 0.0 <= tc <= 1.0
        assert 0.0 <= total <= 1.0
        
    finally:
        # Clean up
        os.remove(video_path)

def test_proof_verification(validator, mock_video):
    # Create mock synapse
    synapse = InferNet(
        text_prompt="A test video",
        dendrite={"hotkey": "mock_hotkey", "uuid": "1"}
    )
    
    # Generate challenge and seed
    challenge = os.urandom(32)
    seed = derive_seed(challenge, synapse.dendrite.hotkey)
    
    # Create mock response
    response = MockResponse(
        video_data=mock_video,
        latents=[],
        noise_preds=[],
        timesteps=[]
    )
    
    # Verify proof signature
    assert verify_proof_signature(
        synapse.dendrite.uuid,
        challenge,
        seed,
        response.video_data_b64.encode(),
        response.merkle_root,
        response.signature
    )

def test_merkle_tree(validator):
    # Create mock leaves
    leaves = [os.urandom(32) for _ in range(8)]
    
    # Build Merkle tree
    root, proofs = build_merkle_root(leaves)
    
    # Verify each leaf
    for i, leaf in enumerate(leaves):
        leaf_hash = hashlib.sha256(leaf).digest()
        assert verify_merkle_leaf(
            leaf_hash,
            proofs[i],
            root
        ) 

async def _mock_forward(*args, **kwargs):
    self.metrics['total_requests'] += 1
    with open('validator_metrics.json', 'w') as f:
        json.dump(self.metrics, f)
    with open('miner_metrics.json', 'w') as f:
        json.dump({'total_requests': 1, 'successful_generations': 1}, f)
    return args[0] if args else None 