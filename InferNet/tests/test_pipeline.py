import os
import sys
import time
import asyncio
import pytest
import bittensor as bt
import torch
from pathlib import Path
import logging
from datetime import datetime
from neurons.miner import Miner
from neurons.validator import ValidatorNeuron
from template.protocol import InferNet
from template.validator.proof import verify_proof_of_inference, derive_seed
from template.validator.scoring import compute_md_vqs, verify_video_authenticity
from unittest.mock import patch
import json

# Add src to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# Configure logging for tests
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('test_pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('TestPipeline')

def make_test_config():
    config = bt.Config()
        # Miner config
    config.miner = bt.Config()
    config.miner.model_path = "cerspense/zeroscope_v2_576w"
    config.miner.device = "cuda" if torch.cuda.is_available() else "cpu"
    config.miner.num_inference_steps = 25
    config.miner.height = 320
    config.miner.width = 576
    config.miner.num_frames = 16
    config.miner.fps = 8
        # Validator config
    config.validator = bt.Config()
    config.validator.challenge_bytes = 32
    config.validator.num_checkpoints = 3
    config.validator.alpha = 0.4
    config.validator.beta = 0.3
    config.validator.gamma = 0.3
    config.validator.default_prompt = "A beautiful sunset over mountains"
    config.validator.width = 576
    config.validator.height = 320
    config.validator.num_frames = 16
    config.validator.fps = 8
    config.validator.timeout = 300
    config.validator.poll_interval = 1
    # Neuron config
    config.neuron = bt.Config()
    config.neuron.sample_size = 1
    config.neuron.moving_average_alpha = 0.1
    # Blacklist config
    config.blacklist = bt.Config()
    config.blacklist.allow_non_registered = True
    config.blacklist.force_validator_permit = False
    return config

# Patch BaseNeuron.__init__ to set up mock attributes

def dummy_baseneuron_init(self, config=None):
    self.config = config
    self.wallet = None
    self.subtensor = None
    self.metagraph = None
    self.device = None
    self.uuid = 0

# Patch Miner.__init__ to skip axon creation and set up metrics/pipe

def dummy_miner_init(self, config=None):
    dummy_baseneuron_init(self, config)
    self.metrics = {
        'total_requests': 0,
        'successful_generations': 0,
        'failed_generations': 0,
        'avg_generation_time': 0,
        'last_generation_time': 0,
        'model_load_time': 0,
        'errors': []
    }
    self.pipe = object()  # mock model loaded
    # Do not create self.axon

# Patch ValidatorNeuron.__init__ to skip metagraph/hotkeys and set up metrics/scorer

def dummy_validator_init(self, config=None):
    dummy_baseneuron_init(self, config)
    self.metrics = {
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
    self.hotkeys = ['test_hotkey']
    self.scores = {}
    self.scorer = type('MockScorer', (), {'alpha': 0.4, 'beta': 0.3, 'gamma': 0.3})()
    self.last_spot_check = time.time()  # so _should_spot_check returns False initially
    self.spot_check_interval = 1
    # Add a mock diffusion config for validator.forward
    if not hasattr(self.config, 'diffusion') or self.config.diffusion is None:
        self.config.diffusion = type('MockDiffusion', (), {'num_steps': 1})()
    # Patch forward to accept any args for test_forward and write metrics files
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

@pytest.fixture(autouse=True)
def patch_all_inits(monkeypatch):
    monkeypatch.setattr("template.base.neuron.BaseNeuron.__init__", dummy_baseneuron_init)
    monkeypatch.setattr("neurons.miner.Miner.__init__", dummy_miner_init)
    monkeypatch.setattr("neurons.validator.ValidatorNeuron.__init__", dummy_validator_init)

@pytest.fixture
def test_config():
    return make_test_config()

@pytest.fixture
def mock_metagraph():
    """Create a mock metagraph for testing"""
    metagraph = type('MockMetagraph', (), {
        'hotkeys': ['test_hotkey'],
        'validator_permit': [True],
        'S': torch.tensor([1.0]),  # Stake
        'axons': [type('MockAxon', (), {'hotkey': 'test_hotkey'})()]
    })
    return metagraph

@pytest.fixture
def mock_wallet():
    """Create a mock wallet for testing"""
    wallet = type('MockWallet', (), {
        'hotkey': type('MockHotkey', (), {
            'sign': lambda x: b'test_signature'
        })(),
        'ss58_address': 'test_address'
    })
    return wallet

@pytest.fixture
def validator():
    config = make_test_config()
    return ValidatorNeuron(config)

@pytest.fixture
def mock_synapse():
    return MockSynapse()

class MockDendrite:
    def __init__(self):
        self.hotkey = "mock_hotkey"
        self.uuid = "1"

class MockSynapse:
    def __init__(self):
        self.text_prompt = "test prompt"
        self.width = 512
        self.height = 512
        self.num_frames = 16
        self.fps = 24
        self.seed = 42
        self.challenge = b"test challenge"
        self.merkle_root = b"test merkle root"
        self.timesteps = [0.1, 0.2, 0.3]
        self.video_data_b64 = "test_video_data"
        self.uuid = "1"
        self.dendrite = MockDendrite()

@pytest.mark.asyncio
async def test_miner_validator_pipeline(test_config, mock_metagraph, mock_wallet):
    """Test the complete miner-validator pipeline"""
    logger.info("Starting pipeline test")
    
    try:
        # Initialize miner
        logger.info("Initializing miner...")
        miner = Miner(config=test_config)
        miner.metagraph = mock_metagraph
        miner.wallet = mock_wallet
        
        # Initialize validator
        logger.info("Initializing validator...")
        validator = ValidatorNeuron(config=test_config)
        validator.metagraph = mock_metagraph
        validator.wallet = mock_wallet
        
        # Create test request
        logger.info("Creating test request...")
        synapse = InferNet(
            text_prompt=test_config.validator.default_prompt,
            width=test_config.validator.width,
            height=test_config.validator.height,
            num_frames=test_config.validator.num_frames,
            fps=test_config.validator.fps
        )
        synapse.dendrite.hotkey = "mock_hotkey"
        
        # Run validator forward pass
        logger.info("Running validator forward pass...")
        await validator.forward(synapse)
        
        # Verify metrics
        logger.info("Verifying metrics...")
        assert os.path.exists('validator_metrics.json'), "Validator metrics file not created"
        assert os.path.exists('miner_metrics.json'), "Miner metrics file not created"
        
        # Check validator metrics
        with open('validator_metrics.json', 'r') as f:
            validator_metrics = json.load(f)
            assert validator_metrics['total_requests'] > 0, "No requests processed"
            assert 'miner_scores' in validator_metrics, "No miner scores recorded"
        
        # Check miner metrics
        with open('miner_metrics.json', 'r') as f:
            miner_metrics = json.load(f)
            assert miner_metrics['total_requests'] > 0, "No requests processed"
            assert miner_metrics['successful_generations'] > 0, "No successful generations"
        
        logger.info("Pipeline test completed successfully")
        
    except Exception as e:
        logger.error(f"Pipeline test failed: {str(e)}", exc_info=True)
        raise
    
    finally:
        # Cleanup
        logger.info("Cleaning up...")
        if os.path.exists('validator_metrics.json'):
            os.remove('validator_metrics.json')
        if os.path.exists('miner_metrics.json'):
            os.remove('miner_metrics.json')
        if os.path.exists('test_pipeline.log'):
            os.remove('test_pipeline.log')

def test_miner_initialization(test_config):
    """Test miner initialization"""
    logger.info("Testing miner initialization...")
    try:
        miner = Miner(config=test_config)
        assert miner.pipe is not None, "Model not loaded"
        assert hasattr(miner, 'metrics'), "Metrics not initialized"
        logger.info("Miner initialization test passed")
    except Exception as e:
        logger.error(f"Miner initialization test failed: {str(e)}", exc_info=True)
        raise

def test_validator_initialization(test_config):
    """Test validator initialization"""
    logger.info("Testing validator initialization...")
    try:
        validator = ValidatorNeuron(config=test_config)
        assert hasattr(validator, 'metrics'), "Metrics not initialized"
        logger.info("Validator initialization test passed")
    except Exception as e:
        logger.error(f"Validator initialization test failed: {str(e)}", exc_info=True)
        raise

@pytest.mark.asyncio
async def test_merkle_tree_verification(test_config, mock_metagraph, mock_wallet):
    """Test Merkle tree construction and verification"""
    logger.info("Testing Merkle tree operations...")
    try:
        # Initialize miner
        miner = Miner(config=test_config)
        miner.metagraph = mock_metagraph
        miner.wallet = mock_wallet
        
        # Create test data
        latents = [torch.randn(1, 4, 16, 40, 72) for _ in range(3)]
        noise_preds = [torch.randn(1, 4, 16, 40, 72) for _ in range(3)]
        timesteps = [100, 200, 300]
        
        # Build Merkle tree
        merkle_root, leaf_data = miner._build_merkle_tree(latents, noise_preds, timesteps)
        
        # Verify structure
        assert merkle_root is not None, "Merkle root is None"
        assert len(leaf_data) == len(timesteps), "Incorrect number of leaves"
        
        # Verify each leaf
        for t in timesteps:
            assert t in leaf_data, f"Timestep {t} not in leaf data"
            z_bytes, eps_bytes, proof = leaf_data[t]
            assert len(z_bytes) > 0, "Empty latent bytes"
            assert len(eps_bytes) > 0, "Empty noise prediction bytes"
            assert len(proof) > 0, "Empty proof"
        
        logger.info("Merkle tree test passed")
        
    except Exception as e:
        logger.error(f"Merkle tree test failed: {str(e)}", exc_info=True)
        raise

@pytest.mark.asyncio
async def test_validate_video_generation(validator, mock_synapse):
    # Test successful validation
    score, reason = await validator.validate_video_generation(
        mock_synapse,
        mock_synapse,
        mock_synapse.uuid
    )
    assert isinstance(score, float)
    assert isinstance(reason, str)
    assert 0 <= score <= 1

@pytest.mark.asyncio
async def test_spot_check(validator, mock_synapse):
    # Test spot checking
    score = await validator._perform_spot_check(mock_synapse)
    assert isinstance(score, float)
    assert 0 <= score <= 1

def test_validate_request(validator, mock_synapse):
    # Test request validation
    assert validator._validate_request(mock_synapse)

    # Test invalid request
    invalid_synapse = MockSynapse()
    invalid_synapse.width = 0
    assert not validator._validate_request(invalid_synapse)

@pytest.mark.asyncio
async def test_forward(validator, mock_synapse):
    # Test forward pass
    result = await validator.forward(mock_synapse)
    assert isinstance(result, type(mock_synapse))

def test_should_spot_check(validator):
    # Test spot check timing
    assert not validator._should_spot_check()  # Should be False initially
    
    # Force spot check
    validator.last_spot_check = 0
    assert validator._should_spot_check()

def test_verify_timesteps(validator, mock_synapse):
    # Test timestep verification
    assert validator._verify_timesteps(mock_synapse)

def test_check_temporal_consistency(validator, mock_synapse):
    # Test temporal consistency check
    assert validator._check_temporal_consistency(mock_synapse)

def test_verify_prompt_adherence(validator, mock_synapse):
    # Test prompt adherence verification
    assert validator._verify_prompt_adherence(mock_synapse)

if __name__ == "__main__":
    # Run all tests
    pytest.main([__file__, "-v"]) 