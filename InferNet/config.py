import bittensor as bt

class InferNetConfig(bt.Config):
    def __init__(self):
        super().__init__()
        
        # Bittensor required configuration
        self.neuron = bt.Config()
        self.neuron.name = "validator"
        self.neuron.device = "cuda"
        self.neuron.epoch_length = 100
        self.neuron.dont_save_events = False
        self.neuron.events_retention_size = 1000
        self.neuron.vpermit_tao_limit = 0.0
        self.neuron.sample_size = 1
        self.neuron.moving_average_alpha = 0.1
        self.neuron.disable_set_weights = False
        self.neuron.axon_off = False
        self.neuron.num_concurrent_forwards = 1
        self.neuron.full_path = './neurons'
        
        # Subtensor configuration
        self.subtensor = bt.Config()
        self.subtensor.network = "local"
        self.subtensor.chain_endpoint = "ws://127.0.0.1:9944"
        self.subtensor._mock = False
        
        # Wallet configuration
        self.wallet = bt.Config()
        self.wallet.name = "validator0"
        self.wallet.hotkey = "default"
        self.wallet.path = "~/.bittensor/wallets/"
        
        # Logging configuration
        self.logging = bt.Config()
        self.logging.debug = True
        self.logging.trace = False
        self.logging.info = False
        self.logging.record_log = False
        self.logging.logging_dir = "~/.bittensor/validators"
        
        # Axon configuration
        self.axon = bt.Config()
        self.axon.port = 8091
        self.axon.ip = "[::]"
        self.axon.external_port = None
        self.axon.external_ip = None
        self.axon.max_workers = 10
        
        # Blacklist configuration
        self.blacklist = bt.Config()
        self.blacklist.force_validator_permit = False
        self.blacklist.allow_non_registered = False
        
        # Mock configuration
        self.mock = False
        self.config = False
        self.strict = False
        self.no_version_checking = False
        
        # Network configuration
        self.netuid = 1
        
        # Security parameters 
        self.audit_rate = 0.30  # α* = 0.30
        self.slash_fraction = 0.10  # f* = 0.10
        self.trust_decay = 0.8  # γ = 0.8
        self.user_deposit_split = 0.70  # s = 0.70
        
        # Economic parameters
        self.cost_per_step = 0.0003  # Cstep 
        self.gas_cost = 0.001  # Cgas 
        
        # Quality scoring
        self.quality_threshold = 0.7  # Minimum acceptable quality score
        self.quality_weight = 0.3  # Weight of quality in reward calculation
        
        # Trust weight parameters
        self.honest_drift = 0.1  # η, controls trust weight convergence
        
        # UNet configuration
        self.unet_config = {
            'latent_channels': 4,
            'latent_height': 16,  # 128 // 8 for minimal height
            'latent_width': 16,   # 128 // 8 for minimal width
            'alphas': [0.9999, 0.9998, 0.9997, 0.9996]  # 4 steps for ultra-fast diffusion
        }
        
        # Validator configuration
        self.validator = {
            'alpha': 0.4,  # Prompt fidelity weight
            'beta': 0.3,   # Video quality weight
            'gamma': 0.3,  # Temporal consistency weight
            'default_prompt': 'A beautiful sunset over mountains',
            'width': 128,  # Minimal video width 
            'height': 128,  # Minimal video height 
            'num_frames': 3,  # Minimal number of frames 
            'fps': 1,  # Minimal FPS 
            'challenge_bytes': 32,  # Challenge bytes for proof
            'num_checkpoints': 3,  # Number of checkpoints for spot-checking
            'timeout': 300,  # Timeout for miner requests
            'poll_interval': 3,  # Poll interval for checking requests
            'spot_check_interval': 60  # Interval between spot checks
        }
        
        # Miner configuration
        self.miner = {
            'batch_size': 1,
            'num_frames': 3,  # Minimal frames
            'frame_height': 128,  # Minimal height
            'frame_width': 128   # Minimal width
        }
        
        # Diffusion configuration
        self.diffusion = {
            'num_steps': 4,  # Number of denoising steps (reduced from 10)
            'guidance_scale': 7.5,  # Guidance scale for generation
            'eta': 0.0  # Eta parameter for DDIM
        }
        
        # Neuron configuration
        self.neuron = {
            'sample_size': 1,  # Number of miners to sample
            'moving_average_alpha': 0.1,  # Moving average parameter for scores
            'device': 'cuda',  # Device to use for computation
            'epoch_length': 100,  # Length of an epoch
            'disable_set_weights': False,  # Whether to disable weight setting
            'axon_off': False,  # Whether to turn off axon
            'num_concurrent_forwards': 1,  # Number of concurrent forwards
            'full_path': './neurons',  # Full path for neuron data
            'dont_save_events': False,  # Whether to save events
            'events_retention_size': 1000,  # Number of events to retain
            'vpermit_tao_limit': 0.0  # TAO limit for validator permit
        } 