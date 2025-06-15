import bittensor as bt

class InferNetConfig(bt.Config):
    def __init__(self):
        super().__init__()
        
        # Security parameters 
        self.audit_rate = 0.30  # α* = 0.30
        self.slash_fraction = 0.10  # f* = 0.10
        self.trust_decay = 0.8  # γ = 0.8
        self.user_deposit_split = 0.70  # s = 0.70
        
        # Economic parameters
        self.cost_per_step = 0.0003  # Cstep from paper
        self.gas_cost = 0.001  # Cgas from paper
        
        # Quality scoring
        self.quality_threshold = 0.7  # Minimum acceptable quality score
        self.quality_weight = 0.3  # Weight of quality in reward calculation
        
        # Trust weight parameters
        self.honest_drift = 0.1  # η from paper, controls trust weight convergence
        
        # UNet configuration
        self.unet_config = {
            'latent_channels': 4,
            'latent_height': 16,
            'latent_width': 40,
            'alphas': [0.9999] * 1000  # Example alphas, should be replaced with actual scheduler values
        }
        
        # Validator configuration
        self.validator = {
            'alpha': 0.4,  # Prompt fidelity weight
            'beta': 0.3,   # Video quality weight
            'gamma': 0.3   # Temporal consistency weight
        }
        
        # Miner configuration
        self.miner = {
            'batch_size': 1,
            'num_frames': 16,
            'frame_height': 256,
            'frame_width': 256
        } 