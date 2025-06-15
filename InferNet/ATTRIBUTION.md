# Attribution

This project builds upon several open-source components. Below is a clear delineation of original work versus third-party code.

## Original Work

The following components were developed specifically for InferNet:

### Core Protocol
- `neurons/validator.py`: Complete validator implementation with trust-based scoring, slashing mechanism, and proof verification
- `neurons/miner.py`: Complete miner implementation with UNet model integration, proof generation, and distributed inference
- `template/validator/`:
  - `scoring.py`: Frame-wise CLIP similarity and Multi-Dimensional Video Quality scoring
  - `proof.py`: Proof-of-Inference verification system with:
    - Merkle tree construction and verification
    - Digital signature verification
    - UNet step verification
    - Commit-reveal spot-checking protocol
  - `forward.py`: Request handling and response validation
  - `reward.py`: Trust-based reward distribution and slashing conditions
- `template/protocol.py`: Custom protocol implementation for secure video generation

### Testing
- `tests/test_validator.py`: Test suite for:
  - Proof signature verification
  - Merkle tree construction and validation
  - Spot-checking protocol
- `tests/test_pipeline.py`: End-to-end tests for:
  - Merkle tree operations
  - UNet step verification
  - Complete proof-of-inference flow

### Smart Contracts
- `evm/contracts/InferNetRewards.sol`: Smart contracts for managing the unique crypto-economic mechanisms of InferNet

### API Layer
- `api/prompt_api.py`: REST API for prompt delivery and commit-reveal protocol

### Frontend
- `frontend/`: React-based web interface for submitting inference requests with user deposits

### Configuration
- `config.py`: Network configuration parameters including security thresholds and economic parameters
- `template/subnet_links.py`: Subnet-specific protocol definitions

## Third-Party Components

The following components are derived from or based on third-party work:

### Bittensor Framework
- Basic neuron class structure and network communication utilities
- Metagraph and wallet management
- Basic validator/miner template structure

### Open Source Models
- CLIP model for frame similarity scoring (Hugging Face Transformers)
- UNet model for video generation - initially tested with ModelScope models from Hugging Face
- Merkle tree implementation for proof verification (OpenZeppelin)

### Frontend Dependencies
- React for UI components
- Web3.js for blockchain interaction
- Tailwind CSS for styling

## Modifications

The following components were significantly modified from their original sources:

### Protocol Modifications
- `template/validator/scoring.py`: Implemented frame-wise CLIP similarity and MDVQS from scratch, with spot-checking for proof verification
- `template/base/validator.py`: Implemented trust-based scoring with slashing mechanism, commit-reveal protocol, and on-chain reward distribution
- `template/base/miner.py`: Implemented UNet model integration with Merkle tree proof generation and distributed inference support
- `template/protocol.py`: Designed custom protocol for secure video generation with:
  - Commit-reveal for prompt delivery
  - Merkle tree-based proof-of-inference
  - Trust weight decay for dishonest miners
  - Spot-checking mechanism for verification

### Smart Contract Modifications
- Modified OpenZeppelin's Merkle tree implementation for proof verification
- Implemented custom staking and slashing conditions

## License

This project is licensed under the MIT License. See LICENSE file for details.

## Acknowledgements

- Opentensor Foundation and Bittensor team for the base framework utilities and their pioneering intelligence market design
- Hugging Face for CLIP and UNet models
- OpenZeppelin for smart contract libraries
- React and Web3.js communities
- All other open-source contributors whose work made this project possible 