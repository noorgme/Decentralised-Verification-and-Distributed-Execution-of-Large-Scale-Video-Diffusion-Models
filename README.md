# Decentralised Verification and Distributed Execution of Large-Scale Video Diffusion Models

This repository implements a framework addressing two key challenges in large-scale Latent Video Diffusion Models (LVDMs): accessibility and verification. LVDMs have become the leading video synthesis paradigm, achieving superior output fidelity and temporal consistency. However, they often exceed 5 billion parameters and require tens of gigabytes of activation memory, making them inaccessible with commodity hardware despite increasing demand.

## Key Contributions

### 1. Hybrid Distributed Inference Strategy
Multi-stage Experiments to derive and benchmark a hybrid static-plus-dynamic distributed inference strategy, with context coherency injection and frame-segment smoothing, that enables multiple commodity GPUs to collaboratively execute massive-scale models:
- FSDP Model parameter sharding with spatio-temporal latent partitioning
- Up to 85% reduction in peak VRAM requirements per device
- Context coherence enhancements to address temporal coherence loss, but significant artefacts remain. 
- Enables SOTA latent video diffusion models on consumer-level setups at the trade-off of latency and added artefacts from loss of temporal separation between frame-segment generation.

### 2. InferNet: Decentralised Verifiable Network - (PROOF OF CONCEPT)
Design and implementation of a decentralised network transforming idle, heterogeneous consumer GPUs into cooperative miner nodes:
- Novel commit-then-reveal Proof-of-Inference verification scheme
- Merkle-tree commitments with randomised spot-checking
- Under 10% re-execution required for fraud detection
- Game-theoretically irrational for miners to commit fraud
- Trust-accruing, stake-slashing and reward-weighting mechanics

### 3. Economic Framework
Simulation and analysis demonstrating the economic viability of the framework:
- Potential cost advantages over centralized cloud approaches
- Cryptographic auditability absent from traditional services
- Two-stage optimisation for secure parameter space
- Economically viable user deposit bounds

## Components

### Distribution
Implementation of the hybrid distributed inference strategy:
- [Fully Sharded Data Parallelism (FSDP)](Distribution/fsdp.py)
- [Latent-space partitioning](Distribution/chunk_only.py)
- [Hybrid Strategy with Coherence optimisations](Distribution/fsdp_chunked_coherent.py)

### InferNet
End-to-end implementation of the decentralised network:
- [Validator implementation](InferNet/neurons/validator.py)
- [Miner implementation](InferNet/neurons/miner.py)
- [Proof-of-Inference verification](InferNet/template/validator/proof.py)
- [Economic parameters](InferNet/config.py)

### Economics
Analysis and simulation of the economic framework:
- [Cost comparison with cloud compute](Economics/cost_step_plots.py)
- [Parameter optimisation](Economics/parameter_tuning.py)
- [Fraud detection analysis](Economics/tamper_rate_detection.py)

## Getting Started

See individual component READMEs.:
- [InferNet](InferNet/README.md)
- [Distribution](Distribution/README.md)
- [Economics](Economics/README.md)

## License

This project is licensed under the MIT License - see the LICENSE file for details. 