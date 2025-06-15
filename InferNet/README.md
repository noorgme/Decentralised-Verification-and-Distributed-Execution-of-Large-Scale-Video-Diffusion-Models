# InferNet: Decentralised Verification and Execution of Latent Video Diffusion Models

InferNet is a comprehensive end-to-end proof of concept for a decentralised framework for video generation using latent diffusion models. This project demonstrates the technical and economic viability of decentralised video generation, with detailed economic analysis showing potential cost advantages over traditional cloud compute solutions.

## Overview

The core contribution of this network is the proof-of-inference design underpinning its economic parameters. We show that we can cheaply verify and enforce that miners never tamper with their executed inference requests, and that doing so always leads to a negative expected pay-off for a fraudulent miner.

This proof of concept implements a CLIP scoring system to maintain quality in tandem with penalties such as stake-slashing and miner-trust de-weighting. The framework is designed to be economically sound and potentially more economical than cloud compute, as demonstrated in the economic analysis in `../Economics/`. This analysis shows that under certain conditions, a decentralised network can provide video generation services at a lower cost than centralised cloud providers, while maintaining quality through cryptographic verification.


## Key Features

- Proof-of-Inference commit-then-reveal verification using Merkle trees
- Trust-based scoring and slashing mechanism
- Commit-reveal protocol for prompt delivery
- On-chain reward distribution
- Frame-wise CLIP similarity scoring
- Multi-Dimensional Video Quality scoring

## Important Note

This is a proof of concept implementation and is not production-ready. The code is provided to demonstrate the technical and economic feasibility of the framework. 

## Economic Analysis

The economic viability of this framework is analysed in detail in `../Economics/`. The analysis shows that:

1. The framework can be economically competitive with cloud compute, implementing a proof-of-inference which requires only a small fraction of U-Net denoising step to be re-executed cheaply.
2. The trust-based scoring system effectively maintains quality
3. The crypto-economic mechanisms align incentives for honest behavior
4. The network can scale efficiently with demand


## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Running a Validator

```bash
python neurons/validator.py
```

### Running a Miner

```bash
python neurons/miner.py
```

## Configuration

Key parameters can be configured in `config.py`:

- Security thresholds
- Economic parameters
- Network settings
- Model configurations

## Notes

- Unless utilising distributed configurations such as in ../Distribution, single-device miners should aim for a GPU VRAM of 16 GB to support models tested so far (Zeroscope v2 XL).


## License

This project is licensed under the MIT License. See LICENSE file for details.

## Acknowledgements

See ATTRIBUTION.md for a complete list of acknowledgements and third-party components used in this proof of concept. 