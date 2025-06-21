#!/bin/bash

python ./neurons/miner.py --netuid 1 --wallet.name miner0 --subtensor.network local --subtensor.chain_endpoint ws://127.0.0.1:9944 --diffusion.num_steps 4 --logging.debug