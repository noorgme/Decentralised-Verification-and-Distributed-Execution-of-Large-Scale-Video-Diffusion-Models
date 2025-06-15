#!/bin/bash
source .env
forge create evm/contracts/InferNetRewards.sol:InferNetRewards --rpc-url http://127.0.0.1:8545 --private-key "$INFERNET_VALIDATOR_PRIVATE_KEY" --broadcast --constructor-args "$INFERNET_VALIDATOR_ADDRESS" 0x0000000000000000000000000000000000000000