# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# Developer: Noor Elsheikh
# Copyright © 2025 Noor Elsheikh

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.


import copy
import numpy as np
import asyncio
import argparse
import threading
import bittensor as bt

from typing import List, Union
from traceback import print_exception

from template.base.neuron import BaseNeuron
from template.base.utils.weight_utils import (
    process_weights_for_netuid,
    convert_weights_and_uids_for_emit,
) 
from template.mock import MockDendrite
from template.utils.config import add_validator_args


class BaseValidatorNeuron(BaseNeuron):
    """Base class for Bittensor validators."""

    neuron_type: str = "ValidatorNeuron"

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser):
        super().add_args(parser)
        add_validator_args(cls, parser)

    def __init__(self, config=None):
        super().__init__(config=config)

        # Save a copy of the hotkeys to local memory.
        self.hotkeys = copy.deepcopy(self.metagraph.hotkeys)

        # Dendrite lets us send messages to other nodes (axons) in the network.
        if self.config.mock:
            self.dendrite = MockDendrite(wallet=self.wallet)
        else:
            self.dendrite = bt.dendrite(wallet=self.wallet)
        bt.logging.info(f"Dendrite: {self.dendrite}")

        # Set up initial scoring weights for validation
        bt.logging.info("Building validation weights.")
        self.scores = np.zeros(self.metagraph.n, dtype=np.float32)

        # Init sync with the network. Updates the metagraph.
        self.sync()

        # Serve axon to enable external connections.
        if not self.config.neuron.axon_off:
            self.serve_axon()
        else:
            bt.logging.warning("axon off, not serving ip to chain.")

        # Create asyncio event loop to manage async tasks.
        self.loop = asyncio.get_event_loop()

        # Instantiate runners
        self.should_exit: bool = False
        self.is_running: bool = False
        self.thread: Union[threading.Thread, None] = None
        self.lock = asyncio.Lock()

    def serve_axon(self):
        """Start axon server."""

        bt.logging.info("serving ip to chain...")
        try:
            self.axon = bt.axon(wallet=self.wallet, config=self.config)

            try:
                self.subtensor.serve_axon(
                    netuid=self.config.netuid,
                    axon=self.axon,
                )
                bt.logging.info(
                    f"Running validator {self.axon} on network: {self.config.subtensor.chain_endpoint} with netuid: {self.config.netuid}"
                )
            except Exception as e:
                bt.logging.error(f"Failed to serve Axon with exception: {e}")
                pass

        except Exception as e:
            bt.logging.error(
                f"Failed to create Axon initialize with exception: {e}"
            )
            pass

    async def concurrent_forward(self):
        coroutines = [
            self.forward()
            for _ in range(self.config.neuron.num_concurrent_forwards)
        ]
        await asyncio.gather(*coroutines)

    def run(self):
        """Main loop: query miners, score, set weights, sync chain. Handles Ctrl-C."""

        # Check that validator is registered on the network.
        self.sync()

        bt.logging.info(f"Validator starting at block: {self.block}")

        # This loop maintains the validator's operations until intentionally stopped.
        try:
            while True:
                bt.logging.info(f"step({self.step}) block({self.block})")

                # Run multiple forwards concurrently.
                self.loop.run_until_complete(self.concurrent_forward())

                # Check if we should exit.
                if self.should_exit:
                    break

                # Sync metagraph and potentially set weights.
                self.sync()

                self.step += 1

        # If someone intentionally stops the validator, it'll safely terminate operations.
        except KeyboardInterrupt:
            self.axon.stop()
            bt.logging.success("Validator killed by keyboard interrupt.")
            exit()

        # In case of unforeseen errors, the validator will log the error and continue operations.
        except Exception as err:
            bt.logging.error(f"Error during validation: {str(err)}")
            bt.logging.debug(
                str(print_exception(type(err), err, err.__traceback__))
            )

    def run_in_background_thread(self):
        """Start validator in background thread."""
        if not self.is_running:
            bt.logging.debug("Starting validator in background thread.")
            self.should_exit = False
            self.thread = threading.Thread(target=self.run, daemon=True)
            self.thread.start()
            self.is_running = True
            bt.logging.debug("Started")

    def stop_run_thread(self):
        """Stop validator background thread."""
        if self.is_running:
            bt.logging.debug("Stopping validator in background thread.")
            self.should_exit = True
            self.thread.join(5)
            self.is_running = False
            bt.logging.debug("Stopped")

    def __enter__(self):
        self.run_in_background_thread()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Stop validator background thread on exit."""
        if self.is_running:
            bt.logging.debug("Stopping validator in background thread.")
            self.should_exit = True
            self.thread.join(5)
            self.is_running = False
            bt.logging.debug("Stopped")

    def set_weights(self):
        """
        Sets the validator weights to the metagraph hotkeys based on the scores it has received from the miners. The weights determine the trust and incentive level the validator assigns to miner nodes on the network.
        """

        # Ensure scores is a numpy array with proper dtype
        if isinstance(self.scores, dict):
            # If scores is a dictionary, convert it to a numpy array
            # Assuming the dictionary has numeric values that can be converted to scores
            try:
                # Try to extract values from the dictionary
                if hasattr(self.scores, 'values'):
                    score_values = list(self.scores.values())
                else:
                    score_values = list(self.scores.values()) if hasattr(self.scores, 'values') else []
                
                # Convert to numpy array
                self.scores = np.array(score_values, dtype=np.float32)
                bt.logging.warning(f"Converted scores from dict to numpy array with {len(self.scores)} elements")
            except Exception as e:
                bt.logging.error(f"Failed to convert scores dict to numpy array: {e}")
                # Fallback: create a zero array
                self.scores = np.zeros(self.metagraph.n, dtype=np.float32)
        elif not isinstance(self.scores, np.ndarray):
            # If it's not a dict but also not a numpy array, try to convert it
            try:
                self.scores = np.array(self.scores, dtype=np.float32)
            except Exception as e:
                bt.logging.error(f"Failed to convert scores to numpy array: {e}")
                # Fallback: create a zero array
                self.scores = np.zeros(self.metagraph.n, dtype=np.float32)
        
        # Check if self.scores contains any NaN values and log a warning if it does.
        try:
            if np.isnan(self.scores).any():
                bt.logging.warning(
                    f"Scores contain NaN values. This may be due to a lack of responses from miners, or a bug in your reward functions."
                )
        except (TypeError, ValueError) as e:
            bt.logging.warning(f"Could not check for NaN values in scores: {e}")
            # Convert to float32 to ensure compatibility
            try:
                self.scores = np.array(self.scores, dtype=np.float32)
            except Exception as e:
                bt.logging.error(f"Failed to convert scores to float32: {e}")
                self.scores = np.zeros(self.metagraph.n, dtype=np.float32)

        # Calculate the average reward for each uid across non-zero values.
        # Replace any NaN values with 0.
        # Compute the norm of the scores
        score_array = self.scores.copy()  # Use the numpy array directly
        
        # Ensure score_array is 1D
        if score_array.ndim > 1:
            score_array = score_array.flatten()
        
        norm = np.linalg.norm(score_array, ord=1)

        # Check if the norm is zero or contains NaN values
        if norm == 0 or np.isnan(norm):
            norm = 1.0  # Avoid division by zero or NaN

        # Compute raw_weights safely
        raw_weights = score_array / norm

        bt.logging.debug("raw_weights", raw_weights)
        bt.logging.debug("raw_weight_uids", str(self.metagraph.uids.tolist()))
        # Process the raw weights to final_weights via subtensor limitations.
        (
            processed_weight_uids,
            processed_weights,
        ) = process_weights_for_netuid(
            uids=self.metagraph.uids,
            weights=raw_weights,
            netuid=self.config.netuid,
            subtensor=self.subtensor,
            metagraph=self.metagraph,
        )
        bt.logging.debug("processed_weights", processed_weights)
        bt.logging.debug("processed_weight_uids", processed_weight_uids)

        # Convert to uint16 weights and uids.
        (
            uint_uids,
            uint_weights,
        ) = convert_weights_and_uids_for_emit(
            uids=processed_weight_uids, weights=processed_weights
        )
        bt.logging.debug("uint_weights", uint_weights)
        bt.logging.debug("uint_uids", uint_uids)

        # Set the weights on chain via our subtensor connection.
        result, msg = self.subtensor.set_weights(
            wallet=self.wallet,
            netuid=self.config.netuid,
            uids=uint_uids,
            weights=uint_weights,
            wait_for_finalization=False,
            wait_for_inclusion=False,
            version_key=self.spec_version,
        )
        if result is True:
            bt.logging.info("set_weights on chain successfully!")
        else:
            bt.logging.error("set_weights failed", msg)

    def resync_metagraph(self):
        """Resyncs the metagraph and updates the hotkeys and moving averages based on the new metagraph."""
        bt.logging.info("resync_metagraph()")

        # Ensure scores is a numpy array before proceeding
        if not isinstance(self.scores, np.ndarray):
            if isinstance(self.scores, dict):
                try:
                    score_values = list(self.scores.values())
                    self.scores = np.array(score_values, dtype=np.float32)
                except Exception as e:
                    bt.logging.error(f"Failed to convert scores dict to numpy array: {e}")
                    self.scores = np.zeros(self.metagraph.n, dtype=np.float32)
            else:
                try:
                    self.scores = np.array(self.scores, dtype=np.float32)
                except Exception as e:
                    bt.logging.error(f"Failed to convert scores to numpy array: {e}")
                    self.scores = np.zeros(self.metagraph.n, dtype=np.float32)

        # Copies state of metagraph before syncing.
        previous_metagraph = copy.deepcopy(self.metagraph)

        # Sync the metagraph.
        self.metagraph.sync(subtensor=self.subtensor)

        # Check if the metagraph axon info has changed.
        if previous_metagraph.axons == self.metagraph.axons:
            return

        bt.logging.info(
            "Metagraph updated, re-syncing hotkeys, dendrite pool and moving averages"
        )
        
        # Ensure scores array is the right size
        if len(self.scores) != self.metagraph.n:
            bt.logging.warning(f"Scores array size ({len(self.scores)}) doesn't match metagraph size ({self.metagraph.n})")
            # Resize scores array to match metagraph
            new_scores = np.zeros(self.metagraph.n, dtype=np.float32)
            min_len = min(len(self.scores), self.metagraph.n)
            new_scores[:min_len] = self.scores[:min_len]
            self.scores = new_scores
        
        # Zero out all hotkeys that have been replaced.
        for uid, hotkey in enumerate(self.hotkeys):
            if uid < len(self.scores) and hotkey != self.metagraph.hotkeys[uid]:
                self.scores[uid] = 0  # hotkey has been replaced

        # Check to see if the metagraph has changed size.
        # If so, we need to add new hotkeys and moving averages.
        if len(self.hotkeys) < len(self.metagraph.hotkeys):
            # Update the size of the moving average scores.
            new_moving_average = np.zeros((self.metagraph.n), dtype=np.float32)
            min_len = min(len(self.hotkeys), len(self.scores))
            new_moving_average[:min_len] = self.scores[:min_len]
            self.scores = new_moving_average

        # Update the hotkeys.
        self.hotkeys = copy.deepcopy(self.metagraph.hotkeys)

    def update_scores(self, rewards: np.ndarray, uids: List[int]):
        """Performs exponential moving average on the scores based on the rewards received from the miners."""

        # Ensure self.scores is a numpy array
        if not isinstance(self.scores, np.ndarray):
            if isinstance(self.scores, dict):
                try:
                    score_values = list(self.scores.values())
                    self.scores = np.array(score_values, dtype=np.float32)
                except Exception as e:
                    bt.logging.error(f"Failed to convert scores dict to numpy array: {e}")
                    self.scores = np.zeros(self.metagraph.n, dtype=np.float32)
            else:
                try:
                    self.scores = np.array(self.scores, dtype=np.float32)
                except Exception as e:
                    bt.logging.error(f"Failed to convert scores to numpy array: {e}")
                    self.scores = np.zeros(self.metagraph.n, dtype=np.float32)

        # Check if rewards contains NaN values.
        if np.isnan(rewards).any():
            bt.logging.warning(f"NaN values detected in rewards: {rewards}")
            # Replace any NaN values in rewards with 0.
            rewards = np.nan_to_num(rewards, nan=0)

        # Ensure rewards is a numpy array.
        rewards = np.asarray(rewards)

        # Check if `uids` is already a numpy array and copy it to avoid the warning.
        if isinstance(uids, np.ndarray):
            uids_array = uids.copy()
        else:
            uids_array = np.array(uids)

        # Handle edge case: If either rewards or uids_array is empty.
        if rewards.size == 0 or uids_array.size == 0:
            bt.logging.info(f"rewards: {rewards}, uids_array: {uids_array}")
            bt.logging.warning(
                "Either rewards or uids_array is empty. No updates will be performed."
            )
            return

        # Check if sizes of rewards and uids_array match.
        if rewards.size != uids_array.size:
            raise ValueError(
                f"Shape mismatch: rewards array of shape {rewards.shape} "
                f"cannot be broadcast to uids array of shape {uids_array.shape}"
            )

        # Compute forward pass rewards, assumes uids are mutually exclusive.
        # shape: [ metagraph.n ]
        scattered_rewards: np.ndarray = np.zeros_like(self.scores)
        scattered_rewards[uids_array] = rewards
        bt.logging.debug(f"Scattered rewards: {rewards}")

        # Update scores with rewards produced by this step.
        # shape: [ metagraph.n ]
        alpha: float = self.config.neuron.moving_average_alpha
        self.scores: np.ndarray = (
            alpha * scattered_rewards + (1 - alpha) * self.scores
        )
        bt.logging.debug(f"Updated moving avg scores: {self.scores}")

    def save_state(self):
        """Saves the state of the validator to a file."""
        bt.logging.info("Saving validator state.")

        # Save the state of the validator to file.
        np.savez(
            self.config.neuron.full_path + "/state.npz",
            step=self.step,
            scores=self.scores,
            hotkeys=self.hotkeys,
        )

    def load_state(self):
        """Loads the state of the validator from a file."""
        bt.logging.info("Loading validator state.")

        # Load the state of the validator from file.
        state = np.load(self.config.neuron.full_path + "/state.npz")
        self.step = state["step"]
        self.scores = state["scores"]
        self.hotkeys = state["hotkeys"]
