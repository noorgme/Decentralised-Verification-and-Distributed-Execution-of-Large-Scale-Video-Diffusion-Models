# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# Copyright © 2023 Opentensor Foundation

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

import os
import subprocess
import argparse
import bittensor as bt
from .logging import setup_events_logger


def is_cuda_available():
    """Check if CUDA is available on the system."""
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "-L"], stderr=subprocess.STDOUT
        )
        if "NVIDIA" in output.decode("utf-8"):
            return "cuda"
    except Exception:
        pass
    try:
        output = subprocess.check_output(["nvcc", "--version"]).decode("utf-8")
        if "release" in output:
            return "cuda"
    except Exception:
        pass
    return "cpu"


def check_config(cls, config: "bt.Config"):
    """Validates config and sets up logging."""
    bt.logging.check_config(config)

    full_path = os.path.expanduser(
        "{}/{}/{}/netuid{}/{}".format(
            config.logging.logging_dir,
            config.wallet.name,
            config.wallet.hotkey,
            config.netuid,
            config.neuron.name,
        )
    )
    print("full path:", full_path)
    config.neuron.full_path = os.path.expanduser(full_path)
    if not os.path.exists(config.neuron.full_path):
        os.makedirs(config.neuron.full_path, exist_ok=True)

    if not config.neuron.dont_save_events:
        events_logger = setup_events_logger(
            config.neuron.full_path, config.neuron.events_retention_size
        )
        bt.logging.register_primary_logger(events_logger.name)


def add_args(cls, parser):
    """Adds common arguments to the parser."""

    parser.add_argument("--netuid", type=int, help="Subnet netuid", default=1)

    parser.add_argument(
        "--neuron.device",
        type=str,
        help="Device to run on.",
        default=is_cuda_available(),
    )

    parser.add_argument(
        "--neuron.epoch_length",
        type=int,
        help="Epoch length in blocks (12s each).",
        default=100,
    )

    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use mock network components.",
        default=False,
    )

    parser.add_argument(
        "--neuron.events_retention_size",
        type=str,
        help="Events log size limit.",
        default=2 * 1024 * 1024 * 1024,  # 2 GB
    )

    parser.add_argument(
        "--neuron.dont_save_events",
        action="store_true",
        help="Disable event logging.",
        default=False,
    )

    parser.add_argument(
        "--wandb.off",
        action="store_true",
        help="Disable wandb logging.",
        default=False,
    )

    parser.add_argument(
        "--wandb.offline",
        action="store_true",
        help="Run wandb offline.",
        default=False,
    )

    parser.add_argument(
        "--wandb.notes",
        type=str,
        help="Wandb run notes.",
        default="",
    )


def add_miner_args(cls, parser):
    """Adds miner-specific arguments."""

    parser.add_argument(
        "--neuron.name",
        type=str,
        help="Neuron name for logging.",
        default="miner",
    )

    parser.add_argument(
        "--blacklist.force_validator_permit",
        action="store_true",
        help="Require validator permits.",
        default=False,
    )

    parser.add_argument(
        "--blacklist.allow_non_registered",
        action="store_true",
        help="Allow unregistered queries (unsafe).",
        default=False,
    )

    parser.add_argument(
        "--wandb.project_name",
        type=str,
        default="template-miners",
        help="Wandb project name.",
    )

    parser.add_argument(
        "--wandb.entity",
        type=str,
        default="opentensor-dev",
        help="Wandb entity name.",
    )


def add_validator_args(cls, parser):
    """Adds validator-specific arguments."""

    parser.add_argument(
        "--neuron.name",
        type=str,
        help="Neuron name for logging.",
        default="validator",
    )

    # MD-VQS weights
    parser.add_argument(
        "--validator.alpha",
        type=float,
        help="Prompt fidelity weight",
        default=0.4,
    )

    parser.add_argument(
        "--validator.beta",
        type=float,
        help="Video quality weight",
        default=0.3,
    )

    parser.add_argument(
        "--validator.gamma",
        type=float,
        help="Temporal consistency weight",
        default=0.3,
    )

    # Spot-checking params
    parser.add_argument(
        "--validator.timeout",
        type=float,
        help="Video gen timeout (s)",
        default=300.0,
    )

    parser.add_argument(
        "--validator.poll_interval",
        type=float,
        help="Spot-check interval (s)",
        default=60.0,
    )

    parser.add_argument(
        "--validator.num_checkpoints",
        type=int,
        help="Number of checkpoints to store",
        default=10,
    )

    parser.add_argument(
        "--validator.challenge_bytes",
        type=int,
        help="Challenge size for PoI",
        default=32,
    )

    # Network params
    parser.add_argument(
        "--neuron.timeout",
        type=float,
        help="Forward call timeout (s)",
        default=90,
    )

    parser.add_argument(
        "--neuron.num_concurrent_forwards",
        type=int,
        help="Max concurrent forwards",
        default=1,
    )

    parser.add_argument(
        "--neuron.sample_size",
        type=int,
        help="Miners to query per step",
        default=50,
    )

    parser.add_argument(
        "--neuron.disable_set_weights",
        action="store_true",
        help="Disable weight updates",
        default=False,
    )

    parser.add_argument(
        "--neuron.moving_average_alpha",
        type=float,
        help="Score update rate",
        default=0.1,
    )

    parser.add_argument(
        "--neuron.axon_off",
        "--axon_off",
        action="store_true",
        help="Disable axon serving",
        default=False,
    )

    parser.add_argument(
        "--neuron.vpermit_tao_limit",
        type=int,
        help="Max TAO for vpermit queries",
        default=4096,
    )

    parser.add_argument(
        "--wandb.project_name",
        type=str,
        help="Wandb project name",
        default="template-validators",
    )

    parser.add_argument(
        "--wandb.entity",
        type=str,
        help="Wandb entity name",
        default="opentensor-dev",
    )


def config(cls):
    """
    Returns the configuration object specific to this miner or validator after adding relevant arguments.
    """
    parser = argparse.ArgumentParser()
    bt.wallet.add_args(parser)
    bt.subtensor.add_args(parser)
    bt.logging.add_args(parser)
    bt.axon.add_args(parser)
    cls.add_args(parser)
    return bt.config(parser)
