# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# Developer: Noor Elsheikh
# Copyright © 2025 Noor Elsheikh

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
import numpy as np
from typing import List
import bittensor as bt


def reward(query: int, response: int) -> float:
    """Simple check: if the miner doubled the input, give 1.0 else 0."""
    bt.logging.info(
        f"In rewards, query val: {query}, response val: {response}, rewards val: {1.0 if response == query * 2 else 0}"
    )
    return 1.0 if response == query * 2 else 0


def get_rewards(
    self,
    query: int,
    responses: List[float],
) -> np.ndarray:
    """Vectorised wrapper around `reward` for many responses."""
    # Get all the reward results by iteratively calling your reward() function.

    return np.array([reward(query, response) for response in responses])
