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

import typing
import bittensor as bt
import base64
from pydantic import validator


class InferNet(bt.Synapse):
    """Video generation request protocol."""
    text_prompt: str
    width: int = 256
    height: int = 256
    num_frames: int = 16
    fps: int = 8
    seed: typing.Optional[int] = None
    challenge: typing.Optional[str] = None
    request_id: typing.Optional[str] = None
    merkle_root: typing.Optional[str] = None
    signature: typing.Optional[str] = None
    timesteps: typing.Optional[typing.List[int]] = None
    latents: typing.Optional[typing.List[str]] = None
    noise_preds: typing.Optional[typing.List[str]] = None
    video_data_b64: typing.Optional[str] = None
    proof: typing.Optional[typing.Dict[str, typing.Any]] = None
    
    @validator('width', 'height', 'num_frames', 'fps', 'seed', pre=True)
    def ensure_int(cls, v):
        """Convert string params to int."""
        if isinstance(v, str):
            return int(v)
        return v
    
    @validator('timesteps', pre=True)
    def ensure_timesteps_int(cls, v):
        """Convert timesteps list to integers."""
        if v is None:
            return None
        if isinstance(v, list):
            return [int(x) if isinstance(x, str) else x for x in v]
        return v
    
    def deserialize(self) -> bytes:
        """Decode base64 video data."""
        if self.video_data_b64 is None:
            return b""
        try:
            return base64.b64decode(self.video_data_b64)
        except Exception as e:
            bt.logging.error(f"Error deserializing video data: {str(e)}")
            return b""

    def __getitem__(self, key):
        return getattr(self, key)


class RevealLeavesSynapse(bt.Synapse):
    """Protocol for revealing merkle tree leaves."""
    indices: typing.List[int]
    request_id: str
    caller_hotkey: str
    leaves: typing.Optional[typing.Dict[int, typing.Tuple[str, str, typing.List[str]]]] = None
    
    @validator('leaves', pre=True)
    def _decode_leaves_b64(cls, v):
        """Decode base64 leaf data back to bytes."""
        if v is None:
            return None
        try:
            decoded_leaves = {}
            for idx, (z_b64, eps_b64, proof_path_b64) in v.items():
                z_bytes = base64.b64decode(z_b64)
                eps_bytes = base64.b64decode(eps_b64)
                proof_path_bytes = [base64.b64decode(p) for p in proof_path_b64]
                decoded_leaves[idx] = (z_bytes, eps_bytes, proof_path_bytes)
            return decoded_leaves
        except Exception:
            raise ValueError(f"Invalid base64 in leaves data")
