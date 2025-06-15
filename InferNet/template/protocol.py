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
    challenge: typing.Optional[bytes] = None
    merkle_root: typing.Optional[bytes] = None
    signature: typing.Optional[bytes] = None
    timesteps: typing.Optional[typing.List[int]] = None
    video_data_b64: typing.Optional[str] = None
    
    @validator('width', 'height', 'num_frames', 'fps', 'seed', pre=True)
    def ensure_int(cls, v):
        """Convert string params to int."""
        if isinstance(v, str):
            return int(v)
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
    leaves: typing.Optional[typing.Dict[int, typing.Tuple[bytes, bytes, typing.List[bytes]]]] = None
