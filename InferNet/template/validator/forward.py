# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
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

import time
import os
import bittensor as bt
import base64
import traceback

from template.protocol import InferNet
from template.validator.reward import get_rewards
from template.utils.uids import get_random_uids


async def forward(self):
    """Runs one validation round - queries miners for video generation and updates weights."""
    try:
        # Get random miners to query
        miner_uids = get_random_uids(self, k=self.config.neuron.sample_size)
        
        if len(miner_uids) == 0:
            bt.logging.warning("No miners available to query")
            time.sleep(10)
            return
            
        bt.logging.info(f"Selected miners: {miner_uids}")
        
        # Simple test prompt
        prompt = f"A space shuttle launch, detailed and realistic"
        
        bt.logging.info(f"Sending video generation request with prompt: '{prompt}'")
        
        # Create output dir for videos
        os.makedirs("generated_videos", exist_ok=True)
        
        # Get valid axons
        valid_axons = []
        valid_uids = []
        
        for uid in miner_uids:
            try:
                axon = self.metagraph.axons[uid]
                if axon and axon.ip and axon.port:
                    valid_axons.append(axon)
                    valid_uids.append(uid)
            except Exception as e:
                bt.logging.warning(f"Error accessing axon for UID {uid}: {str(e)}")
        
        if not valid_axons:
            bt.logging.warning("No valid axons found among selected miners")
            time.sleep(10)
            return
            
        bt.logging.info(f"Querying {len(valid_axons)} valid miners")
        
        # Query the network
        responses = await self.dendrite(
            axons=valid_axons,
            synapse=InferNet(
                text_prompt=prompt,
                width=256,
                height=256,
                num_frames=16,
                fps=8
            ),
            deserialize=True,
            timeout=500  # Longer timeout for video gen
        )
        
        # Process responses
        rewards = []
        for i, (response, uid) in enumerate(zip(responses, valid_uids)):
            bt.logging.info(f"Processing response from miner {uid}")
            if response is not None and len(response) > 0:
                # Save video
                video_path = f"./generated_videos/video_{uid}_{int(time.time())}.mp4"
                try:
                    with open(video_path, "wb") as f:
                        f.write(response)
                    bt.logging.info(f"Saved video from miner {uid} to {video_path}, size: {len(response)} bytes")
                    rewards.append(1.0)
                except Exception as e:
                    bt.logging.error(f"Error saving video from miner {uid}: {str(e)}")
                    rewards.append(0.0)
            else:
                bt.logging.warning(f"Miner {uid} returned no video data")
                rewards.append(0.0)
        
        bt.logging.info(f"Scored responses: {rewards}")
        
        # Update scores for queried miners
        if rewards and valid_uids:
            self.update_scores(rewards, valid_uids)
    except Exception as e:
        bt.logging.error(f"Error in forward pass: {str(e)}")
        bt.logging.error(traceback.format_exc())
    
    # Wait before next round
    bt.logging.info("Waiting for next round of requests...")
    time.sleep(120) 
