# The MIT License (MIT)
# Copyright © 2021 Yuma Rao
# Copyright © 2023 Opentensor Foundation
# Copyright © 2023 Opentensor Technologies Inc

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
import random
import bittensor as bt


async def ping_uids(dendrite, metagraph, uids, timeout=3):
    """Pings UIDs to check network availability."""
    axons = [metagraph.axons[uid] for uid in uids]
    try:
        responses = await dendrite(
            axons,
            bt.Synapse(),  # TODO: potentially get the synapses available back?
            deserialize=False,
            timeout=timeout,
        )
        successful_uids = [
            uid
            for uid, response in zip(uids, responses)
            if response.dendrite.status_code == 200
        ]
        failed_uids = [
            uid
            for uid, response in zip(uids, responses)
            if response.dendrite.status_code != 200
        ]
    except Exception as e:
        bt.logging.error(f"Dendrite ping failed: {e}")
        successful_uids = []
        failed_uids = uids
    bt.logging.debug(f"ping() successful uids: {successful_uids}")
    bt.logging.debug(f"ping() failed uids    : {failed_uids}")
    return successful_uids, failed_uids


async def get_query_api_nodes(dendrite, metagraph, n=0.1, timeout=3):
    """Gets available API nodes based on stake and trust."""
    bt.logging.debug(
        f"Fetching available API nodes for subnet {metagraph.netuid}"
    )
    vtrust_uids = [
        uid.item()
        for uid in metagraph.uids
        if metagraph.validator_trust[uid] > 0
    ]
    top_uids = np.where(metagraph.S > np.quantile(metagraph.S, 1 - n))[
        0
    ].tolist()
    init_query_uids = set(top_uids).intersection(set(vtrust_uids))
    query_uids, _ = await ping_uids(
        dendrite, metagraph, list(init_query_uids), timeout=timeout
    )
    bt.logging.debug(
        f"Available API node UIDs for subnet {metagraph.netuid}: {query_uids}"
    )
    if len(query_uids) > 3:
        query_uids = random.sample(query_uids, 3)
    return query_uids


async def get_query_api_axons(
    wallet, metagraph=None, n=0.1, timeout=3, uids=None
):
    """Gets axons for API nodes, optionally filtered by UIDs."""
    dendrite = bt.dendrite(wallet=wallet)

    if metagraph is None:
        metagraph = bt.metagraph(netuid=21)

    if uids is not None:
        query_uids = [uids] if isinstance(uids, int) else uids
    else:
        query_uids = await get_query_api_nodes(
            dendrite, metagraph, n=n, timeout=timeout
        )
    return [metagraph.axons[uid] for uid in query_uids]
