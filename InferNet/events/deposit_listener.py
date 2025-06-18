import logging
import time
from web3 import Web3
from typing import Dict, Any

logger = logging.getLogger('DepositListener')

def start_deposit_listener(validator, contract, w3, config):
    """
    Background thread to listen for Deposit events from the smart contract.
    When a deposit is detected, it triggers the validation process.
    """
    logger = getattr(validator, 'logger', None)
    block_file = 'last_seen_block.txt'
    # Load last-seen block from file, or start from 0
    try:
        with open(block_file, 'r') as f:
            last_seen_block = int(f.read().strip())
    except Exception:
        last_seen_block = 0
    if logger:
        logger.info(f"Starting Deposit event listener from block {last_seen_block}")
    else:
        print(f"Starting Deposit event listener from block {last_seen_block}")
    event_filter = contract.events.Deposit.createFilter(fromBlock=last_seen_block)
    REFUND_TIMEOUT = 10 * 60  # 10 minutes
    while True:
        try:
            new_events = event_filter.get_new_entries()
            for event in new_events:
                request_id = event['args']['requestId']
                user = event['args']['user']
                amount = event['args']['amount']
                prompt_hash = event['args']['promptHash']  # commit-reveal
                validator.active_requests[request_id] = {
                    'user': user,
                    'amount': amount,
                    'promptHash': prompt_hash,
                    'status': 'pending',
                    'created_at': time.time(),
                }
                msg = f"New on-chain request: {request_id} from {user} amount {amount} promptHash {prompt_hash.hex()}"
                if logger:
                    logger.info(msg)
                else:
                    print(msg)
            # Persist the last-seen block number
            if new_events:
                last_block = new_events[-1]['blockNumber']
                with open(block_file, 'w') as f:
                    f.write(str(last_block))
            # --- Check for stuck requests and trigger refunds ---
            now = time.time()
            for rid, info in list(validator.active_requests.items()):
                if info.get('status') == 'pending' and (now - info.get('created_at', now)) > REFUND_TIMEOUT:
                    msg = f"Request {rid} is stuck (no miner responded in {REFUND_TIMEOUT//60} min), marking as failed and attempting refund."
                    if logger:
                        logger.warning(msg)
                    else:
                        print(msg)
                    info['status'] = 'failed'
                    
                    try:
                        tx_data = contract.functions.refundUnused(int(rid))
                        tx_dict = {
                            'from': validator.VALIDATOR_ETH_ADDRESS if hasattr(validator, 'VALIDATOR_ETH_ADDRESS') else config.VALIDATOR_ETH_ADDRESS,
                            'nonce': w3.eth.get_transaction_count(
                                validator.VALIDATOR_ETH_ADDRESS if hasattr(validator, 'VALIDATOR_ETH_ADDRESS') else config.VALIDATOR_ETH_ADDRESS),
                        }
                        try:
                            tx_dict['gas'] = tx_data.estimate_gas(tx_dict)
                        except Exception as e:
                            if logger:
                                logger.warning(f"Gas estimation failed for refundUnused, using fallback: {e}")
                            tx_dict['gas'] = 100000
                        try:
                            tx_dict['gasPrice'] = w3.eth.gas_price
                        except Exception as e:
                            if logger:
                                logger.warning(f"Failed to fetch gas price, using fallback: {e}")
                            tx_dict['gasPrice'] = w3.to_wei('5', 'gwei')
                        tx = tx_data.build_transaction(tx_dict)
                        signed_tx = w3.eth.account.sign_transaction(
                            tx, private_key=validator.VALIDATOR_PRIVATE_KEY if hasattr(validator, 'VALIDATOR_PRIVATE_KEY') else config.VALIDATOR_PRIVATE_KEY)
                        tx_hash = w3.eth.send_raw_transaction(signed_tx.rawTransaction)
                        if logger:
                            logger.info(f"refundUnused tx: {tx_hash.hex()}")
                        else:
                            print(f"refundUnused tx: {tx_hash.hex()}")
                    except Exception as e:
                        if logger:
                            logger.error(f"Failed to call refundUnused for request {rid}: {e}")
                        else:
                            print(f"Failed to call refundUnused for request {rid}: {e}")
        except Exception as e:
            if logger:
                logger.error(f"Error in deposit event listener: {e}")
            else:
                print(f"Error in deposit event listener: {e}")
        time.sleep(2) 