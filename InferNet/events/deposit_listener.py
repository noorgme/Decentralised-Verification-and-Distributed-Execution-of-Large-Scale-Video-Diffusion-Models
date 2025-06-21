import logging
import os 
import time
from web3 import Web3
from typing import Dict, Any

def start_deposit_listener(validator, contract, w3, config):
    """
    Background thread to listen for Deposit events from the smart contract.
    When a deposit is detected, it triggers the validation process.
    """
    # Get logger from validator
    validator_logger = getattr(validator, 'logger', None)
    logger = validator_logger or logging.getLogger('DepositListener')
    
    # Always start fresh: delete old pointer
    try:
        os.remove('last_seen_block.txt')
    except FileNotFoundError:
        pass

    
    if logger:
        logger.info("=== DEPOSIT LISTENER STARTING ===")
        logger.info(f"Contract address: {contract.address}")
        logger.info(f"Web3 connected: {w3.is_connected()}")
    else:
        print("=== DEPOSIT LISTENER STARTING ===")
        print(f"Contract address: {contract.address}")
        print(f"Web3 connected: {w3.is_connected()}")
    
    # build filter from head+1
    try:
        event_filter = contract.events.Deposit.create_filter(from_block='latest')
        if logger:
            logger.info(f"Event filter created successfully: {event_filter}")
        else:
            print(f"Event filter created successfully: {event_filter}")
    except Exception as e:
        if logger:
            logger.error(f"Failed to create event filter: {e}")
        else:
            print(f"Failed to create event filter: {e}")
        return
    
    # Debug: Log the event filter details
    if logger:
        logger.info(f"Created event filter for contract {contract.address}")
        logger.info(f"Event filter: {event_filter}")
        logger.info(f"Listening for Deposit events from block latest")
    else:
        print(f"Created event filter for contract {contract.address}")
        print(f"Event filter: {event_filter}")
        print(f"Listening for Deposit events from block latest")
    
    REFUND_TIMEOUT = 10 * 60  # 10 minutes
    loop_count = 0
    while True:
        try:
            loop_count += 1
            if loop_count % 30 == 0:  # Log every 30 loops (60 seconds)
                if logger:
                    logger.info(f"Event listener still running, loop {loop_count}")
                else:
                    print(f"Event listener still running, loop {loop_count}")
            
            new_events = event_filter.get_new_entries()
            
            # Debug: Log when we check for events
            if logger:
                logger.debug(f"Checking for new events, found {len(new_events)} events")
            else:
                print(f"Checking for new events, found {len(new_events)} events")
            
            for raw_evt in new_events:
                
                if logger:
                    logger.info(f"Raw event: {raw_evt}")
                else:
                    print(f"Raw event: {raw_evt}")
                
                decoded_event = raw_evt  # already decoded
                request_id  = decoded_event['args']['requestId']
                user        = decoded_event['args']['user']
                amount      = decoded_event['args']['amount']
                prompt_hash = decoded_event['args']['promptHash']  # commit-reveal
                
                # Add the request to active_requests
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
                    
                # Log that the request is now available for API calls
                if logger:
                    logger.info(f"Request {request_id} added to active_requests and ready for prompt submission")
                else:
                    print(f"Request {request_id} added to active_requests and ready for prompt submission")
            # Persist the last-seen block number
            # if new_events:
            #     last_block = new_events[-1]['blockNumber']
            #     with open(block_file, 'w') as f:
            #         f.write(str(last_block))
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
                                logger.warning(f"Failed to fetch  price, using fallback: {e}")
                            tx_dict['gasPrice'] = w3.to_wei('5', 'gwei')
                        tx = tx_data.build_transaction(tx_dict)
                        signed_tx = w3.eth.account.sign_transaction(
                            tx, private_key=validator.VALIDATOR_PRIVATE_KEY if hasattr(validator, 'VALIDATOR_PRIVATE_KEY') else config.VALIDATOR_PRIVATE_KEY)
                        tx_hash = w3.eth.send_raw_transaction(signed_tx.raw_transaction)
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