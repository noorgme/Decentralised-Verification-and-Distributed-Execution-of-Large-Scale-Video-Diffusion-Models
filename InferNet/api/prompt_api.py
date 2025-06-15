from flask import Flask, request, jsonify
import logging
import uuid

def create_prompt_api(validator):
    app = Flask(__name__)
    logger = getattr(validator, 'logger', logging.getLogger('PromptAPI'))

    @app.route('/submit_prompt', methods=['POST'])
    def submit_prompt():
        data = request.get_json()
        request_id = data.get('request_id')
        if not request_id:
            # Generate a new request_id if not provided
            request_id = str(uuid.uuid4().int >> 64)  # 64-bit int from uuid4
            logger.info(f"Generated new request_id: {request_id}")
        user_prompt = data.get('prompt')
        logger.info(f"/submit_prompt called for request_id={request_id}")
        try:
            validator.receive_user_prompt(request_id, user_prompt)
            if hasattr(validator, '_pending_prompt_queue') and request_id in validator._pending_prompt_queue:
                validator._pending_prompt_queue.remove(request_id)
            logger.info(f"Prompt received and verified for request {request_id}")
            return jsonify({'status': 'ok', 'message': f'Prompt received for request {request_id}', 'request_id': request_id}), 200
        except Exception as e:
            logger.error(f"Prompt submission failed for request {request_id}: {e}")
            return jsonify({'status': 'error', 'message': str(e), 'request_id': request_id}), 400

    @app.route('/status/<request_id>', methods=['GET'])
    def status(request_id):
        logger.info(f"/status called for request_id={request_id}")
        info = validator.active_requests.get(request_id)
        if not info:
            logger.warning(f"Request {request_id} not found in status endpoint")
            return jsonify({'status': 'not_found'}), 404
        logger.info(f"Status for request {request_id}: {info.get('status')}")
        return jsonify({'status': info.get('status'), 'details': info}), 200

    @app.route('/result/<request_id>', methods=['GET'])
    def result(request_id):
        logger.info(f"/result called for request_id={request_id}")
        info = validator.active_requests.get(request_id)
        if not info or info.get('status') != 'completed':
            logger.warning(f"Result for request {request_id} not ready or not found")
            return jsonify({'status': 'not_ready'}), 404
        logger.info(f"Returning result for request {request_id}")
        return jsonify({'status': 'completed', 'result': info.get('result', {})}), 200

    @app.route('/refund/<request_id>', methods=['POST'])
    def refund(request_id):
        logger.info(f"/refund called for request_id={request_id}")
        try:
            # Optionally, trigger refundUnused on-chain (if allowed)
            tx_hash = validator.trigger_refund(request_id)
            logger.info(f"Refund triggered for request {request_id}, tx: {tx_hash}")
            return jsonify({'status': 'ok', 'tx_hash': tx_hash}), 200
        except Exception as e:
            logger.error(f"Refund failed for request {request_id}: {e}")
            return jsonify({'status': 'error', 'message': str(e)}), 400

    @app.route('/health', methods=['GET'])
    def health():
        logger.info("/health called")
        return jsonify({'status': 'ok'}), 200

    return app 