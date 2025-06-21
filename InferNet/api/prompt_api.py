from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import logging
import uuid
import time
import os
from pathlib import Path
import json

def create_prompt_api(validator):
    app = Flask(__name__)
    CORS(app, origins=['http://localhost:3000'])  # Allow requests from frontend
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
        
        logger.info(f"📝 /submit_prompt called with:")
        logger.info(f"   - request_id: {request_id}")
        logger.info(f"   - user_prompt: '{user_prompt}'")
        logger.info(f"   - data: {data}")
        
        try:
            request_id_int = int(request_id)
        except (ValueError, TypeError):
            logger.error(f"Invalid request_id format: {request_id}")
            return jsonify({'status': 'error', 'message': 'request_id must be an integer'}), 400
        
        logger.info(f"📝 Processing request_id={request_id_int}")
        
        # Wait for the request to appear in active_requests (max 120 seconds)
        max_wait_time = 120
        wait_interval = 0.5
        waited_time = 0
        
        logger.info(f"🔍 Looking for request {request_id_int} in active_requests...")
        logger.info(f"🔍 Current active_requests keys: {list(validator.active_requests.keys())}")
        
        while request_id_int not in validator.active_requests and waited_time < max_wait_time:
            logger.info(f"⏳ Request {request_id_int} not found in active_requests, waiting... (waited {waited_time}s)")
            time.sleep(wait_interval)
            waited_time += wait_interval
        
        if request_id_int not in validator.active_requests:
            logger.error(f"❌ Request {request_id_int} not found after waiting {max_wait_time} seconds")
            logger.error(f"❌ Available active_requests: {list(validator.active_requests.keys())}")
            return jsonify({
                'status': 'error', 
                'message': f'Request {request_id_int} not found. The blockchain transaction may still be processing.', 
                'request_id': request_id_int
            }), 400
        
        logger.info(f"✅ Found request {request_id_int} in active_requests")
        logger.info(f"📋 Request data: {validator.active_requests[request_id_int]}")
        
        try:
            validator.receive_user_prompt(request_id_int, user_prompt)
            if hasattr(validator, '_pending_prompt_queue') and request_id_int in validator._pending_prompt_queue:
                validator._pending_prompt_queue.remove(request_id_int)
            logger.info(f"✅ Prompt received and verified for request {request_id_int}")
            logger.info(f"📋 Updated request data: {validator.active_requests[request_id_int]}")
            return jsonify({'status': 'ok', 'message': f'Prompt received for request {request_id_int}', 'request_id': request_id_int}), 200
        except Exception as e:
            logger.error(f"❌ Prompt submission failed for request {request_id_int}: {e}")
            return jsonify({'status': 'error', 'message': str(e), 'request_id': request_id_int}), 400

    @app.route('/status/<request_id>', methods=['GET'])
    def status(request_id):
        logger.info(f"/status called for request_id={request_id}")
        try:
            request_id_int = int(request_id)
        except (ValueError, TypeError):
            logger.error(f"Invalid request_id format: {request_id}")
            return jsonify({'status': 'error', 'message': 'request_id must be an integer'}), 400
        
        info = validator.active_requests.get(request_id_int)
        if not info:
            logger.warning(f"Request {request_id_int} not found in status endpoint")
            return jsonify({'status': 'not_found'}), 404
        logger.info(f"Status for request {request_id_int}: {info.get('status')}")
        return jsonify({'status': info.get('status'), 'details': info}), 200

    @app.route('/result/<request_id>', methods=['GET'])
    def result(request_id):
        logger.info(f"/result called for request_id={request_id}")
        try:
            request_id_int = int(request_id)
        except (ValueError, TypeError):
            logger.error(f"Invalid request_id format: {request_id}")
            return jsonify({'status': 'error', 'message': 'request_id must be an integer'}), 400
        
        # Check if we have results for this request
        result_file = f"results_{request_id_int}.json"
        result_path = os.path.join(os.getcwd(), result_file)
        
        logger.info(f"Looking for result file: {result_path}")
        logger.info(f"File exists: {os.path.exists(result_path)}")
        
        if not os.path.exists(result_path):
            logger.warning(f"Result file not found: {result_path}")
            return jsonify({'status': 'not_found', 'message': 'Results not yet available'}), 404
        
        try:
            with open(result_path, 'r') as f:
                result_data = json.load(f)
            
            logger.info(f"Loaded result data for request {request_id_int}")
            logger.info(f"Result data keys: {list(result_data.keys())}")
            
            if 'result' in result_data and 'miners' in result_data['result']:
                miners = result_data['result']['miners']
                logger.info(f"Found {len(miners)} miners in result data")
                
                for i, miner in enumerate(miners):
                    logger.info(f"Miner {i}: uid={miner.get('uid')}, status={miner.get('status')}, score={miner.get('score')}")
                    logger.info(f"Miner {i}: video_path={miner.get('video_path')}")
                    
                    # Check if video file exists
                    if miner.get('video_path'):
                        video_file = miner['video_path']
                        video_exists = os.path.exists(video_file)
                        logger.info(f"Miner {i}: video file exists: {video_exists}")
                        
                        if video_exists:
                            # Convert to relative path for frontend
                            video_filename = os.path.basename(video_file)
                            miner['video_url'] = f"/videos/{video_filename}"
                            logger.info(f"Miner {i}: video_url set to: {miner['video_url']}")
                        else:
                            logger.warning(f"Miner {i}: video file does not exist: {video_file}")
                            miner['video_url'] = None
                    else:
                        logger.warning(f"Miner {i}: no video_path in result data")
                        miner['video_url'] = None
            else:
                logger.warning(f"No 'result' or 'miners' key found in result data")
            
            logger.info(f"Returning result data for request {request_id_int}")
            return jsonify(result_data)
            
        except Exception as e:
            logger.error(f"Error reading result file: {str(e)}")
            return jsonify({'status': 'error', 'message': f'Error reading results: {str(e)}'}), 500

    @app.route('/video/<filename>', methods=['OPTIONS'])
    def video_options(filename):
        """Handle CORS preflight requests for video endpoint"""
        response = jsonify({'status': 'ok'})
        response.headers['Access-Control-Allow-Origin'] = 'http://localhost:3000'
        response.headers['Access-Control-Allow-Methods'] = 'GET, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
        return response

    @app.route('/videos/<filename>', methods=['GET'])
    def serve_video(filename):
        logger.info(f"/videos/{filename} called")
        
        # Look for video in generated_videos directory
        video_path = os.path.join('generated_videos', filename)
        logger.info(f"Looking for video at: {video_path}")
        logger.info(f"Video file exists: {os.path.exists(video_path)}")
        
        if not os.path.exists(video_path):
            logger.warning(f"Video file not found: {video_path}")
            # List available videos for debugging
            generated_videos_dir = 'generated_videos'
            if os.path.exists(generated_videos_dir):
                available_videos = os.listdir(generated_videos_dir)
                logger.info(f"Available videos in {generated_videos_dir}: {available_videos}")
            else:
                logger.warning(f"Generated videos directory does not exist: {generated_videos_dir}")
            return jsonify({'error': 'Video not found'}), 404
        
        try:
            file_size = os.path.getsize(video_path)
            logger.info(f"Video file size: {file_size} bytes")
            
            # Set CORS headers
            response = send_file(video_path, mimetype='video/mp4')
            response.headers['Access-Control-Allow-Origin'] = '*'
            response.headers['Access-Control-Allow-Methods'] = 'GET, OPTIONS'
            response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
            logger.info(f"✅ Video served successfully: {filename}")
            return response
            
        except Exception as e:
            logger.error(f"Error serving video {filename}: {str(e)}")
            return jsonify({'error': f'Error serving video: {str(e)}'}), 500

    @app.route('/refund/<request_id>', methods=['POST'])
    def refund(request_id):
        logger.info(f"/refund called for request_id={request_id}")
        try:
            
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