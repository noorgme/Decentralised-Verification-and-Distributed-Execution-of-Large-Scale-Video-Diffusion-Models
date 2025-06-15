#!/usr/bin/env python3

import os
import time
import logging
import torch
from diffusers import DiffusionPipeline
from huggingface_hub import HfFolder, login
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_hf_session():

    session = requests.Session()
    retries = Retry(
        total=5,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504]
    )
    session.mount('https://', HTTPAdapter(max_retries=retries))
    return session

def download_model(model_id):

    token = os.getenv('HF_TOKEN')
    if not token:
        logger.warning("No HF_TOKEN found in environment variables")
        logger.warning("Please set your token using: export HF_TOKEN=your_token_here")
        logger.warning("You can get a token from https://huggingface.co/settings/tokens")
        return False


    try:
        login(token=token)
        logger.info("Successfully logged in to Hugging Face")
    except Exception as e:
        logger.error(f"Failed to login to Hugging Face: {str(e)}")
        return False


    session = setup_hf_session()
    
    max_attempts = 5
    base_delay = 10 
    
    for attempt in range(max_attempts):
        try:
            logger.info(f"Attempt {attempt + 1} to download model {model_id}")
            

            try:
                pipe = DiffusionPipeline.from_pretrained(
                    model_id,
                    torch_dtype=torch.float16,
                    low_cpu_mem_usage=True,
                    use_safetensors=False,  
                    local_files_only=True
                )
                logger.info("Successfully loaded model from local cache")
                return True
            except Exception as e:
                logger.info(f"Model not in local cache, will download: {str(e)}")
            
            # Download if not in cache
            pipe = DiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                use_safetensors=False,  
                local_files_only=False
            )
            logger.info("Successfully downloaded and loaded model")
            return True
            
        except Exception as e:
            delay = base_delay * (2 ** attempt)  # Exponential backoff
            logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
            if attempt < max_attempts - 1:
                logger.info(f"Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                logger.error(f"Failed to download model after {max_attempts} attempts")
                return False

if __name__ == "__main__":
    model_id = "cerspense/zeroscope_v2_XL"
    success = download_model(model_id)
    if not success:
        logger.error("Failed to download model. Please check your token and try again.")
        exit(1) 