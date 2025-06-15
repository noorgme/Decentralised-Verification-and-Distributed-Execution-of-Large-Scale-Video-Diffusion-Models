#!/bin/bash

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Add src directory to PYTHONPATH
# export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

# Install test dependencies if not already installed
# pip install pytest pytest-asyncio

# Run the tests
python -m pytest tests/ -v

# Create test directories
mkdir -p generated_videos
mkdir -p logs

# Run tests with verbose output
python -m pytest tests/test_pipeline.py -v

# Clean up
rm -rf generated_videos
rm -f *.log
rm -f *.json 