#!/bin/bash
# This script is intended to be run on an Amazon Linux EC2 instance
#update the system
sudo apt update
sudo apt install python3-pip 

# Create a Python virtual environment
python3 -m venv monitron-genai
source monitron-genai/bin/activate

# Please ensure you have the correct repository enabled or install method before running this command
pip install --upgrade pip
pip install --no-cache-dir -r requirements.txt

deactivate












