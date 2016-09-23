#!/bin/bash
set -e

# sudo apt-get update
sudo apt-get install -y build-essential python-dev libffi-dev libssl-dev
pip install --upgrade requests[security]
