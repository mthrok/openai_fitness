#!/bin/bash

set -eu

BASE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
DATA_DIR="${BASE_DIR}/data/dqn"

luchador "${DATA_DIR}/ALEEnvironment_train.yml" --agent "${DATA_DIR}/DQNAgent_train.yml" --episodes 50 --steps 500
