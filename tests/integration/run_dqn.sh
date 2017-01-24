#!/bin/bash
set -eux

BASE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
DATA_DIR="${BASE_DIR}/data/dqn"

if [ "${COUNT_INTEGRATION_COVERAGE:-false}" = true ]; then
    TEST_COMMAND="coverage run --parallel-mode tool/profile.py"
else
    TEST_COMMAND="luchador"
fi

${TEST_COMMAND} exercise "${DATA_DIR}/ALEEnvironment_train.yml" --agent "${DATA_DIR}/DQNAgent_train.yml" --episodes 10 --steps 500
