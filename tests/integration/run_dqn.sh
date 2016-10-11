#!/bin/bash

set -e

BASE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
DATA_DIR="${BASE_DIR}/data/dqn"

if [ "${COUNT_INTEGRATION_COVERAGE}" = true ]; then
    TEST_COMMAND="coverage run --source luchador -a tool/profile.py"
else
    TEST_COMMAND="luchador"
fi

${TEST_COMMAND} "${DATA_DIR}/ALEEnvironment_train.yml" --agent "${DATA_DIR}/DQNAgent_train.yml" --episodes 20 --steps 500
