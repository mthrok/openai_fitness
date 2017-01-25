#!/bin/bash
set -eu

BASE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
DATA_DIR="${BASE_DIR}/data/env"

if [ "${COUNT_INTEGRATION_COVERAGE:-false}" = true ]; then
    TEST_COMMAND="coverage run --parallel-mode tool/profile.py"
else
    TEST_COMMAND="luchador"
fi

RETURN=0
for dir in ${DATA_DIR}/*/
do
    echo "${dir}agent.yml"
    ${TEST_COMMAND} exercise "${dir}env.yml" --agent "${dir}agent.yml" --episodes 10 --steps 500
    if [[ ! $? = 0 ]]; then
	RETURN=1
    fi
done

exit ${RETURN}
