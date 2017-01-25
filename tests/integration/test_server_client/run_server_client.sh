#!/bin/sh
set -eu

if [ ${COUNT_INTEGRATION_COVERAGE:-false} = true ]; then
    TEST_COMMAND="coverage run --parallel-mode tool/profile.py"
    PORT="12345"
else
    TEST_COMMAND="luchador"
    PORT="12345"
fi

${TEST_COMMAND} serve env --environment example/ALEEnvironment_train.yml --port ${PORT} &
sleep 5;
${TEST_COMMAND} exercise example/RemoteEnv.yml --episode 1 --port ${PORT} --kill
