#!/bin/sh

set -eu

if [ ${COUNT_INTEGRATION_COVERAGE:-false} = true ]; then
    TEST_COMMAND1="coverage run --parallel-mode tool/profile.py"
    TEST_COMMAND2="coverage run --parallel-mode"
    PORT="12345"
else
    TEST_COMMAND1="luchador"
    TEST_COMMAND2="python"
    PORT="5000"
fi

${TEST_COMMAND1} serve env --environment example/ALEEnvironment_train.yml --port ${PORT} &
sleep 5;
${TEST_COMMAND2} tests/integration/test_server_client/run_client.py --port ${PORT}
