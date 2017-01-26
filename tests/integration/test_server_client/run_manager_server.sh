#!/bin/sh
set -eu

if [ ${COUNT_INTEGRATION_COVERAGE:-false} = true ]; then
    TEST_COMMAND="coverage run --parallel-mode tool/profile.py"
    MAN_PORT="5008"
    ENV_PORT="5009"
else
    TEST_COMMAND="luchador"
    MAN_PORT="5000"
    ENV_PORT="5001"
fi

${TEST_COMMAND} serve manager --port ${MAN_PORT} &
sleep 3
python tests/integration/test_server_client/launch_remote_env.py --man-port ${MAN_PORT} --env-port ${ENV_PORT}
${TEST_COMMAND} exercise example/RemoteEnv.yml --port ${ENV_PORT} --agent example/DQNAgent_train.yml --kill --episode 1
