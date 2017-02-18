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

echo "*** Launching Manager Server"
${TEST_COMMAND} serve manager --port ${MAN_PORT} &
PID="$!"
sleep 3
echo "*** Testing environment launch"
python tests/integration/test_server_client/launch_remote_env.py --man-port ${MAN_PORT} --env-port ${ENV_PORT}
echo "*** Testing running remote env"
${TEST_COMMAND} exercise example/RemoteEnv.yml --port ${ENV_PORT} --agent example/DQNAgent_train.yml --kill --episode 1
echo "*** Killing manager server"
kill "${PID}"
