#!/bin/bash
# This script runs the initialization of tensorflow and theano backend separately and write the result to files.
# Then check if the difference between the results are within threshold
set -eu

CONFIG=$1
if [[ ! -f "${CONFIG}" ]]; then
    echo "Argument must be a YAML file"
    exit 1
fi

if [ "${COUNT_INTEGRATION_COVERAGE:-false}" = true ]; then
    TEST_COMMAND="coverage run --parallel-mode"
else
    TEST_COMMAND="python"
fi
BASE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
TEST_COMMAND="${TEST_COMMAND} ${BASE_DIR}/run_initializer.py ${CONFIG}"

CONFIG_NAME=$(basename ${CONFIG%.*})
echo "********************************************************************************"
echo "***** Checking numerical compatibility of ${CONFIG_NAME}"
echo ""
cat ${CONFIG}
echo ""

RETURN=0
echo "* Running ${CONFIG_NAME} with Theano backend"
LUCHADOR_NN_BACKEND=theano     LUCHADOR_NN_CONV_FORMAT=NCHW ${TEST_COMMAND}
if [[ ! $? = 0 ]]; then RETURN=1; fi
echo ""

echo "* Running ${CONFIG_NAME} with Tensorflow backend"
LUCHADOR_NN_BACKEND=tensorflow LUCHADOR_NN_CONV_FORMAT=NHWC ${TEST_COMMAND}
if [[ ! $? = 0 ]]; then RETURN=1; fi
echo "********************************************************************************"
echo ""

exit ${RETURN}
