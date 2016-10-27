#!/bin/bash
# This script runs the layer IO of tensorflow and theano backend separately and write the result to files.
# Then check if the difference between the results are within threshold.
set -eu

CONFIG=${1}
if [[ ! -f "${CONFIG}" ]]; then
    echo "Argument must be YAML file"
    exit 1
fi

COUNT_INTEGRATION_COVERAGE=${COUNT_INTEGRATION_COVERAGE:-false}
if [ "${COUNT_INTEGRATION_COVERAGE}" = true ]; then
    TEST_COMMAND="coverage run --source luchador -a"
else
    TEST_COMMAND="python"
fi

LAYER_NAME="$( basename ${CONFIG%.*} )"
BASE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
TEST_COMMAND="${TEST_COMMAND} ${BASE_DIR}/run_layer.py ${CONFIG}"
COMPARE_COMMAND="python ${BASE_DIR}/compare_result.py"

FILE1="tmp/test_layer_numerical_compatibility/${LAYER_NAME}/theano.h5"
FILE2="tmp/test_layer_numerical_compatibility/${LAYER_NAME}/tensorflow.h5"

echo "*** Checking numerical compatibility of ${LAYER_NAME} ***"
echo ""
cat ${CONFIG}
echo ""
echo "* Running ${LAYER_NAME} with Theano backend"
LUCHADOR_NN_BACKEND=theano     LUCHADOR_NN_CONV_FORMAT=NCHW ${TEST_COMMAND} --output ${FILE1}
echo "* Running ${LAYER_NAME} with Tensorflow backend"
LUCHADOR_NN_BACKEND=tensorflow LUCHADOR_NN_CONV_FORMAT=NHWC ${TEST_COMMAND} --output ${FILE2}
echo "* Comparing results"
${COMPARE_COMMAND} ${FILE1} ${FILE2}
echo ""
