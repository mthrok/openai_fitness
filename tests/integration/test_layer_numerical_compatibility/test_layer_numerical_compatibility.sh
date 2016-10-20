#!/bin/bash
# This script runs the layer IO of tensorflow and theano backend separately and write the result to files.
# Then check if the difference between the results are within threshold
#
# Arguments:
# --dir: Path to the layer configuration directory. "config.yml", "parameter.h5", "input.h5" must be present
set -eu

COUNT_INTEGRATION_COVERAGE=${COUNT_INTEGRATION_COVERAGE:-false}
BASE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

CONFIG=
while [[ $# -gt 0 ]]
do
    key="$1"
    case $key in
        --config)
	    CONFIG="$2"
            shift
            ;;
        *)
            echo "Unexpected option ${key} was given"
            exit 1
            ;;
    esac
    shift
done

if [[ -z "${CONFIG}" ]]; then
    echo "--config must be given"
    exit 1
fi

LAYER_NAME="$( basename ${CONFIG%.*} )"
if [ "${COUNT_INTEGRATION_COVERAGE}" = true ]; then
    TEST_COMMAND="coverage run --source luchador -a"
else
    TEST_COMMAND="python"
fi
TEST_COMMAND="${TEST_COMMAND} ${BASE_DIR}/run_layer.py ${CONFIG}"
COMPARE_COMMAND="python ${BASE_DIR}/compare_result.py"

FILE1="tmp/test_layer_numerical_comparitbility/${LAYER_NAME}/theano.h5"
FILE2="tmp/test_layer_numerical_comparitbility/${LAYER_NAME}/tensorflow.h5"

echo "*** Checking numerical compatibility of ${LAYER_NAME} ***"
cat ${CONFIG}
echo "* Running ${LAYER_NAME} with Theano backend"
LUCHADOR_NN_BACKEND=theano     LUCHADOR_NN_CONV_FORMAT=NCHW ${TEST_COMMAND} --output ${FILE1}
echo "* Running ${LAYER_NAME} with Tensorflow backend"
LUCHADOR_NN_BACKEND=tensorflow LUCHADOR_NN_CONV_FORMAT=NHWC ${TEST_COMMAND} --output ${FILE2}
echo "* Comparing results"
${COMPARE_COMMAND} ${FILE1} ${FILE2}
echo ""
