#!/bin/bash
# This script checks if Model + Optimizer is serialized in the way training can be resumed.
#
# Arguments:
# --model: Name of model to tests [de]serialization.
# --optimizer: Name of optimizer to test [de]serialization.

set -eu

while [[ $# -gt 0 ]]
do
    key="$1"
    case $key in
        --optimizer)
            OPTIMIZER="$2"
            shift
            ;;
        --model)
            MODEL="$2"
            shift
            ;;
        *)
            echo "Unexpected option ${key} was given"
            exit 1
            ;;
    esac
    shift
done

if [[ -z "${MODEL}" || -z "${OPTIMIZER}" ]]; then
    echo "--model and --optimizer must be given"
    exit 1
fi

if [ "${COUNT_INTEGRATION_COVERAGE:-false}" = true ]; then
    TEST_COMMAND="coverage run --parallel-mode"
else
    TEST_COMMAND="python"
fi
BASE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
TEST_COMMAND="${TEST_COMMAND} ${BASE_DIR}/serialize_model.py ${MODEL} ${OPTIMIZER}"

OPTIMIZER_NAME=$(basename ${OPTIMIZER%.*})
MODEL_NAME=$(basename ${MODEL%.*})

BACKENDS=( "theano" "tensorflow" )
CONV_FORMATS=( "NCHW" "NHWC" )
BASE_OUTPUT_DIR="tmp/test_serialization/${MODEL_NAME}_${OPTIMIZER_NAME}"

echo "*** Checking serialization compatiblity of ${MODEL_NAME} + ${OPTIMIZER_NAME}"
for i in {0..1}
do
    BACKEND="${BACKENDS[${i}]}"
    CONV_FORMAT="${CONV_FORMATS[${i}]}"
    OUTPUT_DIR="${BASE_OUTPUT_DIR}/${BACKEND}"

    echo "* Serializing on ${BACKEND} ${CONV_FORMAT}"
    LUCHADOR_NN_BACKEND=${BACKEND} LUCHADOR_NN_CONV_FORMAT=${CONV_FORMAT} ${TEST_COMMAND} --output ${OUTPUT_DIR}
done

THEANO_PARAM="${BASE_OUTPUT_DIR}/theano/save_0.h5"
TENSORFLOW_PARAM="${BASE_OUTPUT_DIR}/tensorflow/save_0.h5"

echo "* Deserializing Theano param on Theano backend"
LUCHADOR_NN_BACKEND=theano LUCHADOR_NN_CONV_FORMAT=NCHW ${TEST_COMMAND} --input ${THEANO_PARAM}
echo "* Deserializing Tensorflow param on Theano backend"
LUCHADOR_NN_BACKEND=theano LUCHADOR_NN_CONV_FORMAT=NCHW ${TEST_COMMAND} --input ${TENSORFLOW_PARAM}

echo "* Deserializing Theano param on Tensorflow backend"
LUCHADOR_NN_BACKEND=tensorflow LUCHADOR_NN_CONV_FORMAT=NHWC ${TEST_COMMAND} --input ${THEANO_PARAM}
echo "* Deserializing Tensorflow param on Tensorflow backend"
LUCHADOR_NN_BACKEND=tensorflow LUCHADOR_NN_CONV_FORMAT=NHWC ${TEST_COMMAND} --input ${TENSORFLOW_PARAM}
echo ""
