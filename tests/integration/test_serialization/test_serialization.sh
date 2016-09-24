#!/bin/bash
# This script checks if Model + Optimizer is serialized in the way training can be resumed.
#
# Arguments:
# --model: Name of model to tests [de]serialization.
# --optimizer: Name of optimizer to test [de]serialization.

set -e

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

if [[ -z "${LUCHADOR_NN_BACKEND}" ]]; then
    echo "Environmental variable LUCHADOR_NN_BACKEND must be set"
    exit 1
fi
    
OPTIMIZER_FILENAME=$(basename ${OPTIMIZER})

OUTPUT_DIR="tmp/test_serialization_${MODEL}_${OPTIMIZER_FILENAME%.*}_${LUCHADOR_NN_BACKEND}"
INPUT_FILE="${OUTPUT_DIR}/save_0.h5"

BASE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
TEST_COMMAND="python ${BASE_DIR}/serialize_model.py ${MODEL} ${OPTIMIZER}"

echo "* Serializing ${MODEL} + $( basename "${OPTIMIZER}")"
${TEST_COMMAND} --output ${OUTPUT_DIR}
echo "* Deserializing ${MODEL} + $( basename "${OPTIMIZER}")"
${TEST_COMMAND} --input ${INPUT_FILE}
