#!/bin/bash
set -eu

BASE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
TEST_DIR="${BASE_DIR}/test_serialization"

LUCHADOR_DIR="$( ${BASE_DIR}/get_luchador_dir.py)"

# Get the list of optimizers in optimizer directory
OPTIMIZERS=()
for FILE in ${BASE_DIR}/data/optimizer/*.yml
do
    OPTIMIZERS+=(${FILE})
done

# Get the list of models in luchador model directory
MODELS=()
for FILE in ${LUCHADOR_DIR}/nn/data/*.yml
do
    MODELS+=( $(basename ${FILE%.*}))
done

# Run serialization test on each model + optimizer combination
for MODEL in ${MODELS}
do
    for OPTIMIZER in "${OPTIMIZERS[@]}"
    do
        "${TEST_DIR}/test_serialization.sh" --model "${MODEL}" --optimizer "${OPTIMIZER}"
    done
done
