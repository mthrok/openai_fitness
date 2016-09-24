#!/bin/bash
set -e

BASE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
TEST_DIR="${BASE_DIR}/test_optimizer_numerical_compatibility"

# Get the list of optimizers in optimizer directory
OPTIMIZERS=()
for FILE in ${BASE_DIR}/optimizer/*.yml
do
    OPTIMIZERS+=(${FILE})
done


FORMULAE="$(python ${TEST_DIR}/formula.py)"
for FORMULA in ${FORMULAE}
do
    for OPTIMIZER in "${OPTIMIZERS[@]}"
    do
        ${TEST_DIR}/test_optimizer_numerical_compatibility.sh --optimizer ${OPTIMIZER} --formula ${FORMULA}
    done
done
