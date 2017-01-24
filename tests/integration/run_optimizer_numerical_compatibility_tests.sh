#!/bin/bash
set -eux

BASE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
TEST_DIR="${BASE_DIR}/test_optimizer_numerical_compatibility"

# Get the list of optimizers in optimizer directory
OPTIMIZERS=()
for FILE in ${BASE_DIR}/data/optimizer/*.yml
do
    OPTIMIZERS+=(${FILE})
done

# Run each optimizer on formulae
RETURN=0
FORMULAE="$(python ${TEST_DIR}/formula.py)"
for FORMULA in ${FORMULAE}
do
    for OPTIMIZER in "${OPTIMIZERS[@]}"
    do
        ${TEST_DIR}/test_optimizer_numerical_compatibility.sh --optimizer ${OPTIMIZER} --formula ${FORMULA}
	if [[ ! $? = 0 ]]; then
	    RETURN=1
	fi
    done
done

exit ${RETURN}
