#!/bin/bash
set -eu

BASE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
DATA_DIR="${BASE_DIR}/data/layer"
TEST_DIR="${BASE_DIR}/test_layer_numerical_compatibility"
TEST_COMMAND="${TEST_DIR}/test_layer_numerical_compatibility.sh"

for FILE in ${DATA_DIR}/*.yml
do
    "${TEST_COMMAND}" --config "${FILE}"
done
