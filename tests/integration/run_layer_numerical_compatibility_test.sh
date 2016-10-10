#!/bin/bash
set -eu

BASE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
TEST_DIR="${BASE_DIR}/test_layer_numerical_compatibility"
TEST_COMMAND="${TEST_DIR}/test_layer_numerical_compatibility.sh"

for DIR in ${BASE_DIR}/data/layer/*
do
    if [[ -d ${DIR} ]]; then
	"${TEST_COMMAND}" --dir "${DIR}"
    fi
done
