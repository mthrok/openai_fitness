#!/bin/bash
set -eu

if [[ -n "${CODACY_PROJECT_TOKEN}" ]]; then
    coverage combine
    coverage xml
    python-codacy-coverage -r coverage.xml
fi
