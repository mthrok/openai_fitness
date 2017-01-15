#!/bin/bash
set -eu

if [[ "${CIRCLE_PROJECT_USERNAME:-false}" = "mthrok" ]]; then
    coverage combine
    coverage xml
    python-codacy-coverage -r coverage.xml
fi
