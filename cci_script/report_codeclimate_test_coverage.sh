#!/bin/bash
set -e

pip install --upgrade codeclimate-test-reporter
CODECLIMATE_REPO_TOKEN="${CODECLIMATE_REPO_TOKEN}" codeclimate-test-reporter
