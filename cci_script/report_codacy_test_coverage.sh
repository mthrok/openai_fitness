#!/bin/bash
set -eu

pip install --upgrade codacy-coverage
coverage xml
python-codacy-coverage -r coverage.xml
