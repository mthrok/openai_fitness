#!/bin/bash
set -eu

coverage combine
coverage xml
pip install --upgrade codacy-coverage
python-codacy-coverage -r coverage.xml
