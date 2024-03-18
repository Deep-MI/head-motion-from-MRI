#!/bin/bash

cd /workspace
export PYTHONPATH=/workspace
python3 scripts/eval.py -i $1 -o $2 -type $3
