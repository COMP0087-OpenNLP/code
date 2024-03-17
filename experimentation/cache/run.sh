#!/bin/bash

# Check if a model was provided
if [ "$#" -ne 1 ]; then
  echo "Usage: $0 \"please provide the model name to generate embeddings\""
  exit 1
fi

# Where to put logs output
result_dir=temp/$1

mkdir -p $result_dir

python3 automate_run.py --model $1 &> $result_dir/logs.txt 2>&1