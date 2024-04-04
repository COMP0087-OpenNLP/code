#!/bin/bash

if [ "$#" -ne 1 ]; then
  echo "Usage: $0 \"please provide the name of the envirobment\""
  exit 1
fi

python3 -m venv $1
source $1/bin/activate

echo "Activated the environment: $1"
which python3
python3 --version

# Setting cache
current_dir=$(pwd)/$1
mkdir -p $current_dir/pip_cache
export PIP_CACHE_DIR=$current_dir/pip_cache

# Install the required packages
echo "Installing the required packages..."
python3 -m pip install --upgrade pip
python3 -m pip install numpy pandas torch jupyter certifi scipy

echo "Installing the mteb package..."
cd pytrec_eval
python3 setup.py install
cd ..
python3 -m pip install mteb==1.1.2
python3 -m pip install mteb[beir]
