#!/bin/bash

# Use this script to initialize the data for a model
# bash setup.sh <MODEL_NAME> <LOCAL_PATH_TO_URL> <REPO_URL>

# Ensuring we run from the correct directory
CURRENT_DIR=$(pwd)
if [ ${CURRENT_DIR##*/} != "code" ]; then
    echo "Please run the script from 'code' directory"
    exit 1
fi

# Check for the correct number of arguments
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <MODEL_NAME> <LOCAL_PATH_TO_URL> <REPO_URL>"
    exit 1
fi

MODEL_NAME="$1"
LOCAL_PATH="$2"
REPO_URL="$3"

# Check if the local path exists
if [ ! -d "$LOCAL_PATH" ]; then
    echo "The local path '$LOCAL_PATH' does not exist."
    exit 1
fi

cd "$LOCAL_PATH" || exit

# Check if the push.sh file exists
if [ ! -f "push.sh" ]; then 
    echo "The push.sh file does not exist in $LOCAL_PATH"
    exit 1
fi

echo "Initializing data for $MODEL_NAME"

# Check if the git repository exists
# If not, create a new git repository
if [ ! -d ".git" ]; then
    echo "Creating a new git repository"
    git init
    git remote add origin "$REPO_URL"
    echo "Git repository created"
else
    echo "Git repository already exists"
fi

# Add the push.sh file to the git repository
git add push.sh
git commit -m "Initial commit"
git push -u origin master

# Upload all files in the repository
bash push.sh "$MODEL_NAME"

echo "Initialised $MODEL_NAME data"