#!/bin/bash

# Function to clone or pull a repository
clone_or_pull() {
    repo_url=$1
    folder_name=$(basename "$repo_url" .git)

    if [ -d "$folder_name" ]; then
        echo "Updating repository: $folder_name"
        cd "$folder_name"
        git checkout master

        # Deletes the reflog (to save space)
        git reflog expire --expire=now --all
        git gc --aggressive --prune=now

        # Update with latest
        git pull

        # Deletes the reflog (to save space)
        git reflog expire --expire=now --all
        git gc --aggressive --prune=now

        cd ..
    else
        echo "Cloning repository: $folder_name"
        git clone "$repo_url"
    fi
}

# Set the target folder
target_folder="experimentation/data"

# Create the target folder if it doesn't exist
mkdir -p "$target_folder"
cd "$target_folder"

# List of repositories
repositories=(
    https://github.com/COMP0087-OpenNLP/voyage
    https://github.com/COMP0087-OpenNLP/cohere
    https://github.com/COMP0087-OpenNLP/sentences
    https://github.com/COMP0087-OpenNLP/llmrails
    https://github.com/COMP0087-OpenNLP/gist
    https://github.com/COMP0087-OpenNLP/angle
    https://github.com/COMP0087-OpenNLP/gte-large
)

# Clone or pull the repositories
for repo in "${repositories[@]}"; do
    clone_or_pull "$repo"
done