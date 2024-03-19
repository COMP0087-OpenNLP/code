#!/bin/bash

# List of data repositories
repositories=(
    https://github.com/COMP0087-OpenNLP/voyage
    https://github.com/COMP0087-OpenNLP/cohere
    https://github.com/COMP0087-OpenNLP/sentences
    https://github.com/COMP0087-OpenNLP/llmrails
    https://github.com/COMP0087-OpenNLP/gist
    https://github.com/COMP0087-OpenNLP/angle
    https://github.com/COMP0087-OpenNLP/gte-large
)

reduce_repo_size() {
    echo "Reducing repository size..."
    # Deletes the reflog (to save space)
    git reflog expire --expire=now --all
    git gc --prune=now
}

# Function to clone or pull a repository
clone_or_pull() {
    repo_url=$1
    folder_name=$(basename "$repo_url" .git)

    if [ -d "$folder_name" ]; then
        echo "Updating repository: $folder_name"
        cd "$folder_name"
        git checkout master

        reduce_repo_size

        # Update with latest
        git pull

        reduce_repo_size

        cd ..
    else
        echo "Cloning repository: $folder_name"
        git clone "$repo_url"

        cd "$folder_name"
        reduce_repo_size
        cd ..
    fi
}

# Set the target folder
target_folder="experimentation/data"

# Create the target folder if it doesn't exist
mkdir -p "$target_folder"
cd "$target_folder"

# Clone or pull the repositories
for repo in "${repositories[@]}"; do
    clone_or_pull "$repo"
done