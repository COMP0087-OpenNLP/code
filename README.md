# Code Repository

COMP0087: Statistical Natural Language Processing - OpenNLP

## Cloning the Repository

To clone this repository including all its submodules, run the following command:

```bash
git clone --recurse-submodules https://github.com/COMP0087-OpenNLP/code
```
#### Updating Submodules

If you've cloned the repository and need to update the submodules, use the following commands:

```bash
git submodule init
git submodule update
```

### Adding a New Submodule
If you need to add a new submodule to the repository, follow these steps

1. Ensure you've created and uploaded the data to the new repository to be added
2. Navigate to the local clone of the `code` repository
3. Run the following code from terminal

```bash
git submodule add <repository-URL> experimentation/data/<new_repo>
```
4. Commit the changes and push
```bash
git add .
git commit -m "Added new submodule: [Submodule name]
git push
```

### Updating submodule to latest commits
If you've changed contents of a data repository, then run the following command in the `code` repository:
```bash
git submodule update --remote
```
