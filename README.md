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
1. Create a new repo for your data
2. Run the bash script `setup.sh`
```bash
bash setup.sh <MODEL_NAME> <LOCAL_PATH_TO_URL> <REPO_URL>
```
Example: `setup.sh Angle experimentation/data/data_angle https://github.com/COMP0087-OpenNLP/data_angle

### Updating submodule to latest commits
If you've changed contents of a data repository, then run the following command in the `code` repository:
```bash
git submodule update --remote
```
