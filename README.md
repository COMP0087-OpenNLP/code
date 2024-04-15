# Code Repository

COMP0087: Statistical Natural Language Processing - OpenNLP

## Cloning the Repository

To clone this repository run the following command:

```bash
git clone https://github.com/COMP0087-OpenNLP/code --depth 1
```

### Updating Data Locally

Run the following command to update the data locally:

```bash
bash update_data.sh
```

If the data is not present locally this will clone the necessary repos.

Otherwise, if the data is already present locally, this command will pull the latest version from each sub-repo. Also, it will reduce the size of the respective `.git` files.

### Adding A New Cached Model

Add the GitHub URL to the `update_data.sh` file.
