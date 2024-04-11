#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import itertools
import datasets
import pandas as pd
import os
from tqdm import tqdm
from abc import ABC, abstractmethod
from results_to_csv import main as convert_to_csv
from mteb import MTEB
from model_factory import model_factory
from sklearn.decomposition import PCA
import pickle as pk


# In[ ]:


import logging

def setup_logger(name, log_file, level=logging.INFO):
    """Function to setup a logger for a given name and file."""
    if not os.path.exists(os.path.dirname(log_file)):
        os.makedirs(os.path.dirname(log_file))
        
    logger = logging.getLogger(name)
    if not logger.handlers:  # Only add handler if there are no existing handlers
        handler = logging.FileHandler(log_file)    
        logger.setLevel(level)
        logger.addHandler(handler)
    
    return logger


# ## Model List

# In[ ]:


BASIC_MODELS = os.listdir('data')
BASIC_MODELS.remove("sentences")


# ## Task List

# In[ ]:


from run_utils import TASK_LIST


# In[ ]:


ndims_list = [256, 512, 768, 896, 1024]


# ## Generating PCA Functions

# In[ ]:


all_sentences = {}

for dataset in TASK_LIST:
    mteb = MTEB(tasks = [dataset], task_langs=["en"], trust_remote_code=True)
    task = mteb.tasks[0]
    task.load_data()
    if task.dataset is None:
        print(f"{dataset} has no dataset. Skipping...")
        continue
    print(f"{dataset} has these splits: {task.dataset.keys()}")

    train = False
    if ("train" in task.dataset.keys()):
        train = True
        
    validation = False
    if ("validation" in task.dataset.keys()):
        validation = True

    if not train and not validation:
        print(f"Skipping {dataset}..")
        continue
    
    all_sentences[dataset] = []
    if train:
        if "text" in task.dataset["train"].column_names: 
            all_sentences[dataset] += task.dataset["train"]["text"]
        if "sentence1" in task.dataset["train"].column_names:
            all_sentences[dataset] += task.dataset["train"]["sentence1"]
        if "sentence2" in task.dataset["train"].column_names:
            all_sentences[dataset] += task.dataset["train"]["sentence2"]
    if validation:
        if "text" in task.dataset["validation"].column_names: 
            all_sentences[dataset] += task.dataset["validation"]["text"]
        if "sentence1" in task.dataset["validation"].column_names:
            all_sentences[dataset] += task.dataset["validation"]["sentence1"]
        if "sentence2" in task.dataset["validation"].column_names:
            all_sentences[dataset] += task.dataset["validation"]["sentence2"]


# In[ ]:


script_logger = setup_logger("Generating PCA", f"pca/logs.txt")

for ndims in ndims_list:
    script_logger.info(f"\n\nStarting PCA for {ndims} dimensions")
    
    # creating directory
    path = f"pca/{ndims}"
    if not os.path.exists(path):
        os.makedirs(path)
    
    # generating PCA for all combinations of models
    for r in range(1, len(BASIC_MODELS) + 1):
        combinations_object = itertools.combinations(BASIC_MODELS, r)
        combinations_list = [sorted(list(combination)) for combination in combinations_object]
        
        # generating PCA for each combination
        for combination in combinations_list:
            model_name = "$".join(combination)
            script_logger.info(f"<--> Starting PCA for {model_name}!")
            
            # if model alr exists
            if os.path.exists(f"{path}/{model_name}.pkl"):
                script_logger.info(f"PCA for {model_name} already exists. Skipping...")
                continue
            
            # retrieving embeddings
            all_embeddings = []
            for task in all_sentences.keys():
                model = model_factory(model_name, task)
                embeddings = model.encode(all_sentences[task])
                all_embeddings += embeddings
            script_logger.info(f"Retrieved embeddings for {model_name}")
            
            # PCA
            script_logger.info(f"Fitting PCA for {model_name}")
            pca = PCA(n_components= ndims)
            pca.fit_transform(all_embeddings)
            
            # Saving PCA
            script_logger.info(f"Saving PCA for {model_name}")
            pk.dump(pca, open(f"{path}/{model_name}.pkl", "wb"))


# ## Evaluating Stacked Model with PCA

# In[ ]:


def evaluate_model(model_name: str, ndims: int):
    script_logger = setup_logger(f"eval_models_{ndims}", f"results_pca/{ndims}/logs.txt")
    
    for task in TASK_LIST:
        # if model has already been evaluated
        if os.path.exists(f"results_pca/{ndims}/{model_name}/{task}.json"):
            script_logger.info(f"{model_name} has already been evaluated for {task}. Skipping...")
            continue
        
        # get stacked model
        model = model_factory(model_name + "-pca", task, ndims= ndims)
        
        script_logger.info(f"Evaluating {model_name} on {task}")
        evaluation = MTEB(tasks = [task], task_langs=["en"])
        evaluation.run(model, output_folder=f"results_pca/{ndims}/{model_name}", eval_splits=["test"])

    if os.path.exists(f"results_pca/{ndims}/{model_name}_results.csv"):
        script_logger.info(f"Results for {model_name} already converted to csv. Skipping...")
    elif os.path.exists(f"results_pca/{ndims}/{model_name}"):
        script_logger.info(f"Converting results to csv for {model_name}")
        convert_to_csv(f"results_pca/{ndims}/{model_name}")
    else:
        script_logger.info(f"No results found for {model_name}. Skipping...")
        
    script_logger.info(f"Finished evaluating {model_name} for all tasks\n\n")


# In[ ]:


def evaluate_models(ndims: int):
    script_logger = setup_logger(f"eval_models_{ndims}", f"results_pca/{ndims}/logs.txt")
    for r in range(1, len(BASIC_MODELS) + 1):
        combinations_object = itertools.combinations(BASIC_MODELS, r)
        combinations_list = [sorted(list(combination)) for combination in combinations_object]
        
        for combination in combinations_list:
            model_name = "$".join(combination)
            script_logger.info(f"Starting evaluation for {model_name}")
            evaluate_model(model_name, ndims)


# In[ ]:


for ndims in ndims_list:
    evaluate_models(ndims)

