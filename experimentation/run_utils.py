from typing import List, Optional
import numpy as np
import itertools
import os
from tqdm import tqdm
from mteb import MTEB

from results_to_csv import main as convert_to_csv
from model_factory import BASIC_MODELS, model_factory

TASK_LIST_CLASSIFICATION = [
    "AmazonCounterfactualClassification",
    # "AmazonPolarityClassification",
    # "AmazonReviewsClassification",
    "Banking77Classification",
    "EmotionClassification",
    # "ImdbClassification",
    # "MassiveIntentClassification",
    # "MassiveScenarioClassification",
    # "MTOPDomainClassification",
    # "MTOPIntentClassification",
    # "ToxicConversationsClassification",
    # "TweetSentimentExtractionClassification",
]

TASK_LIST_CLUSTERING = [
    # "ArxivClusteringP2P",
    "ArxivClusteringS2S",
    # "BiorxivClusteringP2P",
    # "BiorxivClusteringS2S",
    # "MedrxivClusteringP2P",
    # "MedrxivClusteringS2S",
    "RedditClustering",
    # "RedditClusteringP2P",
    # "StackExchangeClustering",
    # "StackExchangeClusteringP2P",
    # "TwentyNewsgroupsClustering",
]

TASK_LIST_PAIR_CLASSIFICATION = [
    # "SprintDuplicateQuestions",
    "TwitterSemEval2015",
    # "TwitterURLCorpus",
]

TASK_LIST_RERANKING = [
    "AskUbuntuDupQuestions",
#     "MindSmallReranking",
#     "SciDocsRR",
#     "StackOverflowDupQuestions",
]

TASK_LIST_RETRIEVAL = [
    "ArguAna",
    # "ClimateFEVER",
    # "CQADupstackAndroidRetrieval",
    # "CQADupstackEnglishRetrieval",
    # "CQADupstackGamingRetrieval",
    # "CQADupstackGisRetrieval",
    # "CQADupstackMathematicaRetrieval",
    # "CQADupstackPhysicsRetrieval",
    # "CQADupstackProgrammersRetrieval",
    # "CQADupstackStatsRetrieval",
    # "CQADupstackTexRetrieval",
    # "CQADupstackUnixRetrieval",
    # "CQADupstackWebmastersRetrieval",
    # "CQADupstackWordpressRetrieval",
    # "DBPedia",
    # "FEVER",
    # "FiQA2018",
    # "HotpotQA",
    # "MSMARCO",
    # "NFCorpus",
    # "NQ",
    # "QuoraRetrieval",
    # "SCIDOCS",
    "SciFact",
    # "Touche2020",
    # "TRECCOVID",
]

TASK_LIST_STS = [
    "SICK-R",
    "STS12",
    "STS13",
    "STS14",
    "STS15",
    "STS16",
    "STS17",
    "STS22",
    "STSBenchmark",
    # "BIOSSES",
]


TASK_LIST = (
    TASK_LIST_CLASSIFICATION
    + TASK_LIST_CLUSTERING
    + TASK_LIST_PAIR_CLASSIFICATION
    + TASK_LIST_RERANKING
    + TASK_LIST_RETRIEVAL
    + TASK_LIST_STS
)

def run_on_tasks(model_name):
    print(f"Evaluating the model {model_name}...")
    for task in TASK_LIST:
        # skip if task has already been evaluated
        if os.path.exists(f"results/{model_name}/{task}.json"):
            print(f"Skipping {task} as it already exists")
            continue

        model = model_factory(model_name, task)
        evaluation = MTEB(tasks=[task], task_langs=["en"])  # Remove "en" for running all languages
        evaluation.run(model, output_folder=f"results/{model_name}", eval_splits=["test"]) # evaluating only on test set

    if os.path.exists(f"results/{model_name}"):
        print("Converting the results to a CSV file...")
        convert_to_csv(f"results/{model_name}")

    print("--DONE--")