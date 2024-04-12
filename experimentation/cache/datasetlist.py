import datasets
import dformats
import os
from mteb import *
from tqdm.auto import tqdm

TASK_LIST_STS = [
    "BIOSSES",
    "SICK-R",
    "STS12",
    "STS13",
    "STS14",
    "STS15",
    "STS16",
    "STS17",
    "STS22",
    "STSBenchmark",
    "SummEval",
]


# Task list ordered by size:
TASK_LIST = [
    "MassiveIntentClassification",
    "MassiveScenarioClassification",
    "Banking77Classification",
    "MTOPDomainClassification",
    "MTOPIntentClassification",
    "TwitterSemEval2015",
    "EmotionClassification",
    "TwentyNewsgroupsClustering",
    "TweetSentimentExtractionClassification",
    "MedrxivClusteringS2S",
    "BIOSSES",
    "SciFact",
    "BiorxivClusteringS2S",
    "CQADupstackWebmastersRetrieval",
    "STS22",
    "CQADupstackAndroidRetrieval",
    "STS17",
    "SciDocsRR",
    "CQADupstackEnglishRetrieval",
    "CQADupstackMathematicaRetrieval",
    "CQADupstackGamingRetrieval",
    "SprintDuplicateQuestions",
    "RedditClustering",
    "ToxicConversationsClassification",
    "CQADupstackPhysicsRetrieval",
    "SCIDOCS",
    "CQADupstackProgrammersRetrieval",
    # "QuoraRetrieval",
    "StackOverflowDupQuestions",
    "CQADupstackGisRetrieval",
    "AmazonReviewsClassification",
    "CQADupstackStatsRetrieval",
    "FiQA2018",
    "CQADupstackUnixRetrieval",
    "StackExchangeClustering",
    "CQADupstackWordpressRetrieval",
    "ArxivClusteringS2S",
    "ImdbClassification",
    "MedrxivClusteringP2P",
    "StackExchangeClusteringP2P",
    "STS16",
    "CQADupstackTexRetrieval",
    "STS13",
    "BiorxivClusteringP2P",
    "NFCorpus",
    "AmazonCounterfactualClassification",
    # "SummEval",
    "TRECCOVID",
    "STS15",
    "STS14",
    "AskUbuntuDupQuestions",
    "RedditClusteringP2P",
    "ArguAna",
    "STS12",
    "STSBenchmark",
    "Touche2020",
    "SICK-R",
    # "MindSmallReranking",
    # "NQ",
    # "DBPedia",
    # "HotpotQA",
    # "AmazonPolarityClassification",
    # "ClimateFEVER",
    # "FEVER",
    # "MSMARCO",
    # "TwitterURLCorpus",
    # "ArxivClusteringP2P"
]


# Helpers
batch_generator = lambda lst, batch_size: (lst[i:i + batch_size] for i in range(0, len(lst), batch_size)) if batch_size is not None else [lst]


def get_embedding(dataset, sentence, model_name, text_column="text"):
    """
    Given a dataset and the model name, this function returns
    the embedding of the sentence as a numpy array
    """
    return dataset.filter(lambda x: x[text_column] == sentence)[model_name]



def download() -> None:

    for t in TASK_LIST:
        path = f"../data/sentences/{t}"
        if os.path.exists(path):
            print(f"{t} already downloaded, skipping...")
            continue

        print(f"Downloading {t}...")
        mteb = MTEB(tasks=[t], task_langs=["en"])
        task = mteb.tasks[0]
        task.load_data()
        if isinstance(task, AbsTaskClassification):
            dataset = dformats.classification(task)
        elif isinstance(task, AbsTaskClustering):
            dataset = dformats.clustering(task)
        elif isinstance(task, AbsTaskPairClassification):
            dataset = dformats.pair_classification(task)
        elif isinstance(task, AbsTaskReranking):
            dataset = dformats.reranking(task)
        elif isinstance(task, AbsTaskRetrieval):
            dataset = dformats.retrieval(task)
        elif isinstance(task, AbsTaskSTS):
            dataset = dformats.sts(task)

        dataset.save_to_disk(path, max_shard_size="75MB")

def encode_wrapper(model, text: str, max_retries: int = 10, wait_time: int = 60):
    while max_retries > 0:
            try:
                result = model.encode(text)
            except Exception as e:
                print(f"Error: {e}")
                print(f"Retrying in {wait_time} seconds...")
                max_retries -= 1
                import time
                time.sleep(wait_time)
            else:
                break
    return result

def cache_embeddings(model_name: str, model, batch_size: int = None):
    """
    `model` can be anything that has an `encode` method
    """
    for task in TASK_LIST:
        download_path = f"../data/sentences/{task}"
        embedding_path = f"../data/{model_name}/{task}"
        if not os.path.exists(download_path):
            print(f"Skipping {task} as it is not downloaded yet...")
            continue
        if os.path.exists(embedding_path):
            print(f"{task} already cached for {model_name}, skipping...")
            continue
        print(f"Caching {task}...")


        dataset = datasets.load_from_disk(download_path)
        print("Loaded dataset")
        
        embeddings = []
        for batch in tqdm(batch_generator(dataset, batch_size), desc="Caching embeddings", total = round(len(dataset) / batch_size)):
            embeddings.extend(encode_wrapper(model, batch["text"]))

        dataset = datasets.Dataset.from_dict({"embeddings": embeddings})
        dataset.save_to_disk(embedding_path, max_shard_size="75MB")

def cache_STS_embeddings(model_name: str, model, batch_size: int = None):
    """
    `model` can be anything that has an `encode` method
    """
    for task in TASK_LIST_STS:
        download_path = f"../data/sentences/{task}"
        embedding_path = f"../data/{model_name}/{task}"
        if not os.path.exists(download_path):
            print(f"Skipping {task} as it is not downloaded yet...")
            continue
        if os.path.exists(embedding_path):
            print(f"{task} already cached for {model_name}, skipping...")
            continue
        print(f"Caching {task}...")


        dataset = datasets.load_from_disk(download_path)
        print("Loaded dataset")
        
        embeddings = []
        for batch in tqdm(batch_generator(dataset, batch_size), desc="Caching embeddings", total = round(len(dataset) / batch_size)):
            embeddings.extend(encode_wrapper(model, batch["text"]))

        dataset = datasets.Dataset.from_dict({"embeddings": embeddings})
        dataset.save_to_disk(embedding_path, max_shard_size="75MB")