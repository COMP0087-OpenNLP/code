from abc import ABC, abstractmethod
import datasets
import numpy as np
import time

BASIC_MODELS = [
    "angle",
    "cohere",
    "flag-embedding",
    "gist",
    "gte-large",
    "llmrails",
    "mixed-bread",
    "voyage",
]

class AbstractModel(ABC):
    
    def __init__(self, model_name: str, task_name: str): # The task name is needed for caching
        self.model_name = model_name
        self.task_name = task_name

    @abstractmethod
    def encode(self, sentences, batch_size=32, **kwargs):
        """
        Returns a list of embeddings for the given sentences.

        NOTE: The vectors should be normalized to unit length (L2 norm)

        Args:
            sentences (`List[str]`): List of sentences to encode
            batch_size (`int`): Batch size for the encoding

        Returns:
            `List[np.ndarray]` or `List[tensor]`: List of embeddings for the given sentences
        """
        pass


class LocallyCachedModel(AbstractModel):
    def __init__(self, model_name: str, task_name: str):
        super().__init__(model_name, task_name)
        print(f"Loading {model_name} from cache for {task_name}...")
        main: datasets.Dataset = datasets.load_from_disk(f"data/sentences/{self.task_name}")
        embeddings: datasets.Dataset = datasets.load_from_disk(f"data/{model_name}/{self.task_name}")

        ds = datasets.concatenate_datasets([main, embeddings], axis=1)
        self.df = ds.to_pandas().drop_duplicates(subset=["text"]).set_index("text")
    
    def encode(self, sentences, batch_size=32, **kwargs):
        # As efficient as possible
        embeddings = self.df.loc[sentences]["embeddings"].values

        # normalize embeddings to unit length (L2 norm)
        embeddings = [e / np.linalg.norm(e) for e in embeddings]

        return embeddings

class StackedModel(AbstractModel):
    def __init__(self, model1: AbstractModel, model2: AbstractModel, task_name: str):
        super().__init__(model_name=f"{model1.mode_name}${model2.mode_name}", task_name=task_name)
        self.model_1 = model1
        self.model_2 = model2
  
    def encode(self, sentences, batch_size=32, **kwargs):
            emb1 = self.model_1.encode(sentences, batch_size)
            emb2 = self.model_2.encode(sentences, batch_size)
            concatenation = np.concatenate((emb1, emb2), axis=1)
            return list(concatenation)

class PCAModel(AbstractModel):
    def __init__(self, model_name: str, task_name: str):
        super().__init__(model_name, task_name)
        main: datasets.Dataset = datasets.load_from_disk(f"data_pca/sentences/{self.task_name}")
        embeddings: datasets.Dataset = datasets.load_from_disk(f"data_pca/{model_name}/{self.task_name}")
        
        ds = datasets.concatenate_datasets([main, embeddings], axis=1)
        self.df = ds.to_pandas().drop_duplicates(subset=["text"]).set_index("text")

    def encode(self, sentences, batch_size=32, **kwargs):
        embeddings = self.df.loc[sentences]["embeddings"].values
        return np.vstack(embeddings)
  
def create_stacked_model(models, task_name):
    if len(models) == 1:
        return LocallyCachedModel(models[0], task_name)
    
    # Recursive case: Divide the models list into two halves and create stacked models.
    mid_point = len(models) // 2
    model1 = create_stacked_model(models[:mid_point], task_name)
    model2 = create_stacked_model(models[mid_point:], task_name)
    stacked_model = StackedModel(model1, model2, task_name)
    
    # Save a reference to the original encode method
    original_encode = stacked_model.encode
    
    def normalize_encode(sentences, batch_size=32, **kwargs):
        embeddings = original_encode(sentences, batch_size, **kwargs)
        
        # normalize embeddings to unit length (L2 norm)
        embeddings = [e / np.linalg.norm(e) for e in embeddings]
        return embeddings
    
    stacked_model.encode = normalize_encode
    return stacked_model

def model_factory(model_name, task_name):
    print(f"Creating model {model_name} for task {task_name}")
    if model_name in BASIC_MODELS:
        return LocallyCachedModel(model_name, task_name)
    elif "$" in model_name and "-pca" in model_name:
        model_name = model_name.replace("-pca", "")
        return PCAModel(model_name, task_name)
    elif "$" in model_name:
        models_names = model_name.split("$")
        return create_stacked_model(models_names, task_name)
    else:
        raise ValueError(f"Model {model_name} not found")