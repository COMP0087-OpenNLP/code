from abc import ABC, abstractmethod
import datasets
import numpy as np

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
        self.mode_name = model_name
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
        self.num_models  = 1
    
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
        self.num_models = model1.num_models  + model2.num_models 
  
    def encode(self, sentences, batch_size=32, **kwargs):
            emb1 = self.model_1.encode(sentences, batch_size)
            emb2 = self.model_2.encode(sentences, batch_size)
            concatenation = np.concatenate((emb1, emb2), axis=1)
            return list(concatenation)
  
def create_stacked_model(models, task_name):
    if len(models) == 1:
        return LocallyCachedModel(models[0], task_name)
    
    stacked_model = stacked_model = StackedModel(create_stacked_model(models[0:1], task_name), create_stacked_model(models[1:], task_name), task_name)
    
    def normalize_encode(sentences, batch_size=32, **kwargs):
        embeddings = stacked_model.encode(sentences, batch_size, **kwargs)
        # Normalize the final stacked embeddings by dividing by the number of models
        num_models = stacked_model.num_models
        embeddings = [e / num_models for e in embeddings]
        return embeddings
    
    stacked_model.encode = normalize_encode
    return stacked_model

def model_factory(model_name, task_name):
    if model_name in BASIC_MODELS:
        return LocallyCachedModel(model_name, task_name)
    elif "$" in model_name:
        models_names = model_name.split("$")
        return create_stacked_model(models_names, task_name)
    else:
        raise ValueError(f"Model {model_name} not found")