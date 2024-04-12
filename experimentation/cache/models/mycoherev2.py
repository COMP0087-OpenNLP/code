import cohere
import numpy as np
co = cohere.Client('hafapiCtIsGxFMc1FR0N3TJiGc0CbxDvog54iDfy') # This is your trial API key

def get_embedding_cohere(list_of_text, model):
  response = co.embed(
    model=model,
    texts=list_of_text,
    input_type='search_document'
  )
  return [np.array(emb) for emb in response.embeddings]

class MyModelCohereV2():
  def __init__(self):
     self.model_name = 'cohere-large'

  def encode(self, sentences, batch_size=32, **kwargs):
      """
      Returns a list of embeddings for the given sentences.
      Args:
          sentences (`List[str]`): List of sentences to encode
          batch_size (`int`): Batch size for the encoding

      Returns:
          `List[np.ndarray]` or `Listz[tensor]`: List of embeddings for the given sentences
      """
      return get_embedding_cohere(sentences, 'embed-english-v2.0')