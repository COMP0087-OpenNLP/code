import voyageai
import numpy as np
vo = voyageai.Client(api_key="pa-VCH9vZjYrUmElnBtID3MDKkz-JStn8r5PzBnapyBdXg")

def get_embeddings_voyage(list_of_text, model):
  result = vo.embed(list_of_text, model=model, truncation=True)
  return [np.array(emb) for emb in result.embeddings]

class MyModelVoyage():
  def __init__(self):
    self.model_name = "voyage"

  def encode(self, sentences, batch_size=32, **kwargs):
      """
      Returns a list of embeddings for the given sentences.
      Args:
          sentences (`List[str]`): List of sentences to encode
          batch_size (`int`): Batch size for the encoding

      Returns:
          `List[np.ndarray]` or `List[tensor]`: List of embeddings for the given sentences
      """
      return get_embeddings_voyage(sentences, "voyage-2")