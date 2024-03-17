from sentence_transformers import SentenceTransformer

class MyModelFlagEmbedding():
  def __init__(self):
    self.model_name = "flag-embedding"
    self.flag_embedding = SentenceTransformer('BAAI/bge-large-en-v1.5')

  def encode(self, sentences, **kwargs):
      """
      Returns a list of embeddings for the given sentences.
      Args:
          sentences (`List[str]`): List of sentences to encode
          batch_size (`int`): Batch size for the encoding

      Returns:
          `List[np.ndarray]` or `List[tensor]`: List of embeddings for the given sentences
      """
      return list(self.flag_embedding.encode(sentences))