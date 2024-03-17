from sentence_transformers import SentenceTransformer

class MyModellmrails():
  def __init__(self):
    self.model_name = "llmrails"
    self.llmrails_model = SentenceTransformer('llmrails/ember-v1').cuda()

  def encode(self, sentences, **kwargs):
      """
      Returns a list of embeddings for the given sentences.
      Args:
          sentences (`List[str]`): List of sentences to encode
          batch_size (`int`): Batch size for the encoding

      Returns:
          `List[np.ndarray]` or `List[tensor]`: List of embeddings for the given sentences
      """
      return list(self.llmrails_model.encode(sentences))
