# ! pip install angle-emb
from angle_emb import AnglE
angle = AnglE.from_pretrained('WhereIsAI/UAE-Large-V1', pooling_strategy='cls').cuda()

class MyModelAngle():
  def __init__(self):
    self.model_name = "angle"

  def encode(self, sentences, **kwargs):
      """
      Returns a list of embeddings for the given sentences.
      Args:
          sentences (`List[str]`): List of sentences to encode
          batch_size (`int`): Batch size for the encoding

      Returns:
          `List[np.ndarray]` or `List[tensor]`: List of embeddings for the given sentences
      """

      return angle.encode(sentences, to_numpy=True)