from sentence_transformers import SentenceTransformer

class MyModelGist():
    def __init__(self):
        self.model_name = "gist"
        self.gist_model = SentenceTransformer("avsolatorio/GIST-large-Embedding-v0", revision=None).cuda()
    
    def encode(self, sentences, **kwargs):
        """
        Returns a list of embeddings for the given sentences.
        Args:
            sentences (`List[str]`): List of sentences to encode
            batch_size (`int`): Batch size for the encoding
    
        Returns:
            `List[np.ndarray]` or `List[tensor]`: List of embeddings for the given sentences
        """
        return list(self.gist_model.encode(sentences))
