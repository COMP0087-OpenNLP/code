from openai import OpenAI
import numpy as np
from typing import List
API_KEY = "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"
client = OpenAI(api_key=API_KEY, max_retries=5)
import tiktoken
  
class MyModelOpenAI:
    """
    Benchmark OpenAIs embeddings endpoint.
    """
    def __init__(self, task_name=None, batch_size=32, save_emb=False, **kwargs):
        self.model_name = "text-embedding-3-large"
        self.engine = self.model_name
        self.max_token_len = 8191
        self.batch_size = batch_size
        self.save_emb = save_emb # Problematic as the filenames may end up being the same
        self.tokenizer = tiktoken.get_encoding('cl100k_base')
        self.task_name = task_name
        self.client = OpenAI(api_key=API_KEY)

    def encode(self, 
            sentences,
            decode=True,
            idx=None,
            **kwargs
        ):

        fin_embeddings = []

        sentences = [self.tokenizer.decode(
            self.tokenizer.encode(sentence)[:self.max_token_len]) 
            for sentence 
            in sentences]

        out = [datum.embedding for datum in self.client.embeddings.create(input=sentences, model=self.engine).data]

        fin_embeddings.extend(out)

        assert len(sentences) == len(fin_embeddings)

        fin_embeddings = [np.array(embedding) for embedding in fin_embeddings]
        return fin_embeddings