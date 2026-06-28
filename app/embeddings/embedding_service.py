from sentence_transformers import SentenceTransformer
import numpy as np


class EmbeddingService:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, chunks):
        texts = [chunk["content"] for chunk in chunks]

        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=True
        )

        return embeddings
    
    def embed_query(self,query):
        embeddings=self.model.encode(
            query,
            convert_to_numpy=True,
            show_progress_bar=True
        )

        return embeddings