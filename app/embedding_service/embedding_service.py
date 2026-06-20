from sentence_transformers import SentenceTransformer
import numpy as np


class EmbeddingService:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, documents: list[str]) -> np.ndarray:
        
        embeddings = self.model.encode(
            documents,
            convert_to_numpy=True,
            show_progress_bar=True
        )

        return embeddings