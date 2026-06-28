from app.embeddings.embedding_service import EmbeddingService
from app.vector_database.qdrant_manager import QdrantManager

embedding_service = EmbeddingService()
qdrant = QdrantManager()

query="what are the skills present?"

query_embedding=embedding_service.embed_query(query)

results=qdrant.search(query_embedding.tolist(),limit=3)

print(results)