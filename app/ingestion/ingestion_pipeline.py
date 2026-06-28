from app.ingestion.pdf_load import load_pdf
from app.ingestion.chunker import sentence_chunker
from app.embeddings.embedding_service import EmbeddingService
from app.vector_database.qdrant_manager import QdrantManager


embedding_service = EmbeddingService()
qdrant = QdrantManager()
qdrant.delete_collection()
pdf_path="D:\TINU\Trinabh_Singh_Thakur_Resume1.pdf"

text = load_pdf(pdf_path)

chunks = sentence_chunker(text)

embeddings = embedding_service.embed_documents(chunks)

qdrant.create_collection(vector_size=384)

qdrant.upload_documents(
    embeddings,
    chunks
)