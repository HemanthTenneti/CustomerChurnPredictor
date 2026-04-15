"""RAG retriever — queries the ChromaDB vector store for relevant retention strategies."""

import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

CHROMA_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")


class RetentionRetriever:
    """Retrieves top-k relevant chunks from the retention strategies knowledge base."""

    def __init__(self) -> None:
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

        if not os.path.exists(CHROMA_DIR) or not os.listdir(CHROMA_DIR):
            # Auto-build if missing
            from rag.ingest import build_vector_store

            build_vector_store()

        self._store = Chroma(
            persist_directory=CHROMA_DIR,
            embedding_function=embeddings,
        )

    def retrieve(self, query: str, k: int = 3) -> list[str]:
        """Return the top-k most relevant chunks as plain strings."""
        docs = self._store.similarity_search(query, k=k)
        return [doc.page_content for doc in docs]
