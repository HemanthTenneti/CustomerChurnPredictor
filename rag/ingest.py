"""Build the ChromaDB vector store from the retention strategies knowledge base."""

import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Paths relative to project root
KB_PATH = os.path.join(
    os.path.dirname(__file__), "knowledge_base", "retention_strategies.md"
)
CHROMA_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")


def build_vector_store() -> None:
    """Load, chunk, embed, and persist the knowledge base into ChromaDB."""
    loader = TextLoader(KB_PATH, encoding="utf-8")
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=80)
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
    )

    print(f"Vector store built: {len(chunks)} chunks indexed.")


if __name__ == "__main__":
    build_vector_store()
