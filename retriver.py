"""Retriever and vector store utilities for Swiss AI Office Assistant.

This module is responsible for:
- Loading `.txt` documents from the local `docs/` folder
- Splitting into 500-character chunks (with small overlap)
- Building/Loading a Chroma vector store persisted on disk
- Providing a ready-to-use vector store for similarity search
"""

from __future__ import annotations

from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import Chroma
from openai import OpenAI


# Constants
DOCS_DIR = Path("docs")
PERSIST_DIR = Path(".chroma_store")
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


def _read_txt_documents(docs_dir: Path) -> List[Document]:
    docs: List[Document] = []
    for path in sorted(docs_dir.glob("*.txt")):
        try:
            text = path.read_text(encoding="utf-8")
        except Exception:
            text = path.read_text(errors="ignore")
        metadata = {"source": str(path)}
        docs.append(Document(page_content=text, metadata=metadata))
    return docs


def _split_documents(docs: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " "]
    )
    return splitter.split_documents(docs)


class OpenAIEmbeddingsLite(Embeddings):
    """Minimal Embeddings wrapper using OpenAI SDK directly.

    Avoids compatibility issues with langchain-openai while providing the
    interface expected by LangChain vector stores.
    """

    def __init__(self, model: str = "text-embedding-3-small", api_key: str | None = None):
        self.model = model
        self._client = OpenAI(api_key=api_key) if api_key else OpenAI()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:  # type: ignore[override]
        if not texts:
            return []
        resp = self._client.embeddings.create(model=self.model, input=texts)
        return [d.embedding for d in resp.data]

    def embed_query(self, text: str) -> list[float]:  # type: ignore[override]
        resp = self._client.embeddings.create(model=self.model, input=[text])
        return resp.data[0].embedding


def _vectorstore_exists() -> bool:
    return PERSIST_DIR.exists() and any(PERSIST_DIR.iterdir())


def ensure_index_built() -> None:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    PERSIST_DIR.mkdir(parents=True, exist_ok=True)

    if _vectorstore_exists():
        return

    # Load and split documents
    raw_docs = _read_txt_documents(DOCS_DIR)
    if not raw_docs:
        Chroma.from_texts(texts=[""], embedding=OpenAIEmbeddingsLite(), persist_directory=str(PERSIST_DIR)).persist()
        return

    chunks = _split_documents(raw_docs)

    # Build and persist store
    vs = Chroma.from_documents(documents=chunks, embedding=OpenAIEmbeddingsLite(), persist_directory=str(PERSIST_DIR))
    vs.persist()


def get_vectorstore() -> Chroma | None:
    if not _vectorstore_exists():
        return None
    return Chroma(persist_directory=str(PERSIST_DIR), embedding_function=OpenAIEmbeddingsLite())


__all__ = ["ensure_index_built", "get_vectorstore"]
