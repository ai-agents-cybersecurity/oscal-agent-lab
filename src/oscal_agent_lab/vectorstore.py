# src/oscal_agent_lab/vectorstore.py
"""Build and manage FAISS vectorstore for OSCAL content."""

from __future__ import annotations

from pathlib import Path

from langchain_community.vectorstores.faiss import FAISS
from langchain_openai import OpenAIEmbeddings

from .config import OSCAL_CATALOG_PATH, OPENAI_EMBED_MODEL, VECTORSTORE_PATH
from .oscal_loader import load_oscal_json, catalog_to_documents

_vectorstore: FAISS | None = None


def get_embeddings() -> OpenAIEmbeddings:
    """Get the configured embeddings model."""
    return OpenAIEmbeddings(model=OPENAI_EMBED_MODEL)


def get_vectorstore(force_rebuild: bool = False) -> FAISS:
    """
    Lazy-build and cache a FAISS vectorstore from the OSCAL catalog.

    Args:
        force_rebuild: If True, rebuild the index even if cached.

    Returns:
        FAISS vectorstore instance.
    """
    global _vectorstore

    if _vectorstore is not None and not force_rebuild:
        return _vectorstore

    # Try to load from disk first
    index_path = VECTORSTORE_PATH / "faiss_index"
    if index_path.exists() and not force_rebuild:
        try:
            embeddings = get_embeddings()
            _vectorstore = FAISS.load_local(
                str(index_path),
                embeddings,
                allow_dangerous_deserialization=True,
            )
            print(f"Loaded vectorstore from {index_path}")
            return _vectorstore
        except Exception as e:
            print(f"Failed to load cached vectorstore: {e}")

    # Build from scratch
    if not OSCAL_CATALOG_PATH.exists():
        raise FileNotFoundError(
            f"OSCAL catalog not found at {OSCAL_CATALOG_PATH}. "
            "Did you run: git submodule add https://github.com/usnistgov/oscal-content.git data/oscal-content"
        )

    print(f"Building vectorstore from {OSCAL_CATALOG_PATH}...")
    catalog_json = load_oscal_json(OSCAL_CATALOG_PATH)
    docs = catalog_to_documents(catalog_json)
    print(f"Loaded {len(docs)} control documents")

    embeddings = get_embeddings()
    _vectorstore = FAISS.from_documents(docs, embeddings)

    # Persist to disk
    VECTORSTORE_PATH.mkdir(parents=True, exist_ok=True)
    _vectorstore.save_local(str(index_path))
    print(f"Saved vectorstore to {index_path}")

    return _vectorstore


def get_retriever(k: int = 5):
    """
    Get a retriever for the OSCAL vectorstore.

    Args:
        k: Number of documents to retrieve.

    Returns:
        A LangChain retriever.
    """
    vs = get_vectorstore()
    return vs.as_retriever(search_kwargs={"k": k})


def add_documents_to_store(documents: list, save: bool = True) -> None:
    """
    Add additional documents to the vectorstore.

    Useful for adding SSPs, profiles, etc.

    Args:
        documents: List of LangChain Documents to add.
        save: Whether to persist changes to disk.
    """
    vs = get_vectorstore()
    vs.add_documents(documents)

    if save:
        index_path = VECTORSTORE_PATH / "faiss_index"
        vs.save_local(str(index_path))
        print(f"Added {len(documents)} documents and saved to {index_path}")
