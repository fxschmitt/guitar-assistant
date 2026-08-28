"""Build the Chroma vector stores used for retrieval.

See the README's "Indexing pipeline" section: documents are embedded with
OpenAI's `text-embedding-3-small`. `build_vector_store` holds them in an
ephemeral (in-memory) Chroma collection rebuilt on every process start — the
right amount of ceremony for the 3-document hand-written demo corpus.
`open_persistent_vector_store` is the Wikipedia-ingestion counterpart from
docs/scaling_strategy.md (#2): a Chroma collection persisted to a local
directory, so embeddings written by a prior `guitar-assistant-ingest` run
survive across process restarts instead of being re-embedded every time.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from pathlib import Path
from typing import Final

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings

EMBEDDING_MODEL: Final = "text-embedding-3-small"
DEFAULT_PERSIST_DIRECTORY: Final = Path(".chroma")
_WIKIPEDIA_COLLECTION_NAME: Final = "guitar_models"

load_dotenv()


def build_vector_store(
    documents: Sequence[Document], embeddings: Embeddings | None = None
) -> Chroma:
    """Embed documents and load them into an in-memory Chroma vector store.

    Args:
        documents: Documents to embed and index.
        embeddings: Embeddings model to use. Defaults to OpenAI's
            `text-embedding-3-small`. Overridable for testing without network
            access.

    Returns:
        A Chroma vector store containing the embedded documents.
    """
    return Chroma.from_documents(
        list(documents),
        embedding=embeddings or OpenAIEmbeddings(model=EMBEDDING_MODEL),
        # Chroma's default in-memory client reuses one process-wide collection
        # keyed by name, so a fixed name would leak documents between
        # unrelated calls (e.g. separate tests, or repeated indexing runs).
        collection_name=uuid.uuid4().hex,
    )


def open_persistent_vector_store(
    persist_directory: Path = DEFAULT_PERSIST_DIRECTORY, embeddings: Embeddings | None = None
) -> Chroma:
    """Open (or create) the persistent Chroma store of Wikipedia-ingested articles.

    Unlike `build_vector_store`'s ephemeral collection, this store survives
    across process restarts: chunks already embedded by a prior
    `guitar-assistant-ingest` run are read back from `persist_directory`
    rather than being re-embedded.

    Args:
        persist_directory: Local directory Chroma persists its data to.
            Defaults to `.chroma/` at the current working directory
            (gitignored, like `mlflow.db`).
        embeddings: Embeddings model to use. Defaults to OpenAI's
            `text-embedding-3-small`. Overridable for testing without network
            access.

    Returns:
        A Chroma vector store backed by `persist_directory`, with a fixed
        collection name so repeated calls (e.g. an ingestion run followed by
        a query-time load) see the same collection.
    """
    return Chroma(
        collection_name=_WIKIPEDIA_COLLECTION_NAME,
        embedding_function=embeddings or OpenAIEmbeddings(model=EMBEDDING_MODEL),
        persist_directory=str(persist_directory),
    )
