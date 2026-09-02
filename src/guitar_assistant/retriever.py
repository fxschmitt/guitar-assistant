"""Build the Chroma vector stores used for retrieval.

See the README's "Indexing pipeline" section: documents are embedded with
OpenAI's `text-embedding-3-small`. `build_vector_store` holds them in an
ephemeral (in-memory) Chroma collection rebuilt on every process start — the
right amount of ceremony for the 3-document hand-written demo corpus.
`open_persistent_vector_store` is the Wikipedia-ingestion counterpart from
docs/scaling_strategy.md (#2): a Chroma collection persisted to a local
directory, so embeddings written by a prior `guitar-assistant-ingest` run
survive across process restarts instead of being re-embedded every time.
`load_corpus` picks between the two, controlled by the `GUITAR_ASSISTANT_CORPUS`
environment variable, so the CLI/MLflow model entrypoints don't each duplicate
that choice.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import Sequence
from pathlib import Path
from typing import Final

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings

from guitar_assistant.data import load_documents

EMBEDDING_MODEL: Final = "text-embedding-3-small"
DEFAULT_PERSIST_DIRECTORY: Final = Path(".chroma")
_WIKIPEDIA_COLLECTION_NAME: Final = "guitar_models"

CORPUS_ENV_VAR: Final = "GUITAR_ASSISTANT_CORPUS"
DEMO_CORPUS: Final = "demo"
WIKIPEDIA_CORPUS: Final = "wikipedia"

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


def list_indexed_guitar_models(vector_store: Chroma) -> list[str]:
    """Return the distinct `guitar_model` metadata values already indexed in `vector_store`.

    Reads the collection's metadata directly rather than depending on
    `IngestionManifest`: the manifest is an ingestion-time bookkeeping detail, not a
    query-time API, and this stays correct even if the manifest file is missing or
    stale relative to the store.

    Args:
        vector_store: A Chroma vector store whose documents carry a `guitar_model`
            metadata field.

    Returns:
        The distinct `guitar_model` values found across all indexed documents, sorted.
    """
    metadatas = vector_store.get(include=["metadatas"])["metadatas"]
    return sorted({metadata["guitar_model"] for metadata in metadatas})


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


def load_corpus(
    embeddings: Embeddings | None = None,
    persist_directory: Path = DEFAULT_PERSIST_DIRECTORY,
) -> tuple[Chroma, list[str]]:
    """Build or open the vector store selected by the `GUITAR_ASSISTANT_CORPUS` env var.

    Reads `GUITAR_ASSISTANT_CORPUS` (`"demo"` or `"wikipedia"`), defaulting to
    `"wikipedia"`: the persistent, Wikipedia-ingested corpus is the real runtime
    corpus and the intended default once a `guitar-assistant-ingest` run has
    populated it. `"demo"` is an explicit opt-in that rebuilds an ephemeral store
    from the 3 bundled spec sheets instead — the quick-start/test fixture corpus,
    not the default runtime path. `"wikipedia"` opens the persistent store and
    derives the available models from what's actually indexed there, since there
    is no in-memory `documents` list to read them from at query time.

    Args:
        embeddings: Embeddings model to use. Defaults to OpenAI's
            `text-embedding-3-small`. Overridable for testing without network
            access.
        persist_directory: Forwarded to `open_persistent_vector_store` when the
            `"wikipedia"` corpus is selected; ignored for `"demo"`. Overridable
            for testing, so tests don't touch the real `.chroma/` directory.

    Returns:
        A `(vector_store, available_guitar_models)` pair, ready to pass to
        `agent.build_agent`.

    Raises:
        ValueError: `GUITAR_ASSISTANT_CORPUS` is set to neither `"demo"` nor
            `"wikipedia"`.
    """
    corpus = os.environ.get(CORPUS_ENV_VAR, WIKIPEDIA_CORPUS)
    if corpus == WIKIPEDIA_CORPUS:
        vector_store = open_persistent_vector_store(persist_directory, embeddings=embeddings)
        return vector_store, list_indexed_guitar_models(vector_store)
    if corpus == DEMO_CORPUS:
        documents = load_documents()
        available_guitar_models = sorted(
            {document.metadata["guitar_model"] for document in documents}
        )
        return build_vector_store(documents, embeddings=embeddings), available_guitar_models
    raise ValueError(
        f"{CORPUS_ENV_VAR}={corpus!r} is not supported; use {DEMO_CORPUS!r} or "
        f"{WIKIPEDIA_CORPUS!r}."
    )
