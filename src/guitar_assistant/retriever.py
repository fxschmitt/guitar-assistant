"""Build the in-memory Chroma vector store used for retrieval.

See the README's "Indexing pipeline" section: documents are embedded with
OpenAI's `text-embedding-3-small` and stored in an ephemeral (in-memory)
Chroma collection that is rebuilt on every process start.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from typing import Final

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings

EMBEDDING_MODEL: Final = "text-embedding-3-small"

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
