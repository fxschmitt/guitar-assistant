"""Ingest Wikipedia guitar-model articles into the persistent vector store.

See docs/scaling_strategy.md (#1/#2/#4): `run_ingestion` walks the Wikipedia
category tree, fetches each candidate article, skips ones already ingested at
their current revision (per the local `IngestionManifest`), and upserts the rest
as chunked, embedded documents into the persistent Chroma store. This is
idempotent by construction (upsert keyed by article title, old chunks of a
changed article are cleared first), so it's safe to run repeatedly — e.g. from a
weekly cron job or a manual `uv run guitar-assistant-ingest` — and each run only
ever re-processes new or changed articles.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Final, Protocol

import click
from langchain_chroma import Chroma

from guitar_assistant.chunking import chunk_article
from guitar_assistant.infobox_parser import parse_article
from guitar_assistant.manifest import DEFAULT_MANIFEST_PATH, IngestionManifest
from guitar_assistant.retriever import open_persistent_vector_store
from guitar_assistant.wikipedia_client import (
    ArticleNotFoundError,
    ELECTRIC_GUITARS_BY_MANUFACTURER_CATEGORY,
    FetchedArticle,
    WikipediaClient,
)

_CATEGORY_WALK_DEPTH: Final = 2
_DEFAULT_MAX_REQUESTS: Final = 2000
_logger = logging.getLogger(__name__)


class ArticleSource(Protocol):
    """What `run_ingestion` needs from a Wikipedia client: discover, then fetch.

    `WikipediaClient` satisfies this structurally; kept as a separate `Protocol`
    so unit tests can substitute a network-free fake without touching `httpx`.
    """

    # Method bodies are `...`, not docstrings: pyright only recognizes a bare `...`
    # as a Protocol stub, and the signatures below mirror `WikipediaClient`'s
    # already-documented methods, so a docstring here would just repeat them.
    def walk_category(  # pylint: disable=missing-function-docstring
        self, category: str, *, max_depth: int
    ) -> Iterator[str]: ...

    def fetch_wikitext(  # pylint: disable=missing-function-docstring
        self, title: str
    ) -> FetchedArticle: ...


def run_ingestion(
    client: ArticleSource,
    vector_store: Chroma,
    manifest: IngestionManifest,
    *,
    category: str = ELECTRIC_GUITARS_BY_MANUFACTURER_CATEGORY,
    max_depth: int = _CATEGORY_WALK_DEPTH,
) -> int:
    """Walk `category` and ingest every new or changed guitar-model article found.

    Args:
        client: Wikipedia API client to discover and fetch articles with.
        vector_store: Persistent vector store to upsert ingested chunks into.
        manifest: Tracks which revision of each article was last ingested;
            updated in place as articles are ingested. Not persisted by this
            function — call `manifest.save()` afterward.
        category: Wikipedia category to walk for candidate article titles.
        max_depth: Maximum subcategory depth to descend into.

    Returns:
        The number of articles actually ingested (new or changed; excludes
        titles skipped as already up to date, filtered out as non-guitar-model
        pages, or unresolvable).
    """
    ingested_count = 0
    for title in client.walk_category(category, max_depth=max_depth):
        if _ingest_one_article(client, vector_store, manifest, title):
            ingested_count += 1
    return ingested_count


def _ingest_one_article(
    client: ArticleSource, vector_store: Chroma, manifest: IngestionManifest, title: str
) -> bool:
    """Ingest `title` if it's a new or changed guitar-model article.

    Returns:
        Whether `title` was actually (re-)ingested.
    """
    try:
        article = client.fetch_wikitext(title)
    except ArticleNotFoundError:
        _logger.warning("Skipping %r: no longer resolves to a Wikipedia article.", title)
        return False
    if manifest.is_up_to_date(title, article.revision_id):
        return False
    parsed_article = parse_article(title, article.wikitext)
    if parsed_article is None:
        _logger.info("Skipping %r: no Infobox Guitar model template.", title)
        return False
    chunks = chunk_article(parsed_article)
    # Clear this article's previous chunks first: a changed article can gain,
    # lose, or rename sections between revisions, so an id-keyed upsert alone
    # could leave stale chunks behind from a section that no longer exists.
    vector_store.delete(where={"source_uri": title})
    vector_store.add_documents(chunks, ids=[f"{title}#{index}" for index in range(len(chunks))])
    manifest.mark_ingested(title, article.revision_id)
    _logger.info("Ingested %r: %d chunks.", title, len(chunks))
    return True


@click.command()
@click.option(
    "--manifest-path",
    type=click.Path(path_type=Path),
    default=DEFAULT_MANIFEST_PATH,
    show_default=True,
    help="Local JSON file tracking each article's last-ingested revision.",
)
@click.option(
    "--max-requests",
    type=int,
    default=_DEFAULT_MAX_REQUESTS,
    show_default=True,
    help="Maximum Wikipedia API requests this run may make.",
)
def main(manifest_path: Path, max_requests: int) -> None:
    """Run the Wikipedia ingestion pipeline once, updating the persistent vector store."""
    logging.basicConfig(level=logging.INFO)
    manifest = IngestionManifest.load(manifest_path)
    vector_store = open_persistent_vector_store()
    with WikipediaClient(max_requests=max_requests) as client:
        ingested_count = run_ingestion(client, vector_store, manifest)
    manifest.save(manifest_path)
    _logger.info("Done: %d article(s) ingested.", ingested_count)
