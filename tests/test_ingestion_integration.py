"""Integration test for the ingestion pipeline: hits the real Wikipedia API.

Hits the network, so it's marked `integration` and excluded from the default
`uv run pytest` run. Run explicitly with
`uv run pytest -m integration tests/test_ingestion_integration.py`.

Points `run_ingestion` at `Category:Fender Stratocasters` rather than the full
by-manufacturer tree: unlike the other integration tests, `run_ingestion` walks
its whole target category with no early-break title cap, so a narrow,
known-small leaf category (~15 articles, no subcategories) is what keeps this
test's request count bounded rather than a `max_requests` budget alone. Requires
`WIKIPEDIA_CONTACT_EMAIL` to be set (see `.env`), just like
`test_wikipedia_client_integration.py`.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from guitar_assistant.ingestion import run_ingestion
from guitar_assistant.manifest import IngestionManifest
from guitar_assistant.retriever import open_persistent_vector_store
from guitar_assistant.wikipedia_client import WikipediaClient

_STRATOCASTER_MODELS_CATEGORY = "Category:Fender Stratocasters"


@pytest.mark.integration
def test_run_ingestion_indexes_real_articles_and_skips_them_on_a_second_run(tmp_path: Path):
    # GIVEN a fresh persistent store and an empty manifest
    vector_store = open_persistent_vector_store(tmp_path / ".chroma")
    manifest = IngestionManifest()
    # WHEN a real batch of Stratocaster-model titles is ingested for the first time
    with WikipediaClient(max_requests=30) as client:
        first_run_count = run_ingestion(
            client, vector_store, manifest, category=_STRATOCASTER_MODELS_CATEGORY, max_depth=0
        )
    # THEN real guitar-model articles are chunked, embedded, and retrievable
    assert first_run_count > 0
    assert vector_store._collection.count() > 0
    results = vector_store.similarity_search("Fender Stratocaster scale length", k=1)
    assert results[0].metadata["source_uri"] == "Fender Stratocaster"
    # WHEN the same category is ingested again without any revision having changed
    with WikipediaClient(max_requests=30) as client:
        second_run_count = run_ingestion(
            client, vector_store, manifest, category=_STRATOCASTER_MODELS_CATEGORY, max_depth=0
        )
    # THEN nothing is re-ingested, since every already-seen title is up to date
    assert second_run_count == 0
