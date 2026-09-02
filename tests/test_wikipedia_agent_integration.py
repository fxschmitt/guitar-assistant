"""Integration test: the real agent, querying the real persistent Wikipedia store.

Exercises the Wikipedia migration's query-time wiring end to end against real
data: a real Wikipedia ingestion run populates a persistent Chroma store, then
`retriever.load_corpus` reopens that same store (a separate call, mirroring how a
real query-time process never shares an in-memory `documents` list with the
ingestion run that wrote it) and the real, OpenAI-backed agent answers a question
against it — proving `guitar-assistant-ingest` and `guitar-assistant` actually
compose, not just each in isolation.

Needs a real `OPENAI_API_KEY` (embeds ingested articles and answers the query) and
`WIKIPEDIA_CONTACT_EMAIL` (see `test_ingestion_integration.py`), and hits both the
Wikipedia and OpenAI APIs, so it's marked `integration` and excluded from the
default `uv run pytest` run. Run explicitly with:

```bash
uv run pytest -m integration tests/test_wikipedia_agent_integration.py
```

Points ingestion at `Category:Fender Stratocasters`, the same small, known leaf
category `test_ingestion_integration.py` uses (~15 articles, no subcategories), so
this test's real API usage stays bounded regardless of `max_requests` alone.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from guitar_assistant.agent import build_agent
from guitar_assistant.ingestion.manifest import IngestionManifest
from guitar_assistant.ingestion.pipeline import run_ingestion
from guitar_assistant.ingestion.wikipedia_client import WikipediaClient
from guitar_assistant.retriever import (
    CORPUS_ENV_VAR,
    WIKIPEDIA_CORPUS,
    load_corpus,
    open_persistent_vector_store,
)

_STRATOCASTER_MODELS_CATEGORY = "Category:Fender Stratocasters"
_STRATOCASTER_GUITAR_MODEL = "fender_stratocaster"


@pytest.mark.integration
def test_agent_answers_from_the_real_persistent_wikipedia_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    # GIVEN a persistent store populated by a real ingestion run over a small, known
    # category, exactly as `guitar-assistant-ingest` would leave it
    persist_directory = tmp_path / ".chroma"
    ingestion_vector_store = open_persistent_vector_store(persist_directory)
    with WikipediaClient(max_requests=30) as client:
        ingested_count = run_ingestion(
            client,
            ingestion_vector_store,
            IngestionManifest(),
            category=_STRATOCASTER_MODELS_CATEGORY,
            max_depth=0,
        )
    assert ingested_count > 0

    # WHEN a separate, later call reopens that same store the way a real query-time
    # process does: via retriever.load_corpus, not the ingestion run's own handle
    monkeypatch.setenv(CORPUS_ENV_VAR, WIKIPEDIA_CORPUS)
    vector_store, available_guitar_models = load_corpus(persist_directory=persist_directory)
    assert _STRATOCASTER_GUITAR_MODEL in available_guitar_models

    # AND the real, OpenAI-backed agent is built over it and asked a real question
    # naming the model directly
    agent = build_agent(vector_store, available_guitar_models)
    result = agent.invoke({"query": "What is the scale length of the Fender Stratocaster?"})

    # THEN it routes to the real ingested article via the fuzzy-match stage (no LLM
    # routing call needed, since the query names the model directly) and answers,
    # grounded in that article
    assert result["guitar_model"] == _STRATOCASTER_GUITAR_MODEL
    assert "25.5" in result["answer"]
