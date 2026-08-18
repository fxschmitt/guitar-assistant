"""Integration test for the Wikipedia client: hits the real Wikipedia API.

Hits the network, so it's marked `integration` and excluded from the default
`uv run pytest` run. Run explicitly with
`uv run pytest -m integration tests/test_wikipedia_client_integration.py`.

Uses a small `max_requests` budget throughout, so a run of this test can never
walk more than a handful of pages into the real `Category:Electric guitars`
tree. Requires `WIKIPEDIA_CONTACT_EMAIL` to be set (see `.env`), just like the
OpenAI-backed integration tests require a real `OPENAI_API_KEY`.
"""

from __future__ import annotations

import pytest

from guitar_assistant.wikipedia_client import ELECTRIC_GUITARS_CATEGORY, WikipediaClient


@pytest.mark.integration
def test_walk_category_finds_real_electric_guitar_articles():
    # GIVEN a client with a small request budget, pointed at the real Wikipedia API
    with WikipediaClient(max_requests=5) as client:
        # WHEN the real Electric guitars category is walked
        titles = []
        for title in client.walk_category(ELECTRIC_GUITARS_CATEGORY, max_depth=1):
            titles.append(title)
            if len(titles) >= 3:
                break
        # THEN at least one real article title is found
        assert titles


@pytest.mark.integration
def test_fetch_wikitext_returns_real_article_content():
    # GIVEN a client with a small request budget, pointed at the real Wikipedia API
    with WikipediaClient(max_requests=5) as client:
        # WHEN a well-known article's wikitext is fetched
        wikitext = client.fetch_wikitext("Fender Stratocaster")
        # THEN the raw wikitext is returned and contains the expected infobox template
        assert "Infobox" in wikitext
