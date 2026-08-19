"""Integration test for the infobox parser: hits the real Wikipedia API.

Hits the network, so it's marked `integration` and excluded from the default
`uv run pytest` run. Run explicitly with
`uv run pytest -m integration tests/test_infobox_parser_integration.py`.

Uses a small `max_requests` budget throughout, so a run of this test can never
walk more than a handful of pages into the real `Category:Electric guitars`
tree. Requires `WIKIPEDIA_CONTACT_EMAIL` to be set (see `.env`), just like
`test_wikipedia_client_integration.py`.
"""

from __future__ import annotations

import pytest

from guitar_assistant.infobox_parser import parse_article
from guitar_assistant.wikipedia_client import (
    ArticleNotFoundError,
    ELECTRIC_GUITARS_BY_MANUFACTURER_CATEGORY,
    WikipediaClient,
)


@pytest.mark.integration
def test_parse_article_extracts_sensible_fields_from_a_known_article():
    # GIVEN a well-known guitar model's real wikitext
    with WikipediaClient(max_requests=5) as client:
        article = client.fetch_wikitext("Fender Stratocaster")
    # WHEN it is parsed
    parsed = parse_article("Fender Stratocaster", article.wikitext)
    # THEN the infobox fields and body read as expected, with no leftover wiki markup
    assert parsed is not None
    assert parsed.infobox["manufacturer"] == "Fender"
    assert "[[" not in parsed.body_markdown
    assert "## History" in parsed.body_markdown
    assert "References" not in parsed.body_markdown


@pytest.mark.integration
def test_parse_article_filters_real_wikipedia_noise_from_the_manufacturer_walk():
    # GIVEN a batch of real titles from the by-manufacturer entry point, which is
    # known to mix guitar-model articles with noise (manufacturer overview pages,
    # "List of ..." pages) that don't carry an Infobox Guitar model template
    with WikipediaClient(max_requests=40) as client:
        titles = []
        for title in client.walk_category(ELECTRIC_GUITARS_BY_MANUFACTURER_CATEGORY, max_depth=2):
            titles.append(title)
            if len(titles) >= 15:
                break
        parsed_articles = []
        for title in titles:
            try:
                article = client.fetch_wikitext(title)
            except ArticleNotFoundError:
                continue
            parsed_article = parse_article(title, article.wikitext)
            if parsed_article is not None:
                parsed_articles.append(parsed_article)
    # WHEN each title is fetched and parsed
    # THEN most titles yield a real guitar-model parse, and known noise is filtered out
    assert len(parsed_articles) >= 10
    parsed_titles = {parsed_article.title for parsed_article in parsed_articles}
    assert "List of B.C. Rich guitars" not in parsed_titles
    assert "B.C. Rich" not in parsed_titles
