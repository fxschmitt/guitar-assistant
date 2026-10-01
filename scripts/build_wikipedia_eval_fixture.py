"""Regenerate `tests/fixtures/wikipedia_eval_corpus.json`.

This fixture is a frozen snapshot of real Wikipedia content, not something built
fresh on every test run: `tests/test_end_to_end.py` is a regression guard for the
agent, and a guard that can fail because a Wikipedia article changed out from
under it (rather than because the agent regressed) is worse than no guard. So
the chunked article content is captured once, here, and committed as a plain-JSON
fixture; `test_end_to_end.py` only ever reads it back and re-embeds it locally
(real OpenAI embeddings, but no network fetch from Wikipedia).

Run this again only to deliberately refresh the fixture (e.g. to add more
models, or to pick up an intentional edit to one of the source articles) — not
as part of the normal test loop:

    uv run python scripts/build_wikipedia_eval_fixture.py

Picks a small, deliberately varied set of real guitar-model articles: several
manufacturers, a mix of short and long articles, and (per model) both an
infobox-derived overview chunk and at least one body-section chunk, so the
golden dataset in `tests/fixtures/wikipedia_golden_questions.csv` can exercise
both the fuzzy-match and the shortlist+LLM routing paths, and both chunk types,
not just infobox spec lookups.

Needs `WIKIPEDIA_CONTACT_EMAIL` set (see `.env`), same as real ingestion.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Final

from guitar_assistant.ingestion.chunking import chunk_article
from guitar_assistant.ingestion.infobox_parser import parse_article
from guitar_assistant.ingestion.wikipedia_client import WikipediaClient

_FIXTURE_PATH: Final = (
    Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "wikipedia_eval_corpus.json"
)
# One small, varied set of real guitar-model articles: 5 manufacturers, 6 models,
# a mix of short (Rickenbacker 360/12, Gibson Explorer) and longer (Gibson Flying
# V, Ibanez RG) articles. Deliberately not the Stratocaster/Telecaster/SG/Les
# Paul flagship articles used elsewhere (those run 20-40 chunks each; re-embedding
# that many chunks on every test run would make the regression guard slow and
# costly to run repeatedly while iterating on the agent).
_ARTICLE_TITLES: Final = (
    "Gibson Flying V",
    "Gibson Explorer",
    "Gretsch 6120",
    "Rickenbacker 360/12",
    "Epiphone Casino",
    "Ibanez RG",
)


def _fetch_and_chunk_articles(titles: tuple[str, ...]) -> list[dict]:
    """Fetch, parse, and chunk each of `titles`, as plain `{page_content, metadata}` dicts.

    Args:
        titles: Exact Wikipedia article titles to fetch.

    Returns:
        Every chunk from every title, in title order, ready to serialize as JSON.

    Raises:
        ValueError: A title has no `Infobox Guitar model` template, i.e. isn't a
            real guitar-model article.
    """
    chunks = []
    with WikipediaClient(max_requests=len(titles)) as client:
        for title in titles:
            fetched = client.fetch_wikitext(title)
            parsed = parse_article(title, fetched.wikitext)
            if parsed is None:
                raise ValueError(f"{title!r} has no Infobox Guitar model template.")
            chunks.extend(
                {"page_content": document.page_content, "metadata": document.metadata}
                for document in chunk_article(parsed)
            )
    return chunks


if __name__ == "__main__":
    article_chunks = _fetch_and_chunk_articles(_ARTICLE_TITLES)
    _FIXTURE_PATH.write_text(json.dumps(article_chunks, indent=2) + "\n", encoding="utf-8")
    print(
        f"Wrote {len(article_chunks)} chunks from {len(_ARTICLE_TITLES)} articles to "
        f"{_FIXTURE_PATH}."
    )
