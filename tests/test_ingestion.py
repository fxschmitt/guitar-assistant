"""Unit tests for guitar_assistant.ingestion."""

from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

from langchain_chroma import Chroma
import pytest

from guitar_assistant.ingestion import run_ingestion
from guitar_assistant.manifest import IngestionManifest
from guitar_assistant.retriever import open_persistent_vector_store
from guitar_assistant.wikipedia_client import ArticleNotFoundError, FetchedArticle

_STRATOCASTER_WIKITEXT = """{{Infobox Guitar model
|manufacturer=[[Fender]]
|scale=25.5 in
}}
The Stratocaster is a model of electric guitar.

==Overall design==
A solid-body electric guitar.
"""

_COMPANY_OVERVIEW_WIKITEXT = """{{Infobox company
|name=B.C. Rich Guitars
}}
An American brand of guitars.
"""


@dataclass
class _FakeArticleSource:
    """Network-free stand-in for `WikipediaClient`, used to unit-test `run_ingestion`."""

    titles: list[str]
    articles: dict[str, FetchedArticle] = field(default_factory=dict)

    def walk_category(self, category: str, *, max_depth: int) -> Iterator[str]:
        del category, max_depth  # unused: the fake ignores which category was requested
        return iter(self.titles)

    def fetch_wikitext(self, title: str) -> FetchedArticle:
        try:
            return self.articles[title]
        except KeyError:
            raise ArticleNotFoundError(f"No article for {title!r}.") from None


@pytest.fixture(name="persistent_vector_store")
def fixture_persistent_vector_store(tmp_path: Path, fake_embeddings) -> Chroma:
    return open_persistent_vector_store(tmp_path / ".chroma", embeddings=fake_embeddings)


def test_run_ingestion_indexes_a_new_guitar_model_article(persistent_vector_store):
    # GIVEN a source with one guitar-model article, and an empty manifest
    client = _FakeArticleSource(
        titles=["Fender Stratocaster"],
        articles={"Fender Stratocaster": FetchedArticle(_STRATOCASTER_WIKITEXT, revision_id=1)},
    )
    manifest = IngestionManifest()
    # WHEN ingestion is run
    ingested_count = run_ingestion(client, persistent_vector_store, manifest)
    # THEN the article is chunked, embedded, and recorded in the manifest
    assert ingested_count == 1
    assert persistent_vector_store._collection.count() > 0
    assert manifest.is_up_to_date("Fender Stratocaster", 1)


def test_run_ingestion_skips_an_article_already_up_to_date(persistent_vector_store):
    # GIVEN an article already ingested at its current revision
    client = _FakeArticleSource(
        titles=["Fender Stratocaster"],
        articles={"Fender Stratocaster": FetchedArticle(_STRATOCASTER_WIKITEXT, revision_id=1)},
    )
    manifest = IngestionManifest()
    manifest.mark_ingested("Fender Stratocaster", 1)
    # WHEN ingestion is run
    ingested_count = run_ingestion(client, persistent_vector_store, manifest)
    # THEN nothing is (re-)ingested
    assert ingested_count == 0
    assert persistent_vector_store._collection.count() == 0


def test_run_ingestion_reingests_an_article_whose_revision_changed(persistent_vector_store):
    # GIVEN an article ingested at an older revision than the one now fetched
    client = _FakeArticleSource(
        titles=["Fender Stratocaster"],
        articles={"Fender Stratocaster": FetchedArticle(_STRATOCASTER_WIKITEXT, revision_id=2)},
    )
    manifest = IngestionManifest()
    manifest.mark_ingested("Fender Stratocaster", 1)
    # WHEN ingestion is run
    ingested_count = run_ingestion(client, persistent_vector_store, manifest)
    # THEN the article is re-ingested and the manifest reflects the new revision
    assert ingested_count == 1
    assert manifest.is_up_to_date("Fender Stratocaster", 2)


def test_run_ingestion_replaces_stale_chunks_of_a_changed_article(persistent_vector_store):
    # GIVEN an article that previously had a section no longer present in its latest revision
    stale_wikitext = _STRATOCASTER_WIKITEXT + "\n==Vibrato system==\nSome old content.\n"
    client = _FakeArticleSource(
        titles=["Fender Stratocaster"],
        articles={"Fender Stratocaster": FetchedArticle(stale_wikitext, revision_id=1)},
    )
    manifest = IngestionManifest()
    run_ingestion(client, persistent_vector_store, manifest)
    stale_chunk_count = persistent_vector_store._collection.count()
    # WHEN the article is re-ingested at a new revision with that section removed
    client.articles["Fender Stratocaster"] = FetchedArticle(_STRATOCASTER_WIKITEXT, revision_id=2)
    run_ingestion(client, persistent_vector_store, manifest)
    # THEN the resulting chunk count reflects only the current sections, not both
    assert persistent_vector_store._collection.count() < stale_chunk_count


def test_run_ingestion_skips_a_page_without_a_guitar_infobox(persistent_vector_store):
    # GIVEN a source with only a non-guitar-model page (e.g. a manufacturer overview)
    client = _FakeArticleSource(
        titles=["B.C. Rich"],
        articles={"B.C. Rich": FetchedArticle(_COMPANY_OVERVIEW_WIKITEXT, revision_id=1)},
    )
    manifest = IngestionManifest()
    # WHEN ingestion is run
    ingested_count = run_ingestion(client, persistent_vector_store, manifest)
    # THEN nothing is ingested and nothing is recorded in the manifest
    assert ingested_count == 0
    assert not manifest.is_up_to_date("B.C. Rich", 1)


def test_run_ingestion_skips_an_unresolvable_title(persistent_vector_store):
    # GIVEN a title with no corresponding article (e.g. a stale category listing)
    client = _FakeArticleSource(titles=["A Deleted Page"], articles={})
    manifest = IngestionManifest()
    # WHEN ingestion is run
    ingested_count = run_ingestion(client, persistent_vector_store, manifest)
    # THEN it is skipped rather than raising
    assert ingested_count == 0


def test_run_ingestion_counts_only_articles_actually_ingested(persistent_vector_store):
    # GIVEN a mix of a new guitar-model article, an up-to-date one, and non-guitar noise
    client = _FakeArticleSource(
        titles=["Fender Stratocaster", "Gibson SG", "B.C. Rich"],
        articles={
            "Fender Stratocaster": FetchedArticle(_STRATOCASTER_WIKITEXT, revision_id=1),
            "Gibson SG": FetchedArticle(_STRATOCASTER_WIKITEXT, revision_id=1),
            "B.C. Rich": FetchedArticle(_COMPANY_OVERVIEW_WIKITEXT, revision_id=1),
        },
    )
    manifest = IngestionManifest()
    manifest.mark_ingested("Gibson SG", 1)
    # WHEN ingestion is run
    ingested_count = run_ingestion(client, persistent_vector_store, manifest)
    # THEN only the new guitar-model article counts as ingested
    assert ingested_count == 1
