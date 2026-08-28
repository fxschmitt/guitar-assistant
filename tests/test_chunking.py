"""Unit tests for guitar_assistant.chunking."""

from guitar_assistant.chunking import chunk_article
from guitar_assistant.infobox_parser import ParsedArticle

_ARTICLE = ParsedArticle(
    title="Fender Stratocaster",
    infobox={
        "title": "Fender Stratocaster",
        "caption": "A guitar in a museum.",
        "image": "Stratocaster.png",
        "manufacturer": "Fender",
        "scale": "25.5 in",
    },
    body_markdown=(
        "The Stratocaster is a model of electric guitar.\n\n"
        "## Overall design\n"
        "The archetypal Stratocaster is a solid-body electric guitar.\n\n"
        "## History\n"
        "It was introduced in 1954.\n\n"
        "### 1954-1984\n"
        "The original run.\n"
    ),
)


def test_chunk_article_returns_the_overview_chunk_first():
    # GIVEN a parsed article with an infobox and a body with sections
    # WHEN it is chunked
    chunks = chunk_article(_ARTICLE)
    # THEN the first chunk is the overview chunk
    assert chunks[0].metadata["chunk_type"] == "overview"


def test_overview_chunk_includes_the_lead_paragraph_and_spec_table():
    # GIVEN a parsed article whose body has a lead paragraph before its first heading
    # WHEN it is chunked
    overview_chunk = chunk_article(_ARTICLE)[0]
    # THEN the overview chunk's content carries the lead paragraph and spec fields
    assert "The Stratocaster is a model of electric guitar." in overview_chunk.page_content
    assert "| manufacturer | Fender |" in overview_chunk.page_content
    assert "| scale | 25.5 in |" in overview_chunk.page_content


def test_overview_chunk_excludes_non_spec_infobox_fields():
    # GIVEN an infobox with presentational fields (title, caption, image)
    # WHEN it is chunked
    overview_chunk = chunk_article(_ARTICLE)[0]
    # THEN those fields are excluded from the rendered spec table
    assert "caption" not in overview_chunk.page_content
    assert "Stratocaster.png" not in overview_chunk.page_content


def test_chunk_article_splits_the_body_into_one_chunk_per_section():
    # GIVEN a parsed article with two top-level sections, one holding a subsection
    # WHEN it is chunked
    chunks = chunk_article(_ARTICLE)
    section_chunks = [chunk for chunk in chunks if chunk.metadata["chunk_type"] == "section"]
    # THEN there is one chunk per heading, including the subsection
    assert len(section_chunks) == 3
    assert {chunk.metadata["section"] for chunk in section_chunks} == {
        "Overall design",
        "History",
    }


def test_section_chunk_metadata_carries_its_subsection_when_present():
    # GIVEN a section with a nested subsection heading
    # WHEN it is chunked
    chunks = chunk_article(_ARTICLE)
    subsection_chunk = next(chunk for chunk in chunks if chunk.metadata.get("subsection"))
    # THEN the chunk is tagged with both its section and its subsection
    assert subsection_chunk.metadata["section"] == "History"
    assert subsection_chunk.metadata["subsection"] == "1954-1984"


def test_section_chunk_content_is_prefixed_with_the_article_title():
    # GIVEN a section chunk, which will be embedded independently of its neighbors
    # WHEN it is chunked
    chunks = chunk_article(_ARTICLE)
    section_chunk = next(chunk for chunk in chunks if chunk.metadata["chunk_type"] == "section")
    # THEN its content starts with the article title, so it still reads sensibly alone
    assert section_chunk.page_content.startswith("Fender Stratocaster")


def test_every_chunk_shares_the_same_article_level_metadata():
    # GIVEN a parsed article
    # WHEN it is chunked
    chunks = chunk_article(_ARTICLE)
    # THEN every chunk carries the same manufacturer/guitar_model/source_uri
    for chunk in chunks:
        assert chunk.metadata["manufacturer"] == "fender"
        assert chunk.metadata["guitar_model"] == "fender_stratocaster"
        assert chunk.metadata["source_uri"] == "Fender Stratocaster"


def test_chunk_article_handles_a_body_with_no_sections():
    # GIVEN a parsed article whose body is a lead paragraph with no headings at all
    article = ParsedArticle(
        title="Short Article",
        infobox={"manufacturer": "Fender"},
        body_markdown="Just a short lead paragraph, nothing else.",
    )
    # WHEN it is chunked
    chunks = chunk_article(article)
    # THEN only the overview chunk is produced
    assert len(chunks) == 1
    assert chunks[0].metadata["chunk_type"] == "overview"
    assert "Just a short lead paragraph, nothing else." in chunks[0].page_content
