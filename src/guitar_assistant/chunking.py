"""Split a parsed Wikipedia article into indexable LangChain Documents.

See the "Chunking" section of docs/scaling_strategy.md (#3): `chunk_article` turns
one `ParsedArticle` into an overview chunk (a rendered infobox spec table plus the
article's lead paragraph) and one chunk per `##`/`###` section of the body, all
tagged with the same article-level metadata (`manufacturer`, `guitar_model`,
`source_uri`) used later to filter retrieval by guitar model.
"""

from __future__ import annotations

import re
from typing import Final

from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter

from guitar_assistant.infobox_parser import ParsedArticle

# Infobox fields that describe the page itself (image caption, raw title) rather
# than the guitar's specification; excluded from the rendered spec table.
_NON_SPEC_INFOBOX_FIELDS: Final = frozenset({"title", "caption", "image"})
_SECTION_HEADERS: Final = [("##", "section"), ("###", "subsection")]
_FIRST_SECTION_HEADING_PATTERN: Final = re.compile(r"^##\s", re.MULTILINE)


def chunk_article(article: ParsedArticle) -> list[Document]:
    """Split `article` into an overview chunk plus one chunk per body section.

    Args:
        article: A parsed article, as returned by `infobox_parser.parse_article`.

    Returns:
        One overview `Document` (spec table + lead paragraph), followed by one
        `Document` per `##`/`###` section of the body. Every chunk carries the
        same `manufacturer`/`guitar_model`/`source_uri` metadata.
    """
    base_metadata = _base_metadata(article)
    lead_paragraph, sections_markdown = _split_lead_and_sections(article.body_markdown)
    return [
        _overview_chunk(article, lead_paragraph, base_metadata),
        *_section_chunks(article, sections_markdown, base_metadata),
    ]


def _base_metadata(article: ParsedArticle) -> dict[str, str]:
    """Metadata shared by every chunk of `article`."""
    return {
        "manufacturer": article.infobox.get("manufacturer", "").lower(),
        "guitar_model": _slugify(article.title),
        "source_uri": article.title,
    }


def _split_lead_and_sections(body_markdown: str) -> tuple[str, str]:
    """Split `body_markdown` into its lead paragraph and its `##`-headed sections."""
    match = _FIRST_SECTION_HEADING_PATTERN.search(body_markdown)
    if match is None:
        return body_markdown.strip(), ""
    return body_markdown[: match.start()].strip(), body_markdown[match.start() :]


def _overview_chunk(
    article: ParsedArticle, lead_paragraph: str, base_metadata: dict[str, str]
) -> Document:
    """Build the article's overview chunk: lead paragraph + rendered spec table."""
    spec_table = _render_spec_table(article.infobox)
    content = "\n\n".join(
        part for part in (f"# {article.title}", lead_paragraph, spec_table) if part
    )
    return Document(
        page_content=content,
        metadata={**base_metadata, "section": "", "subsection": "", "chunk_type": "overview"},
    )


def _render_spec_table(infobox: dict[str, str]) -> str:
    """Render the infobox's spec fields as a Markdown table, dropping non-spec ones."""
    rows = [
        f"| {field} | {value} |"
        for field, value in infobox.items()
        if field not in _NON_SPEC_INFOBOX_FIELDS
    ]
    if not rows:
        return ""
    return "\n".join(["| Field | Value |", "| --- | --- |", *rows])


def _section_chunks(
    article: ParsedArticle, sections_markdown: str, base_metadata: dict[str, str]
) -> list[Document]:
    """Split the body's sections into one chunk per `##`/`###` heading.

    Headers are kept in each chunk's content (`strip_headers=False`) and the
    article title is prefixed, so a chunk still reads sensibly once it's
    separated from its neighbors for embedding.
    """
    if not sections_markdown:
        return []
    splitter = MarkdownHeaderTextSplitter(headers_to_split_on=_SECTION_HEADERS, strip_headers=False)
    return [
        Document(
            page_content=f"{article.title}\n\n{split_document.page_content}",
            metadata={
                **base_metadata,
                "section": split_document.metadata.get("section", ""),
                "subsection": split_document.metadata.get("subsection", ""),
                "chunk_type": "section",
            },
        )
        for split_document in splitter.split_text(sections_markdown)
    ]


def _slugify(text: str) -> str:
    """Turn `text` into a lowercase, underscore-separated slug."""
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
