"""Load the guitar spec-sheet Markdown files and turn them into LangChain Documents.

One chunk per source document (see the "Indexing pipeline" section of the
README for the chunking rationale); each chunk carries the ``manufacturer`` and
``guitar_model`` metadata used later to filter retrieval by guitar model.
"""

from __future__ import annotations

from importlib import resources
from importlib.resources.abc import Traversable
from typing import Final, NamedTuple

from langchain_core.documents import Document

_RESOURCES_PACKAGE: Final = "guitar_assistant.resources"


class _SourceDocument(NamedTuple):
    """Maps one source Markdown file to its routing metadata."""

    filename: str
    manufacturer: str
    guitar_model: str


_SOURCE_DOCUMENTS: Final = (
    _SourceDocument("Fender_Telecaster.md", manufacturer="fender", guitar_model="telecaster"),
    _SourceDocument("Fender_Stratocaster.md", manufacturer="fender", guitar_model="stratocaster"),
    _SourceDocument("Gibson_SG.md", manufacturer="gibson", guitar_model="sg"),
)


def load_documents(source_dir: Traversable | None = None) -> list[Document]:
    """Load the guitar spec sheets as whole-document LangChain Documents.

    Args:
        source_dir: Directory containing the source Markdown files. Defaults to
            the spec sheets bundled inside the installed package, so this works
            identically whether the code is run from a source checkout or a
            pip-installed wheel.

    Returns:
        One Document per source file, each tagged with `manufacturer`,
        `guitar_model`, and `source` metadata.
    """
    directory = source_dir if source_dir is not None else resources.files(_RESOURCES_PACKAGE)
    return [
        Document(
            page_content=(directory / source.filename).read_text(encoding="utf-8"),
            metadata={
                "manufacturer": source.manufacturer,
                "guitar_model": source.guitar_model,
                "source": source.filename,
            },
        )
        for source in _SOURCE_DOCUMENTS
    ]
