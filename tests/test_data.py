"""Unit tests for guitar_assistant.data."""

from pathlib import Path
from typing import Final

import pytest

from guitar_assistant.data import load_documents

_FILE_CONTENTS: Final = {
    "Fender_Telecaster.md": "Telecaster spec sheet content.",
    "Fender_Stratocaster.md": "Stratocaster spec sheet content.",
    "Gibson_SG.md": "SG spec sheet content.",
}


@pytest.fixture(name="source_dir")
def fixture_source_dir(tmp_path: Path) -> Path:
    for filename, content in _FILE_CONTENTS.items():
        (tmp_path / filename).write_text(content, encoding="utf-8")
    return tmp_path


def test_load_documents_returns_one_document_per_source_file(source_dir: Path):
    # GIVEN a directory containing the three source Markdown files
    # WHEN the documents are loaded
    documents = load_documents(source_dir)
    # THEN exactly one Document is returned per source file
    assert len(documents) == len(_FILE_CONTENTS)


@pytest.mark.parametrize(
    ("filename", "expected_manufacturer", "expected_model"),
    [
        ("Fender_Telecaster.md", "fender", "telecaster"),
        ("Fender_Stratocaster.md", "fender", "stratocaster"),
        ("Gibson_SG.md", "gibson", "sg"),
    ],
)
def test_load_documents_attaches_routing_metadata(
    source_dir: Path, filename: str, expected_manufacturer: str, expected_model: str
):
    # GIVEN a directory containing the three source Markdown files
    # WHEN the documents are loaded
    documents = load_documents(source_dir)
    # THEN the document for each source file carries its expected manufacturer/model metadata
    matching = next(document for document in documents if document.metadata["source"] == filename)
    assert matching.metadata["manufacturer"] == expected_manufacturer
    assert matching.metadata["guitar_model"] == expected_model


def test_load_documents_preserves_file_content(source_dir: Path):
    # GIVEN a directory containing the three source Markdown files
    # WHEN the documents are loaded
    documents = load_documents(source_dir)
    # THEN each document's page content matches its source file verbatim
    for document in documents:
        expected_content = _FILE_CONTENTS[document.metadata["source"]]
        assert document.page_content == expected_content


def test_load_documents_defaults_to_the_bundled_package_resources():
    # GIVEN no explicit source directory
    # WHEN the documents are loaded using the default source directory
    documents = load_documents()
    # THEN the spec sheets bundled inside the package are found and loaded
    assert len(documents) == len(_FILE_CONTENTS)
    assert {document.metadata["source"] for document in documents} == set(_FILE_CONTENTS)
