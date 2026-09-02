"""Unit tests for guitar_assistant.retriever."""

from pathlib import Path

from langchain_core.documents import Document
import pytest

from guitar_assistant.retriever import (
    CORPUS_ENV_VAR,
    DEMO_CORPUS,
    WIKIPEDIA_CORPUS,
    build_vector_store,
    list_indexed_guitar_models,
    load_corpus,
    open_persistent_vector_store,
)


def test_build_vector_store_indexes_every_document(vector_store, available_guitar_models):
    # GIVEN a vector store indexing one document per guitar model
    # WHEN inspecting the underlying collection
    # THEN every document is retrievable
    assert vector_store._collection.count() == len(available_guitar_models)


def test_build_vector_store_retrieves_the_matching_document_by_similarity(
    vector_store, available_guitar_models
):
    # GIVEN a vector store indexing one document per guitar model
    matching_model = available_guitar_models[1]
    # WHEN searching with a query matching one document's guitar model
    results = vector_store.similarity_search(matching_model, k=1)
    # THEN the document for that guitar model is returned
    assert results[0].metadata["guitar_model"] == matching_model


def test_open_persistent_vector_store_persists_documents_across_reopens(
    tmp_path: Path, fake_embeddings
):
    # GIVEN a document added to a persistent store
    persist_directory = tmp_path / ".chroma"
    store = open_persistent_vector_store(persist_directory, embeddings=fake_embeddings)
    store.add_documents([Document(page_content="stratocaster spec sheet")])
    # WHEN the same persist directory is reopened as a fresh store instance
    reopened_store = open_persistent_vector_store(persist_directory, embeddings=fake_embeddings)
    # THEN the previously added document is still there, without re-adding it
    assert reopened_store._collection.count() == 1


def test_list_indexed_guitar_models_returns_the_distinct_models_sorted(
    vector_store, available_guitar_models
):
    # GIVEN a vector store indexing one document per guitar model, in an arbitrary order
    # WHEN listing the guitar models actually indexed
    indexed_guitar_models = list_indexed_guitar_models(vector_store)
    # THEN every distinct model is returned, sorted, with no duplicates
    assert indexed_guitar_models == sorted(set(available_guitar_models))


def test_list_indexed_guitar_models_deduplicates_repeated_models(fake_embeddings):
    # GIVEN a vector store with two documents sharing the same guitar_model
    documents = [
        Document(
            page_content="overview",
            metadata={"guitar_model": "stratocaster", "source": "overview.md"},
        ),
        Document(
            page_content="specs section",
            metadata={"guitar_model": "stratocaster", "source": "specs.md"},
        ),
    ]
    store = build_vector_store(documents, embeddings=fake_embeddings)
    # WHEN listing the guitar models actually indexed
    indexed_guitar_models = list_indexed_guitar_models(store)
    # THEN the repeated model is only returned once
    assert indexed_guitar_models == ["stratocaster"]


def test_load_corpus_defaults_to_the_wikipedia_corpus_when_the_env_var_is_unset(
    monkeypatch, tmp_path, fake_embeddings
):
    # GIVEN GUITAR_ASSISTANT_CORPUS is not set, and a persistent store pre-populated
    # with a Wikipedia-ingested document
    monkeypatch.delenv(CORPUS_ENV_VAR, raising=False)
    persist_directory = tmp_path / ".chroma"
    seed_store = open_persistent_vector_store(persist_directory, embeddings=fake_embeddings)
    seed_store.add_documents(
        [Document(page_content="stratocaster overview", metadata={"guitar_model": "stratocaster"})]
    )
    # WHEN loading the corpus
    vector_store, available_guitar_models = load_corpus(
        embeddings=fake_embeddings, persist_directory=persist_directory
    )
    # THEN it defaults to opening the persistent Wikipedia store, deriving the models
    # from what's indexed
    assert available_guitar_models == ["stratocaster"]
    assert vector_store._collection.count() == 1


def test_load_corpus_uses_the_demo_corpus_when_explicitly_selected(monkeypatch, fake_embeddings):
    # GIVEN GUITAR_ASSISTANT_CORPUS explicitly set to "demo"
    monkeypatch.setenv(CORPUS_ENV_VAR, DEMO_CORPUS)
    # WHEN loading the corpus
    vector_store, available_guitar_models = load_corpus(embeddings=fake_embeddings)
    # THEN it builds an ephemeral store from the bundled demo spec sheets, not the
    # persistent Wikipedia store
    assert available_guitar_models == ["sg", "stratocaster", "telecaster"]
    assert vector_store._collection.count() == 3


def test_load_corpus_opens_the_persistent_store_when_explicitly_selected(
    monkeypatch, tmp_path, fake_embeddings
):
    # GIVEN a persistent store pre-populated with a Wikipedia-ingested document, and
    # GUITAR_ASSISTANT_CORPUS set to "wikipedia"
    monkeypatch.setenv(CORPUS_ENV_VAR, WIKIPEDIA_CORPUS)
    persist_directory = tmp_path / ".chroma"
    seed_store = open_persistent_vector_store(persist_directory, embeddings=fake_embeddings)
    seed_store.add_documents(
        [Document(page_content="stratocaster overview", metadata={"guitar_model": "stratocaster"})]
    )
    # WHEN loading the corpus
    vector_store, available_guitar_models = load_corpus(
        embeddings=fake_embeddings, persist_directory=persist_directory
    )
    # THEN it reopens the persistent store and derives the models from what's indexed
    assert available_guitar_models == ["stratocaster"]
    assert vector_store._collection.count() == 1


def test_load_corpus_raises_for_an_unsupported_corpus_value(monkeypatch):
    # GIVEN GUITAR_ASSISTANT_CORPUS set to an unsupported value
    monkeypatch.setenv(CORPUS_ENV_VAR, "not-a-real-corpus")
    # WHEN loading the corpus
    # THEN it raises, naming the offending value
    with pytest.raises(ValueError, match="not-a-real-corpus"):
        load_corpus()
