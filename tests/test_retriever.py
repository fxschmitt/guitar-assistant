"""Unit tests for guitar_assistant.retriever."""

from pathlib import Path

from langchain_core.documents import Document

from guitar_assistant.retriever import open_persistent_vector_store


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
