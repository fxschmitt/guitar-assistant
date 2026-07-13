"""Unit tests for guitar_assistant.retriever."""


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
