"""Unit tests for guitar_assistant.agent."""

from typing import Any, cast

import httpx
from langchain_core.documents import Document
from openai import BadRequestError
from pydantic import ValidationError
import pytest

from conftest import AVAILABLE_GUITAR_MODELS as _AVAILABLE_GUITAR_MODELS

from guitar_assistant.agent import (
    ErrorCheck,
    _build_route_decision_schema,
    _check_query,
    _route_after_check,
    build_agent,
    generate,
    reject,
    retrieve,
    route,
    validate,
)

_FAKE_BAD_REQUEST_ERROR = BadRequestError(
    "context length exceeded",
    response=httpx.Response(
        status_code=400, request=httpx.Request("POST", "https://api.openai.com/v1/chat")
    ),
    body=None,
)


@pytest.mark.parametrize("guitar_model", [*_AVAILABLE_GUITAR_MODELS, "all"])
def test_build_route_decision_schema_accepts_available_models_and_all(guitar_model: str):
    # GIVEN a schema built from the corpus's available guitar models
    schema = _build_route_decision_schema(_AVAILABLE_GUITAR_MODELS)
    # WHEN validating a decision naming an available guitar model, or "all"
    decision = schema(guitar_model=guitar_model)
    # THEN it is accepted
    assert cast(Any, decision).guitar_model == guitar_model


def test_build_route_decision_schema_rejects_unknown_model():
    # GIVEN a schema built from the corpus's available guitar models
    schema = _build_route_decision_schema(_AVAILABLE_GUITAR_MODELS)
    # WHEN validating a decision naming a guitar model absent from the corpus
    # THEN it is rejected
    with pytest.raises(ValidationError):
        schema(guitar_model="les_paul")


def test_route_sets_guitar_model_from_classifier_decision():
    # GIVEN a classifier that always decides on the "stratocaster" guitar model
    schema = _build_route_decision_schema(_AVAILABLE_GUITAR_MODELS)

    def classify(query: str):
        return schema(guitar_model="stratocaster")

    # WHEN routing a query
    result = route({"query": "What is the scale length?"}, classify)
    # THEN the state update carries the classifier's decision
    assert result == {"guitar_model": "stratocaster"}


def test_route_sets_error_when_classifier_raises_bad_request():
    # GIVEN a classifier that fails with a non-retryable API error
    def classify(query: str):
        raise _FAKE_BAD_REQUEST_ERROR

    # WHEN routing a query
    result = route({"query": "What is the scale length?"}, classify)
    # THEN the state update carries the error instead of a guitar_model
    assert result == {"error": str(_FAKE_BAD_REQUEST_ERROR)}


def test_retrieve_filters_by_the_routed_guitar_model(vector_store):
    # GIVEN a vector store indexing one document per guitar model
    # WHEN retrieving with a query routed to a single guitar model
    result = retrieve({"query": "stratocaster", "guitar_model": "stratocaster"}, vector_store)
    # THEN only that guitar model's document is returned
    guitar_models = [document.metadata["guitar_model"] for document in result["documents"]]
    assert guitar_models == ["stratocaster"]


def test_retrieve_searches_all_documents_when_guitar_model_is_all(vector_store):
    # GIVEN a vector store indexing one document per guitar model
    # WHEN retrieving with a query routed to "all"
    result = retrieve({"query": "stratocaster", "guitar_model": "all"}, vector_store)
    # THEN documents across guitar models may be returned
    assert len(result["documents"]) == len(_AVAILABLE_GUITAR_MODELS)


def test_generate_synthesizes_answer_from_query_and_documents():
    # GIVEN a synthesize callable that records its arguments and returns a fixed answer
    documents = [Document(page_content="content", metadata={"source": "stratocaster.md"})]
    calls = []

    def synthesize(query: str, docs: list[Document]) -> str:
        calls.append((query, docs))
        return "The scale length is 25.5 in [stratocaster.md]."

    # WHEN generating an answer
    result = generate({"query": "What is the scale length?", "documents": documents}, synthesize)

    # THEN synthesize received the query and documents, and the state update carries its answer
    assert calls == [("What is the scale length?", documents)]
    assert result == {"answer": "The scale length is 25.5 in [stratocaster.md]."}


def test_generate_sets_error_when_synthesize_raises_bad_request():
    # GIVEN a synthesize callable that fails with a non-retryable API error
    def synthesize(query: str, docs: list[Document]) -> str:
        raise _FAKE_BAD_REQUEST_ERROR

    # WHEN generating an answer
    result = generate({"query": "What is the scale length?", "documents": []}, synthesize)

    # THEN the state update carries the error instead of an answer
    assert result == {"error": str(_FAKE_BAD_REQUEST_ERROR)}


@pytest.mark.parametrize(
    ("query", "expected_error"),
    [
        ("", "The question was empty."),
        ("   ", "The question was empty."),
        ("\udc80", "The question contained characters that could not be processed."),
        ("What is the scale length?", None),
    ],
)
def test_check_query(query: str, expected_error: str | None):
    # GIVEN a raw query, possibly empty, whitespace-only, or unencodable
    # WHEN checking it
    # THEN it is flagged with the expected error, or accepted (None)
    assert _check_query(query) == expected_error


def test_validate_sets_error_for_an_empty_query():
    # GIVEN an empty query
    # WHEN validating it
    result = validate({"query": ""})
    # THEN the state update carries an error
    assert result == {"error": "The question was empty."}


def test_validate_is_a_no_op_for_a_usable_query():
    # GIVEN a well-formed query
    # WHEN validating it
    result = validate({"query": "What is the scale length?"})
    # THEN no error is set
    assert not result


def test_reject_turns_the_recorded_error_into_an_answer():
    # GIVEN state carrying a recorded error
    # WHEN rejecting the query
    result = reject({"error": "The question was empty."})
    # THEN the answer explains the failure
    assert result == {"answer": "I couldn't answer that question: The question was empty."}


@pytest.mark.parametrize(
    ("state", "expected"),
    [
        ({"error": "boom"}, "reject"),
        ({"error": None}, "continue"),
        ({}, "continue"),
    ],
)
def test_route_after_check(state: ErrorCheck, expected: str):
    # GIVEN state that may or may not carry a recorded error
    # WHEN deciding how to route
    # THEN the graph continues, or diverts to reject, accordingly
    assert _route_after_check(state) == expected


def test_build_agent_rejects_an_empty_query_without_calling_the_chat_model(
    vector_store, available_guitar_models, fake_chat_model
):
    # GIVEN a compiled agent
    agent = build_agent(vector_store, available_guitar_models, llm=fake_chat_model)
    # WHEN invoking it with an empty query
    result = agent.invoke({"query": ""})
    # THEN it short-circuits to the rejection answer without routing or retrieving
    assert result["answer"] == "I couldn't answer that question: The question was empty."
    assert "guitar_model" not in result
    assert "documents" not in result
