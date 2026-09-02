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
    _fuzzy_match_guitar_model,
    _route_after_check,
    _shortlist_candidates,
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


@pytest.mark.parametrize(
    ("query", "expected_match"),
    [
        ("What is the scale length of the Stratocaster?", "stratocaster"),
        ("What is the SG's body wood?", "sg"),
        ("Tell me about the Telecaster's pickups.", "telecaster"),
        ("Which guitar has better sustain?", None),
        ("Tell me about the Strat pickups", None),
    ],
)
def test_fuzzy_match_guitar_model(query: str, expected_match: str | None):
    # GIVEN the demo corpus's available guitar models
    # WHEN fuzzy-matching a query that confidently names one, or doesn't
    # THEN the matched slug (or None, on a miss) is returned
    assert _fuzzy_match_guitar_model(query, _AVAILABLE_GUITAR_MODELS) == expected_match


def test_fuzzy_match_guitar_model_returns_none_when_no_models_are_available():
    # GIVEN no indexed guitar models at all
    # WHEN fuzzy-matching any query
    # THEN there is nothing to match against, so it returns None
    assert _fuzzy_match_guitar_model("What is the scale length?", []) is None


def test_shortlist_candidates_returns_distinct_models_in_similarity_rank_order(vector_store):
    # GIVEN a vector store indexing one document per guitar model
    matching_model = _AVAILABLE_GUITAR_MODELS[1]
    # WHEN shortlisting candidates for a query matching one document's guitar model
    shortlist = _shortlist_candidates(matching_model, vector_store, search_count=1)
    # THEN that guitar model is the (only) shortlisted candidate
    assert shortlist == [matching_model]


def test_shortlist_candidates_deduplicates_and_caps_at_shortlist_size(vector_store):
    # GIVEN a vector store indexing one document per guitar model
    # WHEN shortlisting with a search wide enough to see every document, capped below the
    # corpus size
    shortlist = _shortlist_candidates(
        "guitar", vector_store, search_count=len(_AVAILABLE_GUITAR_MODELS), shortlist_size=1
    )
    # THEN no more than shortlist_size distinct models are returned
    assert len(shortlist) == 1


def test_route_returns_the_fuzzy_match_without_calling_shortlist_then_classify():
    # GIVEN a confident fuzzy match and a shortlist_then_classify that records whether
    # it was ever called
    calls = []

    def shortlist_then_classify(query: str) -> str:
        calls.append(query)
        return "sg"

    # WHEN routing a query
    result = route(
        {"query": "What is the scale length?"},
        lambda _query: "stratocaster",
        shortlist_then_classify,
    )
    # THEN the fuzzy match is used directly, with no fallback classification
    assert result == {"guitar_model": "stratocaster"}
    assert not calls


def test_route_falls_through_to_shortlist_then_classify_on_a_fuzzy_match_miss():
    # GIVEN a fuzzy-match miss
    # WHEN routing a query
    result = route(
        {"query": "Which guitar has better sustain?"}, lambda _query: None, lambda _query: "sg"
    )
    # THEN the shortlist_then_classify decision is used
    assert result == {"guitar_model": "sg"}


def test_route_sets_error_when_shortlist_then_classify_raises_bad_request():
    # GIVEN a fuzzy-match miss and a shortlist_then_classify that fails with a
    # non-retryable API error
    def shortlist_then_classify(query: str) -> str:
        raise _FAKE_BAD_REQUEST_ERROR

    # WHEN routing a query
    result = route(
        {"query": "Which guitar has better sustain?"}, lambda _query: None, shortlist_then_classify
    )
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


def test_build_agent_routes_a_confident_fuzzy_match_without_calling_the_chat_model(
    vector_store, available_guitar_models, fake_chat_model
):
    # GIVEN a compiled agent, whose fake chat model would answer any classification call
    # with "stratocaster" (the fixture's default), but is only wired for routing on a
    # fuzzy-match miss
    agent = build_agent(vector_store, available_guitar_models, llm=fake_chat_model)
    # WHEN invoking it with a query that confidently names a guitar model by name
    result = agent.invoke({"query": "What is the SG's body wood?"})
    # THEN it routed via the fuzzy match, retrieving the SG's document, and the chat
    # model's structured-output classification was never invoked
    assert result["guitar_model"] == "sg"
    assert not fake_chat_model._structured_output_calls


def test_build_agent_falls_through_to_shortlist_and_classify_on_a_fuzzy_match_miss(
    vector_store, available_guitar_models, fake_chat_model
):
    # GIVEN a compiled agent
    agent = build_agent(vector_store, available_guitar_models, llm=fake_chat_model)
    # WHEN invoking it with a query naming no guitar model directly
    result = agent.invoke({"query": "Which guitar has better sustain?"})
    # THEN it fell through to the shortlist+LLM path, calling the chat model once, and
    # used its (fixture default "stratocaster") decision
    assert result["guitar_model"] == "stratocaster"
    assert len(fake_chat_model._structured_output_calls) == 1


def test_build_agent_scopes_the_fallback_schema_to_the_shortlist(
    vector_store, available_guitar_models, fake_chat_model
):
    # GIVEN a compiled agent over a corpus small enough that the shortlist covers every
    # available guitar model
    agent = build_agent(vector_store, available_guitar_models, llm=fake_chat_model)
    # WHEN invoking it with a query naming no guitar model directly, falling through to
    # the shortlist+LLM path
    agent.invoke({"query": "Which guitar has better sustain?"})
    # THEN the structured-output schema built for that call is restricted to the
    # shortlisted models plus the "all" sentinel, not left open to arbitrary values
    (fallback_schema,) = fake_chat_model._structured_output_calls
    valid_guitar_models = set(fallback_schema.model_fields["guitar_model"].annotation.__args__)
    assert valid_guitar_models == {*available_guitar_models, "all"}
