"""LangGraph agent: validate, route, retrieve, then generate an answer for a query.

See the README's "Agent graph" section: `validate` rejects unusable input
before any LLM call, `route` classifies which guitar model(s) a query concerns,
`retrieve` runs a similarity search filtered to that guitar model (or across all
documents for cross-manufacturer/ambiguous queries), and `generate` synthesizes a
grounded, cited answer from the retrieved chunk(s). `route` and `generate`
divert to `reject` on a non-retryable API error; transient errors are retried
transparently before reaching either node.

Routing is two-stage (docs/scaling_strategy.md #6): `_fuzzy_match_guitar_model`
tries a fuzzy/alias match against the known `guitar_model` slugs first, with no
LLM call at all — the common case, since most queries name a model directly. On
a miss, `_shortlist_candidates` runs a cheap unfiltered vector pre-search to
shortlist a handful of candidate models, and only that shortlist (not the full
corpus) is classified by an LLM call. This keeps the router's prompt and
structured-output schema small regardless of total corpus size.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Final, Literal, Protocol, TypedDict, cast

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import VectorStore
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from openai import APIConnectionError, APITimeoutError, BadRequestError, RateLimitError
from pydantic import BaseModel, create_model
from rapidfuzz import fuzz, process

CHAT_MODEL: Final = "gpt-4o-mini"
_ALL_GUITAR_MODELS_SENTINEL: Final = "all"
_RETRIEVAL_COUNT: Final = 3
_RETRYABLE_ERRORS: Final = (RateLimitError, APIConnectionError, APITimeoutError)
_RETRY_ATTEMPTS: Final = 3
_EMPTY_QUERY_ERROR: Final = "The question was empty."
_UNENCODABLE_QUERY_ERROR: Final = "The question contained characters that could not be processed."
_FUZZY_MATCH_SCORE_CUTOFF: Final = 90
_SHORTLIST_SEARCH_COUNT: Final = 10
_SHORTLIST_SIZE: Final = 5

_ROUTE_PROMPT: Final = ChatPromptTemplate.from_template(
    "You are routing a question about guitar spec sheets to the correct spec sheet(s).\n"
    "Available guitar models: {available_guitar_models}.\n"
    'Pick the single guitar model the question is about, or "{all_guitar_models_sentinel}" if '
    "the question is ambiguous or concerns more than one guitar model.\n\n"
    "Question: {query}"
)

_GENERATE_PROMPT: Final = ChatPromptTemplate.from_template(
    "Answer the question using only the context below. Cite which spec sheet(s) "
    "(the `source` shown with each excerpt) the answer came from.\n\n"
    "Context:\n{context}\n\n"
    "Question: {query}"
)

load_dotenv()


class Query(TypedDict):
    """A raw incoming question, not yet routed.

    Args:
        query: The user's question.
    """

    query: str


class RoutedQuery(TypedDict):
    """A query assigned to a guitar model, ready to be retrieved against.

    Args:
        query: The user's question.
        guitar_model: The guitar model the router selected ("all" for cross-manufacturer/
            ambiguous queries).
    """

    query: str
    guitar_model: str


class RetrievedQuery(TypedDict):
    """A query paired with the context retrieved for it, ready to be answered.

    Args:
        query: The user's question.
        documents: Chunks returned by the retrieve node.
    """

    query: str
    documents: list[Document]


class ErrorCheck(TypedDict, total=False):
    """The subset of state read by `reject` and `_route_after_check`.

    Args:
        error: Set by `validate`, `route`, or `generate` on a non-retryable
            failure (invalid input, or a non-retryable API error); its
            presence routes the graph to the `reject` node instead of
            continuing. Absent, or `None`, on the happy path.
    """

    error: str | None


class AgentState(Query, RoutedQuery, RetrievedQuery, ErrorCheck):
    """Full state threaded through the validate -> route -> retrieve -> generate graph.

    Args:
        answer: The final synthesized, cited answer.
    """

    answer: str


class _RouteDecision(Protocol):  # pylint: disable=too-few-public-methods
    """Structural type for a route decision produced by a dynamic schema.

    A `Protocol` describing only data attributes has no methods by design;
    pylint's public-methods count doesn't account for that.

    `_build_route_decision_schema` builds its schema at runtime via
    `pydantic.create_model`, so its `guitar_model` field is invisible to static
    analysis; this protocol is used to `cast` decision instances back to a
    type that exposes it.

    Args:
        guitar_model: The routed guitar model name (or the "all" sentinel).
    """

    guitar_model: str


def _build_route_decision_schema(available_guitar_models: Sequence[str]) -> type[BaseModel]:
    """Build a structured-output schema whose valid guitar model names match the corpus.

    Args:
        available_guitar_models: The guitar model names present in the indexed corpus.

    Returns:
        A Pydantic model with a `guitar_model` field restricted to
        `available_guitar_models` plus the "all" sentinel, so the router's valid
        answers automatically track whatever manuals are actually indexed.
    """
    return create_model(
        "RouteDecision",
        guitar_model=(Literal[(*available_guitar_models, _ALL_GUITAR_MODELS_SENTINEL)], ...),
    )


def _fuzzy_match_guitar_model(query: str, available_guitar_models: Sequence[str]) -> str | None:
    """Fuzzy-match `query`'s text against the known `guitar_model` slugs.

    No LLM call: this is the first, cheap routing stage, expected to cover most
    real queries since people usually name a model directly (e.g. "Stratocaster",
    matching the `fender_stratocaster` slug).

    Args:
        query: The user's raw question.
        available_guitar_models: The guitar model slugs present in the indexed
            corpus. Slugs are matched with underscores treated as spaces (e.g.
            `fender_stratocaster` matches "Fender Stratocaster").

    Returns:
        The confidently matched guitar model slug, or `None` if no slug scores
        at or above `_FUZZY_MATCH_SCORE_CUTOFF` against the query text.
    """
    slug_by_normalized_name = {model.replace("_", " "): model for model in available_guitar_models}
    match = process.extractOne(
        query.lower(),
        slug_by_normalized_name.keys(),
        scorer=fuzz.partial_ratio,
        score_cutoff=_FUZZY_MATCH_SCORE_CUTOFF,
    )
    return slug_by_normalized_name[match[0]] if match else None


def _shortlist_candidates(
    query: str,
    vector_store: VectorStore,
    *,
    search_count: int = _SHORTLIST_SEARCH_COUNT,
    shortlist_size: int = _SHORTLIST_SIZE,
) -> list[str]:
    """Shortlist candidate `guitar_model` values for `query` via a cheap vector pre-search.

    The second routing stage, run only on a fuzzy-match miss: an unfiltered
    similarity search over the whole corpus, keeping just the distinct
    `guitar_model` values seen among the nearest chunks. The LLM classification
    that follows is scoped to this shortlist (plus the "all" sentinel) instead of
    the full corpus, so its prompt and structured-output schema stay small
    regardless of total corpus size.

    Args:
        query: The user's raw question.
        vector_store: The indexed corpus to search, unfiltered.
        search_count: How many nearest chunks to inspect for candidate models.
        shortlist_size: Maximum number of distinct `guitar_model` values to
            shortlist.

    Returns:
        Up to `shortlist_size` distinct `guitar_model` values, ordered by the
        similarity rank of each value's nearest chunk.
    """
    candidates: list[str] = []
    for document in vector_store.similarity_search(query, k=search_count):
        guitar_model = document.metadata["guitar_model"]
        if guitar_model not in candidates:
            candidates.append(guitar_model)
        if len(candidates) == shortlist_size:
            break
    return candidates


def _check_query(query: str) -> str | None:
    """Check a raw query for problems that would break routing/generation.

    Args:
        query: The user's raw question.

    Returns:
        A user-facing error message if the query is unusable, or `None` if
        it's safe to route and answer.
    """
    if not query.strip():
        return _EMPTY_QUERY_ERROR
    try:
        query.encode("utf-8")
    except UnicodeEncodeError:
        return _UNENCODABLE_QUERY_ERROR
    return None


def validate(state: Query) -> dict:
    """Check the raw query before any LLM call is made.

    Args:
        state: Current graph state; only `query` is read.

    Returns:
        A partial state update setting `error` if the query is unusable, or
        an empty update otherwise.
    """
    error = _check_query(state["query"])
    return {"error": error} if error else {}


def reject(state: ErrorCheck) -> dict:
    """Turn a recorded `error` into a final, user-facing answer.

    Args:
        state: Current graph state; only `error` is read.

    Returns:
        A partial state update setting `answer` to a message explaining the
        failure, in place of a synthesized answer.
    """
    return {"answer": f"I couldn't answer that question: {state.get('error')}"}


def route(
    state: Query,
    fuzzy_match: Callable[[str], str | None],
    shortlist_then_classify: Callable[[str], str],
) -> dict:
    """Classify which guitar model(s) the query concerns.

    Tries `fuzzy_match` first; on a miss (`None`), falls through to
    `shortlist_then_classify`, the LLM-backed fallback.

    Args:
        state: Current graph state; only `query` is read.
        fuzzy_match: Matches the query's text against known `guitar_model`
            slugs, with no LLM call. Returns the matched slug, or `None` on a
            miss.
        shortlist_then_classify: Vector-presearch shortlist plus LLM
            classification scoped to it; invoked only on a `fuzzy_match` miss.

    Returns:
        A partial state update setting `guitar_model`, or setting `error` if
        `shortlist_then_classify` fails with a non-retryable error.
    """
    matched_guitar_model = fuzzy_match(state["query"])
    if matched_guitar_model is not None:
        return {"guitar_model": matched_guitar_model}
    try:
        guitar_model = shortlist_then_classify(state["query"])
    except BadRequestError as error:
        return {"error": str(error)}
    return {"guitar_model": guitar_model}


def retrieve(state: RoutedQuery, vector_store: VectorStore) -> dict:
    """Similarity-search the vector store, filtered to the routed guitar model.

    Args:
        state: Current graph state; reads `query` and `guitar_model`.
        vector_store: The indexed corpus to search.

    Returns:
        A partial state update setting `documents`.
    """
    is_cross_guitar_model_query = state["guitar_model"] == _ALL_GUITAR_MODELS_SENTINEL
    metadata_filter = (
        None if is_cross_guitar_model_query else {"guitar_model": state["guitar_model"]}
    )
    documents = vector_store.similarity_search(
        state["query"], k=_RETRIEVAL_COUNT, filter=metadata_filter
    )
    return {"documents": documents}


def generate(state: RetrievedQuery, synthesize: Callable[[str, list[Document]], str]) -> dict:
    """Synthesize a grounded, cited answer from the retrieved documents.

    Args:
        state: Current graph state; reads `query` and `documents`.
        synthesize: A callable that turns a query and its retrieved documents
            into a final answer.

    Returns:
        A partial state update setting `answer`, or setting `error` if the
        generation call fails with a non-retryable error.
    """
    try:
        answer = synthesize(state["query"], state["documents"])
    except BadRequestError as error:
        return {"error": str(error)}
    return {"answer": answer}


def _route_after_check(state: ErrorCheck) -> Literal["reject", "continue"]:
    """Decide whether to continue the graph or divert to `reject`.

    Args:
        state: Current graph state; only `error` is read.

    Returns:
        `"reject"` if a prior node recorded an `error`, `"continue"` otherwise.
    """
    return "reject" if state.get("error") else "continue"


def build_agent(
    vector_store: VectorStore,
    available_guitar_models: Sequence[str],
    llm: BaseChatModel | None = None,
) -> CompiledStateGraph:
    """Assemble and compile the validate -> route -> retrieve -> generate agent graph.

    Args:
        vector_store: The indexed corpus to retrieve from.
        available_guitar_models: The guitar model names present in the indexed
            corpus, used to constrain the router's structured-output schema.
        llm: Chat model used for both routing and generation. Defaults to
            OpenAI's `gpt-4o-mini`. Overridable for testing without network
            access. Wrapped with retry for transient connection/rate-limit
            errors; non-retryable errors (e.g. context-length-exceeded) are
            handled by the `route`/`generate` nodes instead.

    Returns:
        A compiled LangGraph app exposing `.invoke({"query": ...})`.
    """
    # We built the classify and generate chains here using either a true llm
    # or a mock for testing, so they can close over the specific structured-output
    # schema and retry behavior.
    chat_model = llm or ChatOpenAI(model=CHAT_MODEL)

    retry_kwargs = {
        "retry_if_exception_type": _RETRYABLE_ERRORS,
        "wait_exponential_jitter": True,
        "stop_after_attempt": _RETRY_ATTEMPTS,
    }

    def fuzzy_match(query: str) -> str | None:
        return _fuzzy_match_guitar_model(query, available_guitar_models)

    def shortlist_then_classify(query: str) -> str:
        shortlist = _shortlist_candidates(query, vector_store)
        route_decision_schema = _build_route_decision_schema(shortlist)
        classify_chain = (
            _ROUTE_PROMPT | chat_model.with_structured_output(route_decision_schema)
        ).with_retry(**retry_kwargs)
        decision = classify_chain.invoke(
            {
                "query": query,
                "available_guitar_models": ", ".join(shortlist),
                "all_guitar_models_sentinel": _ALL_GUITAR_MODELS_SENTINEL,
            }
        )
        return cast(_RouteDecision, decision).guitar_model

    def route_node(state: AgentState) -> dict:
        return route(state, fuzzy_match, shortlist_then_classify)

    generate_chain = (_GENERATE_PROMPT | chat_model).with_retry(**retry_kwargs)

    def synthesize(query: str, documents: list[Document]) -> str:
        context = "\n\n".join(
            f"[Source: {document.metadata['source']}]\n{document.page_content}"
            for document in documents
        )
        return cast(str, generate_chain.invoke({"query": query, "context": context}).content)

    def generate_node(state: AgentState) -> dict:
        return generate(state, synthesize)

    # Similarly we built the retrieve node using either a true vector store or a mock for testing.
    def retrieve_node(state: AgentState) -> dict:
        return retrieve(state, vector_store)

    graph = StateGraph(AgentState)
    graph.add_node("validate", validate)
    graph.add_node("route", route_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate", generate_node)
    graph.add_node("reject", reject)
    graph.add_edge(START, "validate")
    graph.add_conditional_edges(
        "validate", _route_after_check, {"reject": "reject", "continue": "route"}
    )
    graph.add_conditional_edges(
        "route", _route_after_check, {"reject": "reject", "continue": "retrieve"}
    )
    graph.add_edge("retrieve", "generate")
    graph.add_conditional_edges(
        "generate", _route_after_check, {"reject": "reject", "continue": END}
    )
    graph.add_edge("reject", END)
    return graph.compile()
