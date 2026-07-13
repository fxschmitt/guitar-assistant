"""LangGraph agent: validate, route, retrieve, then generate an answer for a query.

See the README's "Agent graph" section: `validate` rejects unusable input
before any LLM call, `route` classifies which guitar model(s) a query concerns,
`retrieve` runs a similarity search filtered to that guitar model (or across all
documents for cross-manufacturer/ambiguous queries), and `generate` synthesizes a
grounded, cited answer from the retrieved chunk(s). `route` and `generate`
divert to `reject` on a non-retryable API error; transient errors are retried
transparently before reaching either node.
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

CHAT_MODEL: Final = "gpt-4o-mini"
_ALL_GUITAR_MODELS_SENTINEL: Final = "all"
_RETRIEVAL_COUNT: Final = 3
_RETRYABLE_ERRORS: Final = (RateLimitError, APIConnectionError, APITimeoutError)
_RETRY_ATTEMPTS: Final = 3
_EMPTY_QUERY_ERROR: Final = "The question was empty."
_UNENCODABLE_QUERY_ERROR: Final = "The question contained characters that could not be processed."

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


def route(state: Query, classify: Callable[[str], BaseModel]) -> dict:
    """Classify which guitar model(s) the query concerns.

    Args:
        state: Current graph state; only `query` is read.
        classify: A callable (bound to a specific structured-output schema)
            that classifies a query into a `guitar_model` decision.

    Returns:
        A partial state update setting `guitar_model`, or setting `error` if the
        classification call fails with a non-retryable error.
    """
    try:
        decision = classify(state["query"])
    except BadRequestError as error:
        return {"error": str(error)}
    return {"guitar_model": cast(_RouteDecision, decision).guitar_model}


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

    route_decision_schema = _build_route_decision_schema(available_guitar_models)
    classify_chain = (
        _ROUTE_PROMPT | chat_model.with_structured_output(route_decision_schema)
    ).with_retry(**retry_kwargs)

    def classify(query: str) -> BaseModel:
        return cast(
            BaseModel,
            classify_chain.invoke(
                {
                    "query": query,
                    "available_guitar_models": ", ".join(available_guitar_models),
                    "all_guitar_models_sentinel": _ALL_GUITAR_MODELS_SENTINEL,
                }
            ),
        )

    def route_node(state: AgentState) -> dict:
        return route(state, classify)

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
