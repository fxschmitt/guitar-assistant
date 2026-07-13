"""Streamlit UI for interactively exploring the guitar assistant agent.

Dev-only exploration tool, not part of the installable `guitar_assistant` package: builds the
agent once per server session (see `_load_agent`) and lets you fire several queries at it in a
row to probe routing/retrieval edge-cases, without rebuilding the vector store on every query
like the `guitar-assistant` CLI does.

Run with:
    uv run streamlit run app.py
"""

from __future__ import annotations

import logging

import streamlit as st
from langchain_core.documents import Document
from langgraph.graph.state import CompiledStateGraph
from mlflow import langchain as mlflow_langchain

from guitar_assistant.agent import build_agent
from guitar_assistant.data import load_documents
from guitar_assistant.mlflow_model import configure_default_tracking_uri
from guitar_assistant.retriever import build_vector_store


@st.cache_resource
def _load_agent() -> tuple[CompiledStateGraph, list[str]]:
    """Build the vector store and compile the agent once per Streamlit server session.

    Returns:
        [The compiled agent graph, the sorted list of guitar models available in the corpus].
    """
    logging.basicConfig(level=logging.INFO)
    configure_default_tracking_uri()
    mlflow_langchain.autolog()

    documents = load_documents()
    available_guitar_models = sorted({document.metadata["guitar_model"] for document in documents})
    vector_store = build_vector_store(documents)
    agent = build_agent(vector_store, available_guitar_models)
    return agent, available_guitar_models


def _render_retrieved_document(document: Document) -> None:
    """Render one retrieved document as an expander with its source spec sheet's full text.

    Args:
        document: A retrieved document, carrying `source` and `guitar_model` metadata.
    """
    title = f"{document.metadata['source']} (guitar model: {document.metadata['guitar_model']})"
    with st.expander(title):
        st.markdown(document.page_content)


def _render_turn(query: str, result: dict) -> None:
    """Render one query/answer turn: the answer, routing decision, and retrieved documents.

    Args:
        query: The question that was asked.
        result: The agent's full state dict returned by `agent.invoke`.
    """
    with st.chat_message("user"):
        st.markdown(query)
    with st.chat_message("assistant"):
        st.caption(f"Routed to guitar model: {result['guitar_model']}")
        st.markdown(result["answer"])
        for document in result["documents"]:
            _render_retrieved_document(document)


def main() -> None:
    """Render the exploration UI and handle new queries."""
    st.set_page_config(page_title="Guitar Assistant Explorer", page_icon="🎸")
    st.title("Guitar Assistant Explorer")

    agent, available_guitar_models = _load_agent()
    st.sidebar.subheader("Available guitar models")
    st.sidebar.write(", ".join(available_guitar_models))

    if "history" not in st.session_state:
        st.session_state.history = []

    for query, result in st.session_state.history:
        _render_turn(query, result)

    query = st.chat_input("Ask a question about the Telecaster, Stratocaster, or SG...")
    if query:
        result = agent.invoke({"query": query})
        st.session_state.history.append((query, result))
        _render_turn(query, result)


if __name__ == "__main__":
    main()
