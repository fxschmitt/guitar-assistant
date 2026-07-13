"""CLI entrypoints: answer a query, or log the agent as a versioned MLflow model.

See the README's "Package layout" section: `main` wires `data.load_documents`,
`retriever.build_vector_store`, and `agent.build_agent` together and exposes the
`guitar-assistant "question"` console script; `package` exposes the
`guitar-assistant-package` console script, logging a fresh `GuitarAssistantModel` via
`mlflow_model.log_model`.
"""

from __future__ import annotations

import logging

import click
from mlflow import langchain as mlflow_langchain

from guitar_assistant.agent import build_agent
from guitar_assistant.data import load_documents
from guitar_assistant.mlflow_model import configure_default_tracking_uri, log_model
from guitar_assistant.retriever import build_vector_store

_logger = logging.getLogger(__name__)


@click.command()
@click.argument("query", type=str)
def main(query: str) -> None:
    """Answer questions about Fender/Gibson electric guitar specifications."""
    logging.basicConfig(level=logging.INFO)
    configure_default_tracking_uri()
    mlflow_langchain.autolog()

    documents = load_documents()
    available_guitar_models = sorted({document.metadata["guitar_model"] for document in documents})
    vector_store = build_vector_store(documents)
    agent = build_agent(vector_store, available_guitar_models)

    result = agent.invoke({"query": query})
    click.echo(result["answer"])


@click.command()
def package() -> None:
    """Log the current agent as a new versioned MLflow model artifact."""
    logging.basicConfig(level=logging.INFO)
    configure_default_tracking_uri()
    model_info = log_model()
    click.echo(f"Logged model: {model_info.model_uri}")
