"""Unit tests for guitar_assistant.mlflow_model."""

from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast
from urllib.parse import urlparse

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatResult
from langgraph.graph.state import CompiledStateGraph
import mlflow
import mlflow.artifacts
from mlflow.pyfunc.model import PythonModelContext
import pandas as pd
from pydantic import SecretStr
import pytest

import guitar_assistant.mlflow_model as mlflow_model_module
from guitar_assistant.mlflow_model import GuitarAssistantModel, log_model


@pytest.fixture(name="tracking_uri")
def fixture_tracking_uri(tmp_path) -> Iterator[str]:
    # MLflow's filesystem tracking backend (e.g. "file://.../mlruns") is in
    # maintenance mode as of MLflow 3.x; a local SQLite database is the
    # supported no-server local store.
    uri = f"sqlite:///{tmp_path}/mlflow.db"
    mlflow.set_tracking_uri(uri)
    yield uri
    mlflow.set_tracking_uri("")


def test_load_context_compiles_an_agent_from_injected_fakes(fake_embeddings, fake_chat_model):
    # GIVEN a model configured with fake embeddings and a fake chat model
    model = GuitarAssistantModel(llm=fake_chat_model, embeddings=fake_embeddings)
    # WHEN loading its context
    model.load_context(context=cast(PythonModelContext, None))
    # THEN it compiles a usable agent without any network access
    agent = cast(CompiledStateGraph, model._agent)
    result = agent.invoke({"query": "What is the scale length of the Stratocaster?"})
    assert result["answer"] == "The scale length is 25.5 in [stratocaster.md]."


def test_predict_returns_one_answer_for_a_single_row(fake_embeddings, fake_chat_model):
    # GIVEN a loaded model
    model = GuitarAssistantModel(llm=fake_chat_model, embeddings=fake_embeddings)
    model.load_context(context=cast(PythonModelContext, None))
    # WHEN predicting on a single-row DataFrame
    result = model.predict(
        context=cast(PythonModelContext, None),
        model_input=pd.DataFrame({"query": ["What is the scale length?"]}),
    )
    # THEN it returns a one-element list with the fake model's answer
    assert result == ["The scale length is 25.5 in [stratocaster.md]."]


def test_predict_returns_one_answer_per_row_in_order(fake_embeddings, fake_chat_model):
    # GIVEN a loaded model
    model = GuitarAssistantModel(llm=fake_chat_model, embeddings=fake_embeddings)
    model.load_context(context=cast(PythonModelContext, None))
    # WHEN predicting on a multi-row DataFrame
    result = model.predict(
        context=cast(PythonModelContext, None),
        model_input=pd.DataFrame(
            {"query": ["What is the scale length?", "What is the body wood?"]}
        ),
    )
    # THEN it returns one answer per row, in row order
    assert result == ["The scale length is 25.5 in [stratocaster.md]."] * 2


def test_log_model_records_expected_params_and_tags(tracking_uri, fake_embeddings, fake_chat_model):
    # GIVEN a fresh MLflow tracking store isolated to a temp directory
    # WHEN logging the model with injected fakes
    model_info = log_model(llm=fake_chat_model, embeddings=fake_embeddings)
    run = mlflow.get_run(model_info.run_id)
    # THEN the run records the LLM/embedding params and the chunking/corpus tags
    assert run.data.params["chat_model"] == "gpt-4o-mini"
    assert run.data.params["embedding_model"] == "text-embedding-3-small"
    assert run.data.tags["chunking_strategy"] == "whole-document"
    assert "corpus_version" in run.data.tags


def test_log_model_produces_a_loadable_model_that_predicts(
    tracking_uri, fake_embeddings, fake_chat_model
):
    # GIVEN a model logged with injected fakes
    model_info = log_model(llm=fake_chat_model, embeddings=fake_embeddings)
    # WHEN reloading it and predicting
    loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)
    result = loaded_model.predict(
        pd.DataFrame({"query": ["What is the scale length of the Stratocaster?"]})
    )
    # THEN it answers using the fakes baked into the logged model, with no network access
    assert list(result) == ["The scale length is 25.5 in [stratocaster.md]."]


def test_reject_embedded_credentials_allows_none():
    # GIVEN no llm/embeddings instance at all
    # WHEN checking it for embedded credentials
    # THEN nothing is raised
    mlflow_model_module._reject_embedded_credentials(None)


def test_reject_embedded_credentials_allows_a_credential_free_fake(fake_chat_model):
    # GIVEN a fake chat model with no credential field
    # WHEN checking it for embedded credentials
    # THEN nothing is raised
    mlflow_model_module._reject_embedded_credentials(fake_chat_model)


class _ChatModelWithKey(BaseChatModel):
    """Stand-in for a real provider wrapper (e.g. `ChatOpenAI`) holding an API key."""

    api_key: SecretStr

    @property
    def _llm_type(self) -> str:
        return "fake-with-key"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        raise NotImplementedError


def test_reject_embedded_credentials_allows_an_empty_secret_field():
    # GIVEN an object exposing an unset SecretStr field
    model = _ChatModelWithKey(api_key=SecretStr(""))
    # WHEN checking it for embedded credentials
    # THEN nothing is raised
    mlflow_model_module._reject_embedded_credentials(model)


def test_reject_embedded_credentials_raises_for_a_populated_secret_field():
    # GIVEN an object exposing a populated SecretStr field, as real provider wrappers do
    model = _ChatModelWithKey(api_key=SecretStr("sk-real-key"))
    # WHEN checking it for embedded credentials
    # THEN a ValueError naming the offending field is raised
    with pytest.raises(ValueError, match="api_key"):
        mlflow_model_module._reject_embedded_credentials(model)


def test_guitar_assistant_model_init_rejects_an_llm_carrying_a_populated_secret_field():
    # GIVEN an llm exposing a populated SecretStr field
    llm = _ChatModelWithKey(api_key=SecretStr("sk-real-key"))
    # WHEN constructing a GuitarAssistantModel from it, regardless of how it would later be used
    # THEN construction refuses immediately, before the credential could ever be pickled
    with pytest.raises(ValueError, match="api_key"):
        GuitarAssistantModel(llm=llm)


def test_log_model_rejects_an_llm_carrying_a_populated_secret_field(tracking_uri, fake_embeddings):
    # GIVEN an llm exposing a populated SecretStr field
    llm = _ChatModelWithKey(api_key=SecretStr("sk-real-key"))
    # WHEN logging a model built from it
    # THEN log_model refuses, and no run artifact ever gets the credential pickled into it
    with pytest.raises(ValueError, match="api_key"):
        log_model(llm=llm, embeddings=fake_embeddings)


def test_log_model_bundles_the_package_source_via_code_paths(
    tracking_uri, fake_embeddings, fake_chat_model
):
    # GIVEN a model logged with injected fakes
    model_info = log_model(llm=fake_chat_model, embeddings=fake_embeddings)
    artifact_dir = Path(urlparse(mlflow.artifacts.download_artifacts(model_info.model_uri)).path)
    # The expected relative path is derived from the same `_CODE_PATH` constant `log_model`
    # itself uses, plus this module's real `__file__` — never a hardcoded literal — so a future
    # rename/move of `mlflow_model.py` needs no matching test change. MLflow nests each
    # code_paths entry under "code/<entry's own directory name>".
    expected_relative_path = (
        Path(mlflow_model_module.__file__).resolve().relative_to(mlflow_model_module._CODE_PATH)
    )
    # WHEN inspecting the logged artifact's bundled code
    bundled_module_path = (
        artifact_dir / "code" / mlflow_model_module._CODE_PATH.name / expected_relative_path
    )
    # THEN the package source that defines the pyfunc model is actually shipped in the
    # artifact, not just referenced by an argument that looks right
    assert bundled_module_path.exists()
