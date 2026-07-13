"""Wrap the compiled LangGraph agent in an MLflow `pyfunc` model.

See the README's "MLflow packaging" section: `GuitarAssistantModel` exposes the
route -> retrieve -> generate agent through the standard `pyfunc` contract
(`load_context` / `predict`), and `log_model` records it as an MLflow run
with the LLM/embedding model identifiers as params and the chunking
strategy/corpus version as tags.
"""

from __future__ import annotations

import importlib.util
from importlib import metadata
from pathlib import Path
from typing import Final

import mlflow
import pandas as pd
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langgraph.graph.state import CompiledStateGraph
from mlflow import langchain as mlflow_langchain
from mlflow.models import ModelSignature
from mlflow.models.model import ModelInfo
from mlflow.pyfunc.model import PythonModel, PythonModelContext
from mlflow.types import ColSpec, Schema
from pydantic import SecretStr

from guitar_assistant.agent import CHAT_MODEL, build_agent
from guitar_assistant.data import load_documents
from guitar_assistant.retriever import EMBEDDING_MODEL, build_vector_store

_EXPERIMENT_NAME: Final = "guitar-assistant"
_ARTIFACT_PATH: Final = "guitar_assistant"
_CHUNKING_STRATEGY: Final = "whole-document"
_QUERY_COLUMN: Final = "query"

_SIGNATURE: Final = ModelSignature(
    inputs=Schema([ColSpec("string", _QUERY_COLUMN)]),
    outputs=Schema([ColSpec("string")]),
)


def _reject_embedded_credentials(model: BaseChatModel | Embeddings | None) -> None:
    """Raise if `model` carries a populated credential field.

    `GuitarAssistantModel` stores `llm`/`embeddings` on `self`, and `log_model` pickles the
    whole instance as-is into the MLflow artifact. Real LangChain provider wrappers (e.g.
    `ChatOpenAI`) hold their API key in a `pydantic.SecretStr` field, so constructing this
    model with one verbatim would leak that key into the artifact store. Test doubles (e.g.
    `FakeChatModel`) have no such field and pass through untouched.

    Args:
        model: The `llm` or `embeddings` instance passed to `GuitarAssistantModel`, or `None`.

    Raises:
        ValueError: `model` has a populated `pydantic.SecretStr` field.
    """
    if model is None:
        return
    for field_name, value in vars(model).items():
        if isinstance(value, SecretStr) and value.get_secret_value():
            raise ValueError(
                f"{type(model).__name__}.{field_name} holds a credential; log_model would "
                "pickle it into the MLflow artifact and leak it. Pass a credential-free "
                "test double, or omit llm/embeddings so load_context builds them lazily."
            )


def _find_package_dir() -> Path:
    """Locate the `guitar_assistant` package directory without importing the package itself.

    `guitar_assistant/__init__.py` imports this module, so an `import guitar_assistant` statement
    here would be a real circular import; `find_spec` only locates the package on disk.
    Anchored on the package's own `__init__.py`, not this module's `__file__`, so this stays
    correct no matter how deep `mlflow_model.py` itself is nested inside the package.
    """
    assert __package__, "mlflow_model.py must be imported as part of a package."
    package_spec = importlib.util.find_spec(__package__)
    assert package_spec is not None and package_spec.origin is not None, (
        f"Could not locate the {__package__!r} package on disk."
    )
    return Path(package_spec.origin).resolve().parent


_CODE_PATH: Final = _find_package_dir().parent  # `src/`, so `guitar_assistant` is importable there.
_REPO_ROOT: Final = _CODE_PATH.parent
_DEFAULT_TRACKING_URI: Final = f"sqlite:///{_REPO_ROOT / 'mlflow.db'}"


def configure_default_tracking_uri() -> None:
    """Point MLflow at this repo's `mlflow.db`, regardless of the caller's cwd.

    Without this, MLflow falls back to a store relative to the current working
    directory, so runs and traces fragment across wherever a command happens to be
    invoked from. Call this once at the start of a real run (the console script does).
    A SQLite database, not the filesystem store, because MLflow's filesystem tracking
    backend (e.g. `file://.../mlruns`) is in maintenance mode as of MLflow 3.x and
    silently refuses to record new runs/traces.
    Tests set their own tracking URI via fixtures and never call this, so test runs
    stay isolated from this repo's real `mlflow.db`.
    """
    mlflow.set_tracking_uri(_DEFAULT_TRACKING_URI)


class GuitarAssistantModel(PythonModel):  # pylint: disable=abstract-method
    """MLflow `pyfunc` wrapper around the route -> retrieve -> generate agent.

    `predict_stream` raises `NotImplementedError` in `PythonModel` as an optional hook, not a
    true `abc.abstractmethod`; this model only needs the standard `predict` contract.
    """

    def __init__(
        self, llm: BaseChatModel | None = None, embeddings: Embeddings | None = None
    ) -> None:
        """Store the model/embeddings overrides used later by `load_context`.

        Args:
            llm: Chat model used for routing and generation. Defaults to
                `None`, which builds OpenAI's `CHAT_MODEL` lazily in
                `load_context` (so the API key is read from the loading
                environment, never pickled into the artifact). Overridable
                for testing only, with a credential-free double: any
                instance passed here is pickled as-is into the MLflow
                artifact, so a real, credential-bearing model would leak
                its API key into the artifact store.
            embeddings: Embeddings model used to index the corpus. Defaults
                to `None`, which builds OpenAI's `EMBEDDING_MODEL` lazily in
                `load_context`. Overridable for testing only, under the same
                credential-leak caveat as `llm`.

        Raises:
            ValueError: `llm` or `embeddings` carries a populated credential field.
        """
        _reject_embedded_credentials(llm)
        _reject_embedded_credentials(embeddings)
        self._llm = llm
        self._embeddings = embeddings
        self._agent: CompiledStateGraph | None = None

    def load_context(self, context: PythonModelContext) -> None:
        """Build the vector store and compile the agent graph.

        Args:
            context: MLflow's model context. Unused: the corpus ships inside
                the `guitar_assistant` package, so there are no external
                artifacts to resolve.
        """
        mlflow_langchain.autolog()
        documents = load_documents()
        available_guitar_models = sorted(
            {document.metadata["guitar_model"] for document in documents}
        )
        vector_store = build_vector_store(documents, embeddings=self._embeddings)
        self._agent = build_agent(vector_store, available_guitar_models, llm=self._llm)

    # `PythonModel.predict` carries no return annotation, so pyright infers `None` from its
    # docstring-only body; every real override (including mlflow's own examples) returns
    # actual predictions, so this mismatch is a stub artifact, not a real incompatibility.
    def predict(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, context: PythonModelContext, model_input: pd.DataFrame, params: dict | None = None
    ) -> list[str]:
        """Answer each query in `model_input`.

        Args:
            context: MLflow's model context. Unused.
            model_input: A DataFrame with one `"query"` column, one row per
                question, matching this model's signature.
            params: Unused; accepted for interface compatibility with
                `PythonModel.predict`.

        Returns:
            One grounded, cited answer per input row, in row order.
        """
        if self._agent is None:
            raise RuntimeError("load_context must be called before predict.")
        return [
            self._agent.invoke({"query": query})["answer"] for query in model_input[_QUERY_COLUMN]
        ]


def log_model(
    run_name: str | None = None,
    llm: BaseChatModel | None = None,
    embeddings: Embeddings | None = None,
) -> ModelInfo:
    """Log a fresh `GuitarAssistantModel` as an MLflow run.

    Args:
        run_name: Optional display name for the run. Defaults to MLflow's
            auto-generated name.
        llm: Forwarded to `GuitarAssistantModel`. Defaults to `None` (OpenAI's
            `CHAT_MODEL`, built when the logged model is loaded). Overridable
            for testing without network access; never pass a real,
            credential-bearing model (`GuitarAssistantModel` rejects one).
        embeddings: Forwarded to `GuitarAssistantModel`. Defaults to `None`
            (OpenAI's `EMBEDDING_MODEL`, built when the logged model is
            loaded). Overridable for testing without network access; same
            credential caveat as `llm`.

    Returns:
        Metadata (including the artifact URI) for the logged model.

    Raises:
        ValueError: `llm` or `embeddings` carries a populated credential field.
    """
    mlflow.set_experiment(_EXPERIMENT_NAME)
    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("chat_model", CHAT_MODEL)
        mlflow.log_param("embedding_model", EMBEDDING_MODEL)
        mlflow.set_tag("chunking_strategy", _CHUNKING_STRATEGY)
        mlflow.set_tag("corpus_version", metadata.version("guitar-assistant"))
        return mlflow.pyfunc.log_model(
            name=_ARTIFACT_PATH,
            python_model=GuitarAssistantModel(llm=llm, embeddings=embeddings),
            # MLflow adds each code_paths entry to sys.path at load time, and the pickled
            # class needs `guitar_assistant` importable as a top-level package there.
            code_paths=[str(_CODE_PATH)],
            signature=_SIGNATURE,
            input_example=pd.DataFrame(
                {_QUERY_COLUMN: ["What is the scale length of the Stratocaster?"]}
            ),
        )
