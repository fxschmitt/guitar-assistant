"""End-to-end test: the real agent against the golden dataset (Tier 1 acceptance criteria).

Requires a real `OPENAI_API_KEY` and hits the network, so it's marked
`integration` and excluded from the default `uv run pytest` run. Run
explicitly with `uv run pytest -m integration tests/test_end_to_end.py`.

This runs the golden dataset through `mlflow.genai.evaluate()`, so the same
invocation that proves the Tier 1 accuracy/latency acceptance criteria also
produces a logged MLflow evaluation run (metrics, per-row traces/assessments)
— the Tier 3 evaluation-framework deliverable, not a separate mechanism. The run
is logged to this repo's default MLflow tracking store (`mlflow.db`), so it's
visible in the same `mlflow ui` a real `guitar-assistant` invocation logs to.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path
from typing import Final

import mlflow
import mlflow.genai
import pandas as pd
import pytest
from mlflow.pyfunc import PyFuncModel

from guitar_assistant.evaluation import GoldenQuestion, correctness, load_golden_dataset
from guitar_assistant.mlflow_model import configure_default_tracking_uri, log_model

# Dev/test-only data, not a runtime resource, so a relative path is fine here
# (unlike data.py, which uses importlib.resources for the packaged corpus).
_GOLDEN_DATASET_PATH: Final = Path(__file__).resolve().parent.parent / "task" / "test-questions.csv"
_MINIMUM_PASSING_ANSWERS: Final = 8
_MAXIMUM_LATENCY_SECONDS: Final = 10
_EXPERIMENT_NAME: Final = "guitar-assistant-evaluation"


@pytest.fixture(name="tracking_uri", scope="module")
def fixture_tracking_uri() -> Iterator[None]:
    # Points at this repo's real mlflow.db, not an isolated temp store, so this run
    # shows up in the same `mlflow ui` a real `guitar-assistant` invocation logs to.
    configure_default_tracking_uri()
    mlflow.set_experiment(_EXPERIMENT_NAME)
    # mlflow.genai.evaluate()'s default worker pool (10, matching this dataset's row
    # count) deadlocks, so limit it to 1 worker for this test.
    previous_max_workers = os.environ.get("MLFLOW_GENAI_EVAL_MAX_WORKERS")
    os.environ["MLFLOW_GENAI_EVAL_MAX_WORKERS"] = "1"
    yield
    mlflow.set_tracking_uri("")
    if previous_max_workers is None:
        del os.environ["MLFLOW_GENAI_EVAL_MAX_WORKERS"]
    else:
        os.environ["MLFLOW_GENAI_EVAL_MAX_WORKERS"] = previous_max_workers


@pytest.fixture(name="golden_dataset", scope="module")
def fixture_golden_dataset() -> list[GoldenQuestion]:
    return load_golden_dataset(_GOLDEN_DATASET_PATH)


@pytest.fixture(name="loaded_model", scope="module")
def fixture_loaded_model(tracking_uri) -> PyFuncModel:
    # Mirrors guitar_assistant/__init__.py's package() exactly (log_model with no
    # fakes injected), so this test exercises the real MLflow pyfunc packaging
    # path (code_paths, load_context, signature) a real deployment loads, not
    # just the agent graph.
    model_info = log_model()
    # log_model() calls mlflow.set_experiment("guitar-assistant") internally, which
    # would otherwise leave the active experiment pointed away from
    # _EXPERIMENT_NAME by the time this module's test calls
    # mlflow.genai.evaluate(). Restore it before returning.
    mlflow.set_experiment(_EXPERIMENT_NAME)
    return mlflow.pyfunc.load_model(model_info.model_uri)


@pytest.mark.integration
def test_agent_meets_tier1_accuracy_and_latency_acceptance_criteria(loaded_model, golden_dataset):
    # GIVEN the real OpenAI-backed, packaged MLflow model and the golden dataset of
    # 10 questions

    # `loaded_model.predict()` triggers LangChain autologging spans for each internal
    # LLM call (route, generate) but doesn't itself open a span, so without an
    # explicit root span those child spans have no common parent and
    # mlflow.genai.evaluate() can attribute a row's "request" to an arbitrary nested
    # span (e.g. the generate node's raw OpenAI payload) instead of this function's
    # `{"query": ...}` input. `@mlflow.trace` gives predict_fn its own root span so
    # every nested call is parented under it correctly, per the `predict_fn`
    # parameter docs on `mlflow.genai.evaluate`.
    @mlflow.trace
    def predict_fn(query: str) -> str:
        return loaded_model.predict(pd.DataFrame({"query": [query]}))[0]

    # WHEN evaluating it with the correctness scorer, which logs metrics,
    # per-row traces, and assessments to the current MLflow run
    result = mlflow.genai.evaluate(
        data=[question.to_scorer_inputs() for question in golden_dataset],
        predict_fn=predict_fn,
        scorers=[correctness],
    )

    assert result.result_df is not None, "evaluate() returned no per-row results."
    slow = [
        (row["request"]["query"], row["execution_duration"] / 1000)
        for _, row in result.result_df.iterrows()
        if row["execution_duration"] / 1000 >= _MAXIMUM_LATENCY_SECONDS
    ]
    failed = [
        (row["request"]["query"], row["correctness/rationale"])
        for _, row in result.result_df.iterrows()
        if not bool(row["correctness/value"])
    ]
    passed_count = len(golden_dataset) - len(failed)

    # THEN every query answers within 10s and at least 8/10 are graded correct
    assert not slow, f"Queries exceeding {_MAXIMUM_LATENCY_SECONDS}s: {slow}"
    assert passed_count >= _MINIMUM_PASSING_ANSWERS, (
        f"Only {passed_count}/{len(golden_dataset)} passed. Failures: {failed}"
    )
