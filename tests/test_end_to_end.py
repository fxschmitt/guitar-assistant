"""End-to-end test: the real agent against a frozen real-Wikipedia corpus snapshot.

Requires a real `OPENAI_API_KEY` and hits the network (OpenAI only, not
Wikipedia), so it's marked `integration` and excluded from the default
`uv run pytest` run. Run explicitly with
`uv run pytest -m integration tests/test_end_to_end.py`.

This is a regression guard, not an acceptance test tied to an external brief:
its job is to catch the agent getting worse (routing, retrieval, or generation)
as it's extended, not to certify it against someone else's bar. That shapes two
design choices:

- **The corpus is a frozen fixture, not live Wikipedia or whatever happens to be
  locally ingested.** `tests/fixtures/wikipedia_eval_corpus.json` is a one-time
  snapshot of real, chunked Wikipedia article content (see
  `scripts/build_wikipedia_eval_fixture.py` for how it was built and how to
  deliberately refresh it) — committed so this test is reproducible across
  machines and over time, and so a run can never fail because an article's
  real content drifted out from under it rather than because the agent
  regressed. It's rebuilt into an ephemeral in-memory vector store (real
  OpenAI embeddings, no live Wikipedia fetch) on every run via
  `retriever.build_vector_store`.
- **The passing-answer bar is calibrated from an observed run, not picked in
  advance.** All 10 questions passed on the run this bar was set from; see
  `_MINIMUM_PASSING_ANSWERS`'s comment.

`tests/fixtures/wikipedia_golden_questions.csv` is real content too: single-model
questions only (cross-manufacturer comparison isn't a supported capability yet —
see docs/limitations.md), split between questions a direct fuzzy name match
should route (half the set) and questions worded to miss the fuzzy match and
exercise the shortlist+LLM routing fallback instead (the other half), with
expected answers verified against the fixture's actual article content.

This still runs through `mlflow.genai.evaluate()`, so the same invocation that
checks the accuracy/latency bars also produces a logged MLflow evaluation run
(metrics, per-row traces/assessments) you can inspect over time to see whether
the agent's measured performance is trending up or down as it's extended. The
run is logged to this repo's default MLflow tracking store (`mlflow.db`), so
it's visible in the same `mlflow ui` a real `guitar-assistant` invocation logs to.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Final

from langchain_core.documents import Document
import mlflow
import mlflow.genai
import pytest

from guitar_assistant.agent import build_agent
from guitar_assistant.evaluation import GoldenQuestion, correctness, load_golden_dataset
from guitar_assistant.mlflow_model import configure_default_tracking_uri
from guitar_assistant.retriever import build_vector_store

_FIXTURES_DIR: Final = Path(__file__).resolve().parent / "fixtures"
_CORPUS_SNAPSHOT_PATH: Final = _FIXTURES_DIR / "wikipedia_eval_corpus.json"
_GOLDEN_DATASET_PATH: Final = _FIXTURES_DIR / "wikipedia_golden_questions.csv"
# Calibrated from an actual run against this fixture (10/10 passed, max latency
# ~2s), not guessed in advance. Leaves room for one flaky LLM-judge/model call
# without masking a real regression; drop it further only if that one-off slack
# turns out not to be enough and the test flakes on an otherwise-unchanged agent.
_MINIMUM_PASSING_ANSWERS: Final = 9
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


@pytest.fixture(name="compiled_agent", scope="module")
def fixture_compiled_agent():
    # Real OpenAI embeddings/chat model throughout: this test's whole point is to
    # measure the real agent's real-world behavior, not a faked approximation of it.
    corpus_snapshot = json.loads(_CORPUS_SNAPSHOT_PATH.read_text(encoding="utf-8"))
    documents = [
        Document(page_content=chunk["page_content"], metadata=chunk["metadata"])
        for chunk in corpus_snapshot
    ]
    available_guitar_models = sorted({document.metadata["guitar_model"] for document in documents})
    vector_store = build_vector_store(documents)
    return build_agent(vector_store, available_guitar_models)


@pytest.mark.integration
def test_agent_meets_accuracy_and_latency_regression_bar(
    tracking_uri, compiled_agent, golden_dataset
):
    # GIVEN the real, OpenAI-backed agent built against the frozen Wikipedia corpus
    # snapshot, and its golden dataset of 10 questions

    # `compiled_agent.invoke()` triggers LangChain autologging spans for each internal
    # LLM call (route, generate) but doesn't itself open a span, so without an
    # explicit root span those child spans have no common parent and
    # mlflow.genai.evaluate() can attribute a row's "request" to an arbitrary nested
    # span (e.g. the generate node's raw OpenAI payload) instead of this function's
    # `{"query": ...}` input. `@mlflow.trace` gives predict_fn its own root span so
    # every nested call is parented under it correctly, per the `predict_fn`
    # parameter docs on `mlflow.genai.evaluate`.
    @mlflow.trace
    def predict_fn(query: str) -> str:
        return compiled_agent.invoke({"query": query})["answer"]

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

    # THEN every query answers within 10s and at least 9/10 are graded correct
    assert not slow, f"Queries exceeding {_MAXIMUM_LATENCY_SECONDS}s: {slow}"
    assert passed_count >= _MINIMUM_PASSING_ANSWERS, (
        f"Only {passed_count}/{len(golden_dataset)} passed. Failures: {failed}"
    )
