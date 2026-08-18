# Testing & Validation Strategy

Four layers, each validating a different claim, at different cost/speed tiers.
Only unit tests run by default.

## Unit tests

`tests/test_*.py` mirror `src/guitar_assistant/*.py` one-to-one and test each
module in isolation — routing decisions, retrieval filtering, data loading, CLI
argument handling — against fakes (e.g. a deterministic keyword-based
`Embeddings` stand-in in `tests/conftest.py`) rather than real OpenAI APIs.
Network-free and fast, so they run on every `uv run pytest` and are what CI
would gate on.

## End-to-end test / MLflow evaluation

Unit tests can't validate whether the compiled agent answers real questions
correctly and quickly enough. `tests/test_end_to_end.py` runs the agent against
`task/test-questions.csv` (a golden dataset of 10 questions with expected answers
and evaluation criteria) through the real OpenAI-backed graph, using
`mlflow.genai.evaluate()` rather than a manual loop, and asserts:

- **Accuracy ≥ 8/10** — `Expected_Answer`/`Evaluation_Criteria` are free text, so
  grading uses `correctness` ([evaluation.py](../src/guitar_assistant/evaluation.py)),
  a custom `mlflow.genai` scorer wrapping an LLM-as-judge call (a second, cheap
  OpenAI call scoring the agent's actual answer) rather than exact match.
- **Latency < 10s per query** — read from `execution_duration` on each row of
  `EvaluationResult.result_df`, i.e. the trace `mlflow.genai.evaluate()` captures
  automatically for every `predict_fn` call.

`correctness` is a custom scorer rather than MLflow's builtin `Correctness` or
`Guidelines` judges: the golden dataset carries both a reference `expected_answer`
*and* a per-row `evaluation_criteria` rubric — a hybrid of what those two builtins
grade separately. Running both would cost two LLM calls per row for two disjoint
scores; `correctness` (via `grade_answer`) weighs both signals in one call, one
score. Prebuilt RAG-specific judges (`RetrievalGroundedness`, `RetrievalRelevance`)
were considered and skipped for the same reason noted in
[limitations.md](limitations.md): with 3 documents and near-zero retrieval
ambiguity, they'd check something already trivially true in this corpus.

Because this runs through `mlflow.genai.evaluate()`, the same invocation that
proves the Tier 1 accuracy/latency acceptance criteria also logs an MLflow
evaluation run — metrics (`correctness/mean`), per-row traces, and assessments —
fulfilling the Tier 3 "Comprehensive Evaluation Framework" bonus rather than being
a separate mechanism. Inspect it the same way as other runs/traces (see
[architecture.md](architecture.md#inspecting-agent-execution-logs)), under the
`guitar-assistant-evaluation` experiment.

Needs a real `OPENAI_API_KEY`, costs money, and is slow, so it's marked
`@pytest.mark.integration` and excluded by default
(`addopts = "-m 'not integration'"`):

```bash
uv run pytest -m integration tests/test_end_to_end.py
```

## Judge calibration test

The end-to-end test's accuracy claim is only as good as the judge behind
`correctness`. Passing 8/10 golden questions means nothing if the judge itself
is miscalibrated — e.g. rubber-stamping wrong answers (false positives) or
rejecting correct ones over minor wording differences (false negatives).
`tests/test_evaluation_integration.py` checks the judge directly, independent
of the agent or MLflow: it calls the real `grade_answer()` against 11
hand-crafted question/answer pairs with a known-correct pass/fail verdict,
covering both failure directions —

- correct answers phrased differently from the reference (paraphrase, extra
  detail, equivalent units) that a too-strict judge might fail, and
- plausible-looking wrong answers (wrong numbers, wrong units, inverted
  feature support, hedging non-answers, incomplete comparisons) that a
  too-lenient judge might pass.

Each case is its own parametrized test node, so one miscalibrated case doesn't
mask failures in the others. Same cost/network profile as the end-to-end test,
so it's also `@pytest.mark.integration` and excluded by default:

```bash
uv run pytest -m integration tests/test_evaluation_integration.py
```

## Wikipedia client integration test

`tests/test_wikipedia_client_integration.py` checks
[wikipedia_client.py](../src/guitar_assistant/wikipedia_client.py) against the
real Wikipedia API: that `walk_category` finds real article titles under
`Category:Electric guitars`, that walking from
`ELECTRIC_GUITARS_BY_MANUFACTURER_CATEGORY` at `max_depth=2` (the entry point
ingestion will actually use — `Category:Electric guitars` itself holds mostly
generic topic articles, not models) finds real guitar models rather than that
generic-topic noise, and that `fetch_wikitext` returns real infobox content for
a known article. No API key needed, but it hits the network, so it's
`@pytest.mark.integration` and excluded by default. Each test caps
`max_requests` (5-15), so a run can never wander far into the real category tree:

```bash
uv run pytest -m integration tests/test_wikipedia_client_integration.py
```

## Packaging test

MLflow doesn't play well with this repo's `uv`-managed `src/` layout out of the
box — the pickled model class needs its package source explicitly bundled via
`code_paths=[str(_CODE_PATH)]` in `log_model()`
([mlflow_model.py](../src/guitar_assistant/mlflow_model.py)), or the logged model
can't be reloaded anywhere `guitar_assistant` isn't already installed. Neither the
unit tests nor the end-to-end test would catch a regression here, since both
run in a dev environment that already has `guitar_assistant` importable regardless
of `code_paths`.

`test_log_model_bundles_the_package_source_via_code_paths` in
[test_mlflow_model.py](../tests/test_mlflow_model.py) validates that a logged
model is actually loadable, by checking the real source file lands inside the
artifact rather than just asserting `code_paths` looks right. Fast and
network-free, so it runs by default with the rest of the unit tests.
