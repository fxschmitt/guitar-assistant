# Usage

## Prerequisites

- [`uv`](https://docs.astral.sh/uv/) — manages the Python interpreter, virtual
  environment, and dependencies:

  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```

  `uv` pins the interpreter ([.python-version](../.python-version)), resolves and
  locks dependencies (`pyproject.toml` + `uv.lock`) so every environment is
  reproducible, and is a lot faster than `pip` when iterating on the
  RAG/agent dependency stack.

## Installing dependencies

```bash
uv sync
```

Creates a `.venv/` and installs what's pinned in `uv.lock` (the package itself in
editable mode, plus `pytest` and `ruff` for dev tooling). Prefix any command with
`uv run` to execute it in this environment:

```bash
uv run pytest
uv run ruff check .
```

To add a dependency later:

```bash
uv add <package>          # runtime dependency
uv add --dev <package>    # dev-only dependency
```

This updates `pyproject.toml` + `uv.lock` in one step — commit both.

## Running ingestion

The default corpus (see "Running the CLI" below) is populated by a separate
console script, `guitar-assistant-ingest` (see
[architecture.md](architecture.md#wikipedia-ingestion-pipeline)), which walks
Wikipedia's electric-guitar category tree and writes chunked, embedded articles
into the persistent `.chroma/` store:

```bash
uv run guitar-assistant-ingest
```

Needs `OPENAI_API_KEY` (embeds the fetched articles) and `WIKIPEDIA_CONTACT_EMAIL`
(Wikipedia's API etiquette requires an identifying contact in every request's
`User-Agent`), both set the same way as `OPENAI_API_KEY` above:

```bash
echo "WIKIPEDIA_CONTACT_EMAIL=you@example.com" >> .env
```

Safe to re-run: a second run only re-processes articles whose Wikipedia revision
changed since the last run (tracked in `ingestion_manifest.json`, gitignored like
`.chroma/`), so a manual re-run or a periodic cron job both stay cheap.

## Running the CLI

The package installs a console script, `guitar-assistant`, that takes a question
and prints a grounded, cited answer:

```bash
uv run guitar-assistant "What is the scale length of the Stratocaster?"
```

Needs an OpenAI API key (used for both chat and embeddings), set via a `.env` file
in the repo root:

```bash
echo "OPENAI_API_KEY=sk-..." > .env
```

or exported directly:

```bash
export OPENAI_API_KEY=sk-...
```

By default, each invocation queries the persistent store populated by
`guitar-assistant-ingest` (see
[architecture.md](architecture.md#wikipedia-ingestion-pipeline)) and runs the
agent graph (see
[architecture.md](architecture.md#agent-graph-langgraph-per-query)) against your
question. This requires having run `guitar-assistant-ingest` at least once; the
available guitar models are derived from whatever is indexed in `.chroma/` at
query time. Running the CLI before any ingestion means querying an empty
corpus — no answers, just "all"-routed empty retrievals.

For a quick start with no ingestion step, set `GUITAR_ASSISTANT_CORPUS=demo` to
query the bundled Telecaster/Stratocaster/SG spec sheets instead, building a
fresh in-memory vector store on every invocation:

```bash
GUITAR_ASSISTANT_CORPUS=demo uv run guitar-assistant "What is the scale length of the Stratocaster?"
```

## Packaging

`guitar-assistant-package` logs the current agent as a new versioned MLflow model
artifact (see [architecture.md](architecture.md#mlflow-packaging)):

```bash
uv run guitar-assistant-package
```

Prints the resulting `model_uri`, and records the run under the `guitar-assistant`
experiment in the same `mlflow.db` the query CLI uses.

Logging is not the same as registering: `model_uri`
(`runs:/<run_id>/guitar_assistant`) only resolves within that run, with no stable
name and no version number. To promote a logged run to a named, versioned entry in
the MLflow Model Registry (`models:/guitar-assistant/<version>`), register it
explicitly once you've decided it's the one to keep, e.g. after a passing
evaluation run:

```python
import mlflow

mlflow.set_tracking_uri("sqlite:///mlflow.db")  # matches configure_default_tracking_uri()
mlflow.register_model(model_uri, name="guitar-assistant")
```

There is no console script for this step; it's a deliberate manual gate, not an
automated part of `guitar-assistant-package`.

## Exploration UI

The CLI rebuilds the vector store on every call, which is fine for a single
question but clumsy for probing the agent across several queries. `app.py` is a
small Streamlit app that builds the agent once per session and lets you chat with
it, showing the routed guitar model and retrieved spec sheet(s) alongside each
answer.

```bash
uv sync --group explore
uv run streamlit run app.py
```
