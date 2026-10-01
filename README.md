# guitar-assistant

A multi-tool AI agent that answers questions about electric guitar models,
retrieving from a corpus ingested from every electric guitar model's Wikipedia
article. Built with LangChain/LangGraph as a routed retrieve-and-generate
pipeline, packaged as an installable Python package and tracked with MLflow.

## Quickstart

```bash
uv sync
echo "OPENAI_API_KEY=sk-..." > .env
echo "WIKIPEDIA_CONTACT_EMAIL=you@example.com" >> .env
uv run guitar-assistant-ingest   # populates the persistent corpus; run once
uv run guitar-assistant "What is the scale length of the Stratocaster?"
```

See [docs/usage.md](docs/usage.md) for setup details, the Streamlit exploration UI,
and dependency management with `uv`.

## Docs

- [docs/usage.md](docs/usage.md) — installation, running the CLI, exploration UI.
- [docs/architecture.md](docs/architecture.md) — indexing pipeline, agent graph,
  MLflow packaging, evaluation, package layout.
- [docs/testing.md](docs/testing.md) — unit, end-to-end/MLflow-evaluation,
  judge-calibration, and packaging-isolation tests.
- [docs/limitations.md](docs/limitations.md) — known limitations and what a larger
  deployment would need.
- [docs/scaling_strategy.md](docs/scaling_strategy.md) — what needs to be done to scale this a several hundred real wikipedia pages on guitars.

## Corpus

`guitar-assistant` queries a persistent corpus built by `guitar-assistant-ingest`
from every Wikipedia article with an `Infobox Guitar model` template
(manufacturer, specs, body/neck wood, pickups, etc.), chunked by section and
re-embedded only when an article's revision changes — see
[docs/architecture.md](docs/architecture.md#wikipedia-ingestion-pipeline).
Content comes directly from Wikipedia; this project is not affiliated with,
endorsed by, or sourced from any guitar manufacturer.
