# guitar-assistant

A multi-tool AI agent that answers questions about Fender/Gibson electric guitar
specifications. Built with LangChain/LangGraph as a routed retrieve-and-generate
pipeline over a small spec-sheet corpus, packaged as an installable Python package and
tracked with MLflow.

## Quickstart

```bash
uv sync
echo "OPENAI_API_KEY=sk-..." > .env
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

The three guitar spec sheets in `src/guitar_assistant/resources/` are condensed,
hand-written summaries covering each model's well-known, publicly documented
specifications (body/neck wood, pickups, scale length, bridge type, etc.) — not
scraped or paraphrased from any single source. This is a demonstration corpus for a
generic RAG/agent architecture; it is not affiliated with, endorsed by, or sourced
from Fender Musical Instruments Corporation or Gibson Brands, Inc.
