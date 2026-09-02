# Limitations & Future Work

The target here is a **hobby chatbot covering every electric guitar model with
a Wikipedia article** — on the order of a few hundred to a few thousand
documents, still run by one person for personal use or a small audience. See
[scaling_strategy.md](scaling_strategy.md) for the concrete plan sized to this
goal, and [architecture.md](architecture.md) for what's implemented today: real
Wikipedia ingestion into a persistent vector store (§1/§2/§4), section-aware
chunking (§3), and two-stage fuzzy/LLM routing (§6) are all current behavior,
not future work — the 3-document, hand-written corpus (`data.py`) is now only
an explicit demo/test fixture (`GUITAR_ASSISTANT_CORPUS=demo`), not the default
runtime path. The gaps below are what's left.

- **Evaluation uses one custom correctness scorer, not MLflow's prebuilt
  RAG judges.** `RetrievalGroundedness`/`RetrievalRelevance` were considered for
  `evaluation.py`'s `mlflow.genai.evaluate()` run and skipped: with 3 documents
  and near-zero retrieval ambiguity, they'd score something already trivially
  true in this corpus. Worth revisiting once the corpus grows and retrieval
  ambiguity becomes real (e.g. several similarly named signature models) —
  still a reasonable thing to defer for now, not an urgent gap.
- **No re-ranking step.** With 3 candidate chunks, raw similarity search is
  sufficient. At a few thousand documents, each query is still filtered down
  to one (or a handful of) named model(s) before similarity search runs, so
  the candidate set per query stays small — re-ranking is a "nice to have"
  here rather than a pressing need. Worth adding only if near-duplicate pages
  (e.g. a model and its reissue) turn out to confuse retrieval in practice.
- **Retries are fixed-count and in-process.** `with_retry`'s 3-attempt
  exponential backoff is fine for answering one query at a time, which is how
  this project is used. It would need to change if this ever ran as an
  always-on service handling many concurrent requests, but that's not the
  goal here.
- **Router errors aren't recoverable mid-graph.** A misclassified query falls
  back to "search everything," not a retry loop. That's a reasonable
  trade-off at this scale: a confidence threshold plus a retry/escalation
  path would add real complexity for a benefit that mostly matters once
  wrong-but-confident routing becomes common, which isn't the case here.
- **No access control.** Not applicable to this use case — a hobby chatbot
  answering public, already-public-domain guitar trivia has no documents that
  need restricting to specific users. Not addressed in scaling_strategy.md;
  kept here only to note it was considered and deliberately left out, not
  overlooked.
- **API key management is dev-grade.** A single `OPENAI_API_KEY` in a local
  `.env` file remains the right amount of ceremony for a project like this.
  The one thing worth adding if this became reachable by other people (e.g. a
  shared demo link) is a request-rate/budget cap so a burst of traffic can't
  run up an unexpected bill — see scaling_strategy.md §5. Anything beyond
  that (secret vaults, scheduled key rotation, a shared proxy/gateway in
  front of the API) solves problems this project doesn't have.
