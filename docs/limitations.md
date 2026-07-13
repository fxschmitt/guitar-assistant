# Limitations & Future Work

Design choices below are the right calls for a 3-document, hand-written demo
corpus and a single developer's `OPENAI_API_KEY`. The target here is a **hobby
chatbot covering every electric guitar model with a Wikipedia article** — on
the order of a few hundred to a few thousand documents, still run by one
person for personal use or a small audience. Most of the fixes below are
correspondingly modest — see [scaling_strategy.md](scaling_strategy.md) for
the concrete version sized to this goal.

- **Corpus is hand-written, not sourced from real Wikipedia articles.** This
  is the main gap to close for this goal, not a someday-maybe: the 3 spec
  sheets are condensed summaries, not the real thing. See scaling_strategy.md
  §1 (Wikipedia ingestion).
- **Whole-document chunking doesn't scale to real articles.** A hand-written
  spec sheet is a few hundred words; a real Wikipedia guitar article can run
  several pages with many sections (history, notable players, variants,
  specifications) — too long to embed and retrieve as a single chunk. See
  scaling_strategy.md §3 (chunking).
- **Model→document metadata is hardcoded, not data-driven.** `_SOURCE_DOCUMENTS`
  in [data.py](../src/guitar_assistant/data.py) is a manually-curated tuple mapping
  each of the 3 filenames to its `manufacturer`/`guitar_model`. That doesn't
  scale to "every model with a Wikipedia article" — nobody wants to hand-add a
  tuple entry per guitar — and a single Wikipedia article sometimes covers
  several closely related variants (e.g. reissues, signature models) under one
  page, which the current flat one-file-one-model assumption can't represent.
  See scaling_strategy.md §1 (Wikipedia ingestion) and §2 (metadata + vector
  store).
- **The router's fixed enum doesn't scale to hundreds of models.** `route`
  classifies each query into a closed enum built from every indexed model
  name; that's fine at 3 models but unwieldy at a few hundred — a huge
  structured-output enum inflates every routing call's prompt and tends to
  degrade classification accuracy. See scaling_strategy.md §6 (routing).
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
- **In-memory vector store rebuilds on every run.** Chroma's in-memory mode
  re-embeds the whole corpus on every process start — negligible for 3
  documents, but re-embedding a few thousand Wikipedia articles on every CLI
  invocation is slow and burns API budget for no reason. This is worth fixing
  even at this scale, not just at a much larger one. See scaling_strategy.md
  §2 (persistent vector store).
- **Corpus is bundled into the package.** The 3 spec sheets ship as package
  resources (`src/guitar_assistant/resources/`), so updating one means a new
  package release rather than a config/file change. At Wikipedia scale this
  also isn't practical — the ingestion pipeline needs to own the corpus,
  independent of package releases. See scaling_strategy.md §1 (Wikipedia
  ingestion) and §4 (refresh cadence).
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
