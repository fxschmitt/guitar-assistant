# Limitations & Future Work

The target here is a **hobby chatbot covering every electric guitar model with
a Wikipedia article** — on the order of a few hundred to a few thousand
documents, still run by one person for personal use or a small audience. See
[scaling_strategy.md](scaling_strategy.md) for the concrete plan sized to this
goal, and [architecture.md](architecture.md) for what's implemented today: real
Wikipedia ingestion into a persistent vector store (§1/§2/§4), section-aware
chunking (§3), and two-stage fuzzy/LLM routing (§6) are all current behavior,
not future work. The gaps below are what's left.

- **No cross-manufacturer/multi-model comparison.** The router (`agent.route`)
  only ever resolves a query to one guitar model or to "all" (unfiltered
  search across every indexed chunk) — there's no multi-model fan-out. A query
  like "compare the Telecaster and the SG" routes to "all" rather than
  retrieving each named model and synthesizing a real comparison, which is a
  much weaker fallback once the corpus holds a few hundred models instead of 3.
  Variant-level granularity within one article (e.g. distinguishing a "Player
  Stratocaster" from an "American Ultra Stratocaster" mentioned on the same
  page) has the same gap: a whole article is tagged with one `guitar_model`
  slug. Both are deliberately out of scope for now — see
  [scaling_strategy.md](scaling_strategy.md#what-this-plan-deliberately-doesnt-solve).
- **Evaluation uses one custom correctness scorer, not MLflow's prebuilt
  RAG judges.** `RetrievalGroundedness`/`RetrievalRelevance` were considered for
  `evaluation.py`'s `mlflow.genai.evaluate()` run and skipped: retrieval is
  always filtered to the router's resolved model before similarity search runs
  (see [architecture.md](architecture.md#agent-graph-langgraph-per-query)), so
  ambiguity stays near-zero regardless of total corpus size — they'd score
  something already trivially true. Worth revisiting once within-model
  ambiguity becomes real (e.g. several similarly named signature models sharing
  one article, or variant-level granularity per scaling_strategy.md's "what
  this plan doesn't solve") — still a reasonable thing to defer for now, not an
  urgent gap.
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
