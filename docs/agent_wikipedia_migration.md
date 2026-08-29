# Migrating the agent from the 3-document demo corpus to the Wikipedia corpus

Wikipedia ingestion (`docs/scaling_strategy.md` #1-#4) writes chunks into a
persistent Chroma store, but `agent.py`'s query-time wiring
(`guitar_assistant/__init__.py:main`, `mlflow_model.GuitarAssistantModel`) still
builds an in-memory store from the 3 hand-written spec sheets in `data.py`. This
document tracks the remaining work to wire the real, >100-document corpus into
the agent without breaking the existing demo path, and the concrete problem
(the router's structured-output schema) that motivates most of it.

## Remaining problem: the router doesn't scale past a handful of models

`route`'s structured-output schema is `Literal[*available_guitar_models, "all"]`
(`agent.py:_build_route_decision_schema`), and `available_guitar_models` is
inlined into the routing prompt as a comma-separated list
(`agent.py:_ROUTE_PROMPT`). At 3 models this is free. At 100+:

- every routing call's prompt balloons with the full model list, and
- classification accuracy over a 100+-way enum degrades, and
- the `"all"` fallback stops meaning "ambiguous between a couple of models" and
  starts meaning "searched unfiltered across the whole corpus" — a much weaker
  fallback.

`docs/scaling_strategy.md` #6 already specifies the fix (fuzzy/alias match
first, LLM only over a short vector-presearch shortlist); it just isn't
implemented yet. That's phase 2 below.


## Phase 1 — derive `available_guitar_models` from the persistent store, not from `load_documents()`

Both call sites that build the agent derive `available_guitar_models` from an
in-memory `documents` list:

- `guitar_assistant/__init__.py:main` — `load_documents()` then
  `build_vector_store(documents)`.
- `mlflow_model.py:GuitarAssistantModel.load_context` — same pattern.

That only works because the demo corpus is loaded fresh in the same process
that builds the vector store. Wikipedia ingestion is a separate, offline
process (`guitar-assistant-ingest`) — by query time there is no `documents`
list, only the persisted Chroma collection.

**Plan:**

1. Add a helper (`retriever.py` is the natural home, alongside
   `open_persistent_vector_store`) that returns the distinct `guitar_model`
   metadata values already indexed, e.g. via the underlying collection's
   `get(include=["metadatas"])` deduplicated client-side. This avoids depending
   on `IngestionManifest` at query time (the manifest is an ingestion-time
   bookkeeping detail, not a query-time API) and stays correct even if the
   manifest file is missing or stale relative to the store.
2. Wire `main()` and `GuitarAssistantModel.load_context` to call
   `open_persistent_vector_store()` + this new helper instead of
   `load_documents()` + `build_vector_store()`, gated behind how the corpus is
   selected (see "Keeping the demo path alive" below).
3. Unit-test the new helper against a `Chroma` instance seeded with a couple of
   fake documents (same pattern already used in `test_retriever.py`).

**Risk:** low. Purely additive; the demo path (`data.py`, `build_vector_store`)
is untouched.

## Phase 2 — two-stage routing (implements scaling_strategy.md #6)

Replace the single LLM structured-output call in `route` with:

1. **Fuzzy/alias match first**: match the query's text against the known
   `guitar_model` slugs (and, ideally, a small alias table — "strat" ->
   `fender_stratocaster`, "les paul" -> ...) using `rapidfuzz` (new dependency).
   No LLM call at all on a confident match — this is expected to cover most
   real queries, since people usually name a model directly.
2. **LLM fallback only on a miss**: run a cheap unfiltered vector pre-search to
   shortlist a handful of candidate `guitar_model` values, then build
   `_build_route_decision_schema` from *that* shortlist (plus `"all"`) instead
   of the full corpus. The schema and prompt stay small regardless of total
   corpus size.

**Plan:**

1. Add `rapidfuzz` to `pyproject.toml` dependencies.
2. Add a `_fuzzy_match_guitar_model` helper (module-level `_` function, unit
   tested with parametrized query/expected-model pairs) and a
   `_shortlist_candidates` helper backed by `vector_store.similarity_search`
   over `guitar_model` metadata only (no full document fetch needed for the
   shortlist step).
3. Change `route`'s signature so it takes the fuzzy matcher and a
   shortlist-then-classify callable instead of a single `classify` callable;
   `build_agent` wires both, closing over `available_guitar_models` for the
   fuzzy step and `vector_store` for the shortlist step.
4. Update `test_agent.py`'s routing tests to cover: a confident fuzzy match
   (no LLM call — assert the fake LLM was never invoked), a miss that falls
   through to the shortlist+LLM path, and the shortlist path correctly
   restricting the schema's valid values.

**Risk:** medium — this changes `route`'s public signature and is the one
phase most likely to need iteration on fuzzy-match thresholds. Treat as its
own change, not bundled with phase 1's plumbing fixes.

## Phase 3 — flip the default, close out the docs

Once phases 1-2 are tested end-to-end against the real persistent store:

1. Make the persistent Wikipedia store the default corpus for
   `main()`/`package()`. Keep `data.py`/`build_vector_store`/the 3 spec sheets
   as an explicit demo/test fixture (used by unit tests and
   `docs/usage.md`'s quick-start), not the default runtime path.
2. Fold `docs/scaling_strategy.md` #6 into `docs/architecture.md` as current
   routing behavior (per this repo's convention: a scaling-strategy item that's
   fully implemented gets described as current behavior, not left as a future
   plan).
3. Update `docs/usage.md` if the CLI gains a way to pick which corpus to query
   (e.g. a `--corpus demo|wikipedia` flag, or simply always querying the
   persistent store once ingestion has been run at least once).

## Deliberately out of scope

Carried over from `docs/scaling_strategy.md`'s own "what this plan doesn't
solve" section — not blockers for wiring the real corpus in:

- Cross-manufacturer multi-model fan-out (e.g. "compare the Telecaster and the
  SG") — needs `route` to resolve and retrieve across multiple models, not
  just one or "all".
- Variant-level granularity within one article (e.g. distinguishing a "Player
  Stratocaster" from an "American Ultra Stratocaster" on the same page).
