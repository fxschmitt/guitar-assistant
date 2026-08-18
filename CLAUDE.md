# Coding Guidelines

## Core Principles

- Readability, KISS, YAGNI.
- Single responsibility per class/function. Design for testability.
- Leave touched code better than you found it, but don't refactor unrelated code
  just because it's nearby.

## Style

- Google Python Style Guide + Google docstring format. English comments/docstrings.
- Max line length: 100.
- Docstring shapes in `[A,B,C,D]` style, period after arg/return descriptions.
  `__init__` args documented in the `__init__` docstring, not the class docstring.
- `"` double quotes default.
- No non-standard abbreviations in names — spell out.
- Comments: inline only when not self-explanatory; explain *why*, not *what*.
- Ordering: top-down by importance/flow. Classes at top of module. Within a class:
  `__init__` → public → private, in call order. Module-level `_helper` for static
  methods that don't use class state.

## Typing (Python 3.11+)

- Builtin generics (`list[str]`, `dict[str, int]`), `|` unions, `| None` not `Optional[]`.
- `collections.abc` for abstract containers; `typing` for `TypedDict`/`Literal`/`TypeVar`/`ParamSpec`.
- `from __future__ import annotations` once per file if forward refs are needed.
- Precise types over `Any`; `Any` is fine when a type genuinely can't be inferred.
- Don't annotate local variables when the type is obvious from the RHS — only
  annotate when genuinely ambiguous or initialized to `None`.
- Global `ALL_CAPS` constants: `Final`, omit the type param when inferable
  (`MY_CONST: Final = 42`).

## Correctness & Robustness

- No mutable default arguments — use `None`, assign inside the function body.
- No bare `except:` and no silently swallowed exceptions — handle specific,
  intentional exception types.
- No magic numbers/strings — use named constants or config values.
- No `# noqa` / `# type: ignore` except for genuine false positives — fix the
  underlying issue instead. Linter: `pylint` (required: score > 85%, per the
  challenge brief). Formatter: `ruff format`. Typing: `pyright`.
- Public attributes must leave the object in a consistent state after any write.

## Structure & Imports

- Absolute imports, no wildcards, no importing `_private` functions across modules.
- No globals. Module-level variables used only within one module: prefix `_`.
- `self._attr` private by default; `self.attr` only if public read+write;
  `@property` for read-only.
- Favor `dataclasses` (`frozen=True` for read-only). Composition over inheritance,
  no long inheritance chains.
- `pathlib.Path` over `os.path`; `/` operator and `Path` methods; `str(path)` only
  at external boundaries (subprocess calls, third-party APIs, etc.).
- `logging`, not `print`.
- CLIs: `click`, not `argparse`.
- Scripts: guard with `if __name__ == "__main__":`.
- Workarounds: mark with `FIXME` + date + author (PEP 350). Only introduce
  workarounds if truly necessary, and remove them once no longer required.
- No binary artifacts in the repository.

## Tests

- Every function/method needs a unit test, or a `_` private prefix (trivial
  getters/wrappers exempt).
- Unit = isolated; integration = cross-module interaction. Integration tests don't
  need to cover every input combination — unit tests handle that.
- Tests are independent, order-agnostic, no global state mutation.
- Source `src/guitar_assistant/.../file.py` → test `tests/.../test_file.py`.
- `pytest`, standalone test functions — **no test classes**. Expressive, descriptive
  names that convey what's being tested; tests double as usage examples.
- `pytest.mark.parametrize` for input/output combos; fixtures for shared setup,
  scoped as high as possible (`session` > `module` > `function`); always name
  fixtures explicitly via `@pytest.fixture(name="...")`.
- `GIVEN` / `WHEN` / `THEN` comments required in every test body; no other
  comments inside test bodies. Ordinary explanatory comments are fine on
  fixtures, module-level constants, and other code outside test bodies, same
  as `src/` — explain *why*, not *what*. No docstring on a test function
  unless it explains something the GIVEN/WHEN/THEN can't (e.g. why an edge
  case matters) — never one that just restates them.

  ```python
  def test_foo():
      # GIVEN two numbers to add
      a = 2
      b = 3
      # WHEN they are added together
      result = a + b
      # THEN the result should be their sum
      assert result == 5
  ```

## Docs

- Adding/removing/renaming a module under `src/guitar_assistant/` → update the
  "Package layout" list in `docs/architecture.md` in the same change.
- Adding a new `@pytest.mark.integration` test file → add a short section for
  it to `docs/testing.md`, mirroring the existing entries (what it covers, cost/
  network profile, run command).
- A change that fully implements a numbered item in `docs/scaling_strategy.md`
  → fold that item into `docs/architecture.md`/`docs/testing.md` as current
  behavior, not just leave it described as a future plan.
