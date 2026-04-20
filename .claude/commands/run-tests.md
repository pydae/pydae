---
description: Run pydae's pytest suite, optionally filtered by marker
argument-hint: "[marker]  e.g. parse | symbolic | codegen | build | model | bps | uds"
---

Run the test suite. If `$ARGUMENTS` names a marker (`parse`, `symbolic`, `codegen`, `build`, `model`, `slow`, `bps`, `uds`), run only that marker. Otherwise, run the full suite.

Fast lane (no C compiler needed): `parse`, `symbolic`, `codegen`.
Compile + run lane (needs gcc / MSVC): `build`, `model`.

Steps:
1. If `$ARGUMENTS` is empty, run `uv run pytest -q`.
2. If `$ARGUMENTS` contains a single marker, run `uv run pytest -m "$ARGUMENTS" -q`.
3. If `$ARGUMENTS` looks like a path (contains `/` or ends in `.py`), treat it as a file selector: `uv run pytest "$ARGUMENTS" -q`.
4. Report pass/fail counts and, on failure, the first failing test's short traceback. Do not re-run.

Notes:
- Always use `uv run pytest`, never bare `pytest`.
- If the run fails because C compilation is missing, say so explicitly — don't retry with a different marker silently.
