---
description: Build Sphinx docs for one or all pydae subprojects
argument-hint: "[subproject]  core | bps | uds | all  (default: all)"
---

Build the Sphinx HTML docs. `$ARGUMENTS` selects which subproject:

- `core` → `docs/pydae-core`
- `bps`  → `docs/pydae-bps`
- `uds`  → `docs/pydae-uds`
- empty or `all` → build all three in sequence

For each selected subproject, run:

```bash
uv run sphinx-build -b html docs/pydae-<name> docs/pydae-<name>/_build/html
```

Report:
- For each subproject, `build succeeded` vs the first warning/error.
- The output directory of each successful build.

Notes:
- Do not commit `_build/` or `_autosummary/` — both are gitignored.
- The default branch is `master`. If ReadTheDocs is in play and fails with `couldn't find remote ref refs/heads/main`, that's an RTD admin-panel issue, not a build bug — surface it rather than patching the repo.
- Autosummary pages are generated from each component's `descriptions()` function — if a new component is missing from the docs, check that `descriptions()` is defined and exported.
