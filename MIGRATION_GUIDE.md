# Migration Guide: pydae restructuring

This document describes the step-by-step process for migrating from the
old flat `pydae` repository to the new monorepo with namespace packages.

## Overview of changes

| Aspect | Old | New |
|--------|-----|-----|
| Repository layout | Single flat package | uv workspace monorepo |
| Build backend | flit_core | hatchling |
| Package manager | pip | uv (recommended) or pip |
| Core import | `from pydae.builder.core import Builder` | `from pydae.core import Builder` |
| bmapu import | `from pydae.bmapu import bmapu_builder` | `from pydae.bps import BpsBuilder` |
| urisi import | `from pydae.urisi import urisi_builder` | `from pydae.uds import UdsBuilder` |
| PyPI packages | 1 (`pydae`) | 3 (`pydae`, `pydae-bps`, `pydae-uds`) |

## Step-by-step migration procedure

### Phase 1: Scaffold (done)

The new directory structure is already created. The key architectural
decisions are:

1. **Native namespace packages** — The `src/pydae/` directory in each
   package has NO `__init__.py`. Python's import system automatically
   merges them into one `pydae` namespace.

2. **Core lives in `pydae.core`** — Not directly in `pydae/`. This
   avoids the namespace conflict where one package's `__init__.py`
   would block the others.

3. **uv workspaces** — A single `uv.lock` at the root, shared virtual
   environment, and `workspace = true` source references between
   packages.

### Phase 2: Move core code (partially done)

The following files have been moved and their imports updated:

```
OLD LOCATION                          → NEW LOCATION
src/pydae/builder/core.py             → packages/pydae-core/src/pydae/core/builder/core.py
src/pydae/builder/parser.py           → packages/pydae-core/src/pydae/core/builder/parser.py
src/pydae/builder/symbolic.py         → packages/pydae-core/src/pydae/core/builder/symbolic.py
src/pydae/builder/codegen/cffi_builder.py   → .../core/builder/codegen/cffi_builder.py
src/pydae/builder/codegen/ctypes_builder.py → .../core/builder/codegen/ctypes_builder.py
src/pydae/builder/dae_check.py        → packages/pydae-core/src/pydae/core/diagnostics/dae_check.py
src/pydae/builder/daesolver_dense.c   → packages/pydae-core/src/pydae/core/solver/daesolver_dense.c
src/pydae/builder/daesolver_dense.h   → packages/pydae-core/src/pydae/core/solver/daesolver_dense.h
src/pydae/model_class.py              → packages/pydae-core/src/pydae/core/model_class.py
```

### Phase 3: Move bmapu → pydae-bps (TODO)

1. Copy the contents of `src/pydae/bmapu/` into
   `packages/pydae-bps/src/pydae/bps/`

2. Rename the main class: `bmapu` → `BpsBuilder`

3. Update all internal imports:
   - `from pydae.bmapu.xxx import yyy` → `from pydae.bps.xxx import yyy`
   - `from pydae.builder.core import Builder` → `from pydae.core import Builder`

4. Move component model files (generators, loads, transformers) into
   `pydae/bps/components/`

5. Move utility functions (`lines.py`, etc.) into `pydae/bps/utils/`

### Phase 4: Move urisi → pydae-uds (TODO)

Same procedure as Phase 3 but for the urisi code.

### Phase 5: Update all notebooks and examples

Search and replace across your example notebooks:

```python
# Old
import pydae.build_cffi as db
bldr = db.builder(sys_dict)

# New
from pydae.core import Builder
bldr = Builder(sys_dict, target='cffi')  # or target='ctypes'
```

```python
# Old
from pydae.bmapu import bmapu_builder
grid = bmapu_builder.bmapu('network.json')

# New
from pydae.bps import BpsBuilder
grid = BpsBuilder('network.json')
```

### Phase 6: Deprecation shims (optional)

If you want a graceful transition period, you can add compatibility
shims that issue deprecation warnings:

```python
# packages/pydae-core/src/pydae/core/compat.py
import warnings

def builder(sys_dict, **kwargs):
    warnings.warn(
        "pydae.build_cffi.builder() is deprecated. "
        "Use: from pydae.core import Builder",
        DeprecationWarning, stacklevel=2
    )
    from pydae.core import Builder
    return Builder(sys_dict, **kwargs)
```

### Phase 7: Publish to PyPI

```bash
# Build each package independently
cd packages/pydae-core && uv build
cd packages/pydae-bps && uv build
cd packages/pydae-uds && uv build

# Upload (use twine or uv publish)
uv publish packages/pydae-core/dist/*
uv publish packages/pydae-bps/dist/*
uv publish packages/pydae-uds/dist/*
```

## Import reference

After migration, these are the canonical imports:

```python
# Core (always needed)
from pydae.core import Builder, Model
from pydae.core.diagnostics import diagnose_dae_model

# Balanced power systems (install pydae-bps)
from pydae.bps import BpsBuilder

# Unbalanced distribution systems (install pydae-uds)
from pydae.uds import UdsBuilder
```

## Namespace package rules (critical)

1. **NEVER** add `__init__.py` to `packages/*/src/pydae/`. This would
   break the namespace and make the other sub-packages unimportable.

2. Each sub-package (`core/`, `bps/`, `uds/`) MUST have its own
   `__init__.py`.

3. When developing locally, always use `uv sync --all-packages` to
   install all packages in editable mode. This ensures the namespace
   merging works correctly.
