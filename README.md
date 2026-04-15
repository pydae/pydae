# pydae

**Environment to solve and analyze Differential-Algebraic Equation (DAE) systems**

pydae combines SymPy symbolic computation with compiled C code (via ctypes/CFFI) to provide a fast, user-friendly DAE solver. It is oriented toward power systems analysis but can be used in any field requiring DAE solutions.

## Repository Structure

This is a **monorepo** containing three independent but related packages:

```
pydae/
├── pyproject.toml              ← uv workspace root (not a package)
├── uv.lock                     ← shared lockfile
├── packages/
│   ├── pydae-core/             ← Core DAE solver engine
│   │   ├── pyproject.toml      ← Published as "pydae" on PyPI
│   │   ├── src/pydae/core/
│   │   │   ├── builder/        ← Symbolic → C code pipeline
│   │   │   ├── solver/         ← C source (daesolver_dense, LAPACK)
│   │   │   ├── diagnostics/    ← Jacobian health checks
│   │   │   └── model_class.py  ← Runtime Model API
│   │   └── src/pydae/daesolver/  ← daesolver.h and daesolver.c source files
│   │
│   ├── pydae-bps/              ← Balanced Power Systems (was "bmapu")
│   │   ├── pyproject.toml      ← Published as "pydae-bps" on PyPI
│   │   └── src/pydae/bps/
│   │
│   └── pydae-uds/              ← Unbalanced Distribution Systems (was "urisi")
│       ├── pyproject.toml      ← Published as "pydae-uds" on PyPI
│       └── src/pydae/uds/
│
├── tests/
├── docs/
└── examples/
```

All three packages share the `pydae` **namespace** via Python's native namespace package mechanism (no `__init__.py` in `src/pydae/`).

## Installation

### For users

```bash
pip install pydae              # Core solver only
pip install pydae-bps          # + Balanced power systems builder
pip install pydae-uds          # + Unbalanced distribution systems builder
```

### For developers (full monorepo)

```bash
git clone https://github.com/pydae/pydae.git
cd pydae
uv sync --all-packages         # Installs everything in editable mode
```

## Quick Start

```python
import numpy as np
import sympy as sym
from pydae.core import Builder, Model

# 1. Define your DAE system symbolically
L, G, M, K_d = sym.symbols('L,G,M,K_d', real=True)
p_x, p_y, v_x, v_y = sym.symbols('p_x,p_y,v_x,v_y', real=True)
lam, f_x, theta = sym.symbols('lam,f_x,theta', real=True)

dp_x = v_x
dp_y = v_y
dv_x = (-2*p_x*lam + f_x - K_d*v_x) / M
dv_y = (-M*G - 2*p_y*lam - K_d*v_y) / M

g_1 = p_x**2 + p_y**2 - L**2 - lam*1e-6
g_2 = -theta + sym.atan2(p_x, -p_y)

sys_dict = {
    'name': 'pendulum',
    'params_dict': {'L': 5.21, 'G': 9.81, 'M': 10.0, 'K_d': 1e-3},
    'f_list': [dp_x, dp_y, dv_x, dv_y],
    'g_list': [g_1, g_2],
    'x_list': [p_x, p_y, v_x, v_y],
    'y_ini_list': [lam, f_x],
    'y_run_list': [lam, theta],
    'u_ini_dict': {'theta': np.deg2rad(5.0)},
    'u_run_dict': {'f_x': 0},
    'h_dict': {'E_p': M*G*(p_y+L), 'E_k': 0.5*M*(v_x**2+v_y**2)},
}

# 2. Build (generates and compiles C code)
bld = Builder(sys_dict, target='ctypes')
bld.build()

# 3. Simulate
model = Model('pendulum')
model.ini({'theta': np.deg2rad(10)}, xy_0={'p_x': 0.9, 'p_y': -5.1, 'lam': 0, 'f_x': 1})
model.run(1.0, {})
model.run(20.0, {'f_x': 0.0})
model.post()
```

## Migration Guide (from old structure)

| Old import | New import |
|---|---|
| `import pydae.build_cffi as db` | `from pydae.core import Builder` |
| `db.builder(sys_dict)` | `Builder(sys_dict)` |
| `from pydae.bmapu import bmapu_builder` | `from pydae.bps import BpsBuilder` |
| `from pydae.urisi import urisi_builder` | `from pydae.uds import UdsBuilder` |

## Development

```bash
# Run tests
uv run pytest

# Run tests for a specific package
uv run --package pydae pytest tests/core/

# Lint
uv run ruff check .

# Build a single package for PyPI
cd packages/pydae-core && uv build
```

## License

MIT
