# pydae

Core engine for solving and analyzing Differential-Algebraic Equation (DAE) systems.

pydae uses SymPy to symbolically derive Jacobians, translates them to C, and compiles them into shared libraries for fast numerical simulation via Newton-Raphson.

See the [main repository](https://github.com/pydae/pydae) for full documentation.

## Installation

```bash
pip install pydae
```

## Quick start

```python
from pydae.core import Builder, Model

builder = Builder(sys_dict, target='ctypes')
builder.build()

model = Model('my_system')
model.ini(params, xy_0=initial_guess)
model.run(t_end, inputs)
model.post()
```
