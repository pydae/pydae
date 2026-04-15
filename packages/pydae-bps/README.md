# pydae-bps

Balanced Power Systems builder for the [pydae](https://github.com/pydae/pydae) DAE solver.

Reads JSON/HJSON power system network descriptions (similar to PSS/E or PSAT format) and constructs the symbolic DAE system dictionary for `pydae.core.Builder`.

## Installation

```bash
pip install pydae-bps   # automatically installs pydae as a dependency
```

## Quick start

```python
from pydae.bps import BpsBuilder
from pydae.core import Builder, Model

grid = BpsBuilder('ieee39.json')
grid.construct('ieee39')

bld = Builder(grid.sys_dict, target='ctypes')
bld.build()

model = Model('ieee39')
model.ini({}, xy_0=1)
model.run(10.0, {'p_c_30': 0.8})
model.post()
```
