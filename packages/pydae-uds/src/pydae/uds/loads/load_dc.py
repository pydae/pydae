# -*- coding: utf-8 -*-
r"""
Constant-power DC load between a bus's positive pole (node 0) and negative
pole (node 1).

**Algebraic equation** (one per DC load):

$$0 = i_p (v_p - v_n) - p$$

with $v_p, v_n$ the positive- and negative-pole voltages, $i_p$ the load
current entering at $v_p$, and $p$ the load active power (input).

**Bus current injections**:

$$g_{\text{bus}, \text{node}=0} \mathrel{{+}{=}} +i_p, \qquad
  g_{\text{bus}, \text{node}=1} \mathrel{{+}{=}} -i_p$$

**HJSON snippet**

```hjson
loads: [
    {bus: "D2", kW: 1.0, type: "DC", model: "ZIP"},
]
```

The `kW` field becomes the default `p_load_{bus}` input (in W) and can be
overridden at runtime via `model.ini({"p_load_D2": 5e3}, ...)`.
"""

import numpy as np  # noqa: F401 — keep namespace consistent across loads/*


def descriptions():
    return [
        {"type": "Input",          "tex": "p",     "data": "kW", "model": "p_load_{bus}",         "default": "kW*1e3", "units": "W",  "description": "Constant-power load setpoint"},
        {"type": "Algebraic State","tex": "i_p",   "data": "",   "model": "i_load_{bus}_p_r",     "default": "",       "units": "A",  "description": "Load current at the positive pole"},
        {"type": "Algebraic State","tex": "v_p",   "data": "",   "model": "V_{bus}_0_r",          "default": "",       "units": "V",  "description": "Positive-pole voltage (already in the nodal vector)"},
        {"type": "Algebraic State","tex": "v_n",   "data": "",   "model": "V_{bus}_1_r",          "default": "",       "units": "V",  "description": "Negative-pole voltage (already in the nodal vector)"},
    ]


def load_dc(grid,data):

    self = grid
    bk = grid.backend

    name = data['bus']
    v_p = bk.symbols(f'V_{name}_0_r')
    v_n = bk.symbols(f'V_{name}_1_r')

    i_p = bk.symbols(f'i_load_{name}_p_r')
    p   = bk.symbols(f'p_load_{name}')

    
    self.dae['g'] += [i_p*(v_p - v_n) - p]

    self.dae['y_ini'] += [i_p]
    self.dae['y_run'] += [i_p]

    i_n = -i_p

    idx_r,idx_i = self.node2idx(name,'a')
    self.dae['g'] [idx_r] += i_p
    self.dae['g'] [idx_i] += 0.0

    idx_r,idx_i = self.node2idx(name,'b')
    self.dae['g'] [idx_r] += i_n
    self.dae['g'] [idx_i] += 0.0

    self.dae['u_ini_dict'].update({f'p_load_{name}':data['kW']*1e3})
    self.dae['u_run_dict'].update({f'p_load_{name}':data['kW']*1e3}) 