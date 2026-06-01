r"""
Four-wire breaker (or short series link) between two buses.

The breaker is modelled as a small series R+jX impedance per wire with a
boolean-like input `u_brk_{bus_1}` that scales the bus-side current
injections. Setting `u_brk = 0` disconnects both buses; `u_brk = 1` (default)
behaves as a permanent link with the small impedance.

**Per-phase branch equation** (one per phase $\varphi \in \{a, b, c, n\}$):

$$0 = v_{1,\varphi} - Z\, i_{\varphi} - v_{2,\varphi}, \qquad Z = R + jX$$

split into real form so it works on both SymPy and CasADi backends:

$$0 = v_{1,\varphi}^r - (R\,i_\varphi^r - X\,i_\varphi^i) - v_{2,\varphi}^r$$
$$0 = v_{1,\varphi}^i - (R\,i_\varphi^i + X\,i_\varphi^r) - v_{2,\varphi}^i$$

**Bus current injections** (added into the nodal `g` rows at each bus):

$$g_{\text{bus}_1, \varphi} \mathrel{{+}{=}} -u_{\text{brk}}\, i_\varphi$$
$$g_{\text{bus}_2, \varphi} \mathrel{{+}{=}} +u_{\text{brk}}\, i_\varphi$$

**HJSON snippet**

```hjson
breakers: [{bus_1: "A1", bus_2: "A2"}]
```
"""

import numpy as np


def descriptions():
    return [
        {"type": "Parameter", "tex": "R",      "data": "",  "model": "R_{bus_1}",      "default": 1e-4, "units": r"\Omega", "description": "Series resistance per wire"},
        {"type": "Parameter", "tex": "X",      "data": "",  "model": "X_{bus_1}",      "default": 1e-4, "units": r"\Omega", "description": "Series reactance per wire"},
        {"type": "Input",     "tex": "u_{brk}", "data": "", "model": "u_brk_{bus_1}", "default": 1.0,  "units": "-",       "description": "Switching state (0 = open, 1 = closed)"},
        {"type": "Algebraic State", "tex": r"i^r_\varphi", "data": "", "model": "i_brk_{bus_1}_{phase}_r", "default": "", "units": "A", "description": "Per-phase real branch current"},
        {"type": "Algebraic State", "tex": r"i^i_\varphi", "data": "", "model": "i_brk_{bus_1}_{phase}_i", "default": "", "units": "A", "description": "Per-phase imag branch current"},
        {"type": "Output",    "tex": r"|i_\varphi|", "data": "", "model": "i_brk_{bus}_{phase}_m", "default": "", "units": "A", "description": "Per-phase current magnitude (emitted on both buses)"},
    ]


def add_breakers(grid, data):
    '''
    4-wire breaker between bus_1 and bus_2 with a small series R+jX impedance
    and a switching input u_brk_{bus_1} that gates the bus-side current
    injections. Dual-backend via grid.backend.
    '''
    name1 = data['bus_1']
    name2 = data['bus_2']
    bk = grid.backend

    # parameters / inputs
    R     = bk.symbols(f'R_{name1}')
    X     = bk.symbols(f'X_{name1}')
    u_brk = bk.symbols(f'u_brk_{name1}')

    # node voltages on both sides (rectangular)
    v_1 = {}
    v_2 = {}
    for k in range(4):
        v_1[(k, 'r')] = bk.symbols(f'V_{name1}_{k}_r')
        v_1[(k, 'i')] = bk.symbols(f'V_{name1}_{k}_i')
        v_2[(k, 'r')] = bk.symbols(f'V_{name2}_{k}_r')
        v_2[(k, 'i')] = bk.symbols(f'V_{name2}_{k}_i')

    # breaker currents (per phase, rectangular) — these are referenced to bus_1
    ph_names = ['a', 'b', 'c', 'n']
    i_r = {ph: bk.symbols(f'i_brk_{name1}_{ph}_r') for ph in ph_names}
    i_i = {ph: bk.symbols(f'i_brk_{name1}_{ph}_i') for ph in ph_names}

    # branch equations: v_1 - Z*i - v_2 = 0   in real form
    # Z*i = (R+jX)(i_r+j*i_i) = (R*i_r - X*i_i) + j*(R*i_i + X*i_r)
    for k, ph in enumerate(ph_names):
        grid.dae['g'].append(v_1[(k, 'r')] - (R*i_r[ph] - X*i_i[ph]) - v_2[(k, 'r')])
    for k, ph in enumerate(ph_names):
        grid.dae['g'].append(v_1[(k, 'i')] - (R*i_i[ph] + X*i_r[ph]) - v_2[(k, 'i')])

    grid.dae['y_ini'] += [i_r[ph] for ph in ph_names] + [i_i[ph] for ph in ph_names]
    grid.dae['y_run'] += [i_r[ph] for ph in ph_names] + [i_i[ph] for ph in ph_names]

    # bus-side current injections (gated by u_brk) + magnitude monitors
    for ph in ph_names:
        idx_r1, idx_i1 = grid.node2idx(name1, ph)
        grid.dae['g'][idx_r1] += -i_r[ph]*u_brk
        grid.dae['g'][idx_i1] += -i_i[ph]*u_brk
        grid.dae['h_dict'][f'i_brk_{name1}_{ph}_m'] = (i_r[ph]**2 + i_i[ph]**2)**0.5

    for ph in ph_names:
        idx_r2, idx_i2 = grid.node2idx(name2, ph)
        grid.dae['g'][idx_r2] +=  i_r[ph]*u_brk
        grid.dae['g'][idx_i2] +=  i_i[ph]*u_brk
        grid.dae['h_dict'][f'i_brk_{name2}_{ph}_m'] = (i_r[ph]**2 + i_i[ph]**2)**0.5

    grid.dae['u_ini_dict'].update({f'u_brk_{name1}': 1.0})
    grid.dae['u_run_dict'].update({f'u_brk_{name1}': 1.0})
    grid.dae['params_dict'].update({f'R_{name1}': 1e-4, f'X_{name1}': 1e-4})
