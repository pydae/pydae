# -*- coding: utf-8 -*-
r"""
Single-node shunt admittance attached to one node of one bus.

The shunt is specified in `(R, X)` per-Ω form and converted internally to a
nodal-admittance contribution. Only one bus node is referenced (`bus_nodes[0]`);
the implicit return is system ground.

**Admittance**

$$Z = R + j X$$
$$Y_{jk} = \frac{1}{Z} = g_{jk} + j\, b_{jk}$$

The conductance and susceptance enter the branch primitive matrices at the
shunt's branch index:

$$G_{\text{primitive}}[i, i] = g_{jk}, \qquad
  B_{\text{primitive}}[i, i] = b_{jk}$$

and the bus-to-branch incidence row marks the connected node with `+1`.
The shunt therefore adds the admittance `g_{jk} + j b_{jk}` between the
selected node and ground via the standard nodal-admittance assembly
performed by `UdsBuilder.contruct_grid()`.

**HJSON snippet**

```hjson
shunts: [
    {bus: "A1", R: 3.0, X: 0.0, bus_nodes: [3, 0]},
]
```

`bus_nodes[0]` is the connected node index (here `3`, the neutral); the
second entry is kept for backward compatibility but currently ignored
(ground is always the return path).
"""

import numpy as np  # noqa: F401 — used by callers via grid.backend


def descriptions():
    """Single source of truth for shunt parameters / outputs.

    The shunt has no states or inputs; only two derived parameters per shunt
    (`g_shunt_{bus}_{node}` and `b_shunt_{bus}_{node}`) plus the HJSON inputs
    (`R`, `X`, `bus_nodes`) that produce them.
    """
    return [
        {"type": "Parameter", "tex": "R",       "data": "R",         "model": "",                          "default": "",    "units": r"\Omega",  "description": "Shunt resistance (HJSON field)"},
        {"type": "Parameter", "tex": "X",       "data": "X",         "model": "",                          "default": "",    "units": r"\Omega",  "description": "Shunt reactance (HJSON field)"},
        {"type": "Parameter", "tex": "g_{jk}",  "data": "",          "model": "g_shunt_{bus}_{node}",      "default": "1/R", "units": "S",        "description": r"Derived conductance, $\Re(1/(R+jX))$"},
        {"type": "Parameter", "tex": "b_{jk}",  "data": "",          "model": "b_shunt_{bus}_{node}",      "default": "-X/(R^2+X^2)", "units": "S", "description": r"Derived susceptance, $\Im(1/(R+jX))$"},
        {"type": "Parameter", "tex": "n",       "data": "bus_nodes", "model": "",                          "default": "",    "units": "-",        "description": "Connected node index on the bus (HJSON field)"},
    ]


def add_shunts(self):

    for shunt in self.shunts:
        node_j_str = str(shunt['bus_nodes'][0])
        node_j = '{:s}.{:s}'.format(shunt['bus'], node_j_str)
        col = self.nodes_list.index(node_j)           
        row_j = self.it_branch
        self.A[row_j,col] = 1
        
        #node_k_str = str(shunt['bus_nodes'][1])
        #if not node_k_str == '0': # when connected to ground
        #    node_k = '{:s}.{:s}'.format(shunt['bus'], str(shunt['bus_nodes'][1]))
        #    row_k = self.nodes_list.index(node_k)            
        #    self.A[row_k,col] = -1
        shunt_name = f"shunt_{shunt['bus']}_{node_j_str}"
        g_jk = self.backend.symbols(f"g_{shunt_name}")
        b_jk = self.backend.symbols(f"b_{shunt_name}")
        self.G_primitive[self.it_branch,self.it_branch] = g_jk
        self.B_primitive[self.it_branch,self.it_branch] = b_jk

        Z = shunt['R'] + 1j*shunt['X']
        Y = 1/Z
        self.dae['params_dict'].update({str(g_jk):Y.real})
        self.dae['params_dict'].update({str(b_jk):Y.imag})
       
        self.it_branch += 1



def shunts_preprocess(self):

    for shunt in self.shunts:
        
        self.N_branches += 1

        