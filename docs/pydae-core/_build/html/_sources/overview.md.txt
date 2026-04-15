# Overview

## What pydae-bps does

`pydae-bps` is a **network-to-DAE builder** for *balanced* (positive-sequence)
power systems. You describe the network in a JSON/HJSON file — buses, lines,
transformers, synchronous machines and their AVRs/governors/PSSs, VSCs,
VSGs, loads — and `pydae-bps` assembles the corresponding differential and
algebraic equations and hands them to `pydae.core.Builder` for compilation.

```{mermaid}
flowchart LR
    A[network.json] --> B[BpsBuilder]
    B --> C[sys_dict]
    C --> D[pydae.core.Builder]
    D --> E[Compiled model]
    E --> F[pydae.core.Model<br/>ini / run / post]
```

It is the successor to the older `bmapu` package and keeps compatible JSON
schemas while renaming the top-level entry point to `BpsBuilder`.

## What's in the package

The package is organised by element family:

| Subpackage        | Covers                                                      |
|-------------------|-------------------------------------------------------------|
| `bps.syns`        | Synchronous machines (Milano 2/3/4/6-order, SCIB, etc.)     |
| `bps.avrs`        | Automatic voltage regulators (SEXS, NTSST1, NTSST4, …)      |
| `bps.govs`        | Turbine governors (HYGOV, NTSIEEEG1, …)                     |
| `bps.vsgs`        | Virtual synchronous generators (LEON, REGFM-B1, uVSG, …)    |
| `bps.vscs`        | Voltage-source converters (grid-following/forming, BESS)    |
| `bps.vsc_ctrls`   | VSC outer-loop controls (PQ, VSG variants, uVSG)            |
| `bps.vsc_models`  | VSC inner-loop / physical models                             |
| `bps.wecs`        | Wind energy conversion systems (pitch, mechanical, PMSM)    |
| `bps.pvs`         | PV plants (dq current, string, steady-state, VRT)           |
| `bps.loads`       | ZIP loads                                                   |
| `bps.lines`       | Lines, line-DTR                                             |
| `bps.pods`        | Power oscillation dampers                                   |
| `bps.sources`     | Ideal voltage sources, GENAPE                               |
| `bps.pssdesigner` | PSS design helper                                           |

## Main object: `BpsBuilder`

```python
from pydae.bps import BpsBuilder

grid = BpsBuilder("my_network.json")
grid.checker()
grid.uz_jacs = False
grid.construct("my_system")   # emits sys_dict
```

The returned `grid.sys_dict` is the exact shape described in
[pydae-core overview](https://pydae-core.readthedocs.io/en/latest/overview.html), so the next step is the same as for a
hand-written system.

## Minimal network JSON

```json
{
  "system": {"name": "two_bus", "S_base": 100e6, "K_p_agc": 0.0, "K_i_agc": 0.0},
  "buses": [
    {"name": "1", "P_W": 0.0, "Q_var": 0.0, "U_kV": 20.0},
    {"name": "2", "P_W": 0.0, "Q_var": 0.0, "U_kV": 20.0}
  ],
  "lines":  [ ... ],
  "syns":   [ ... ],
  "avrs":   [ ... ],
  "govs":   [ ... ]
}
```

The full schema is documented in the element reference (see [API](api.md) —
to be expanded).

## Where to go next

- Hands-on build + simulation of a 2-bus Milano system:
  [Getting started](getting_started.md).
- Browse all elements: [API](api.md).
- Under the hood: [pydae-core overview](https://pydae-core.readthedocs.io/en/latest/overview.html).
