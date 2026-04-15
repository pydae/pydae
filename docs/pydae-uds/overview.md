# Overview

## What pydae-uds does

`pydae-uds` assembles DAE models for distribution networks in their **native
unbalanced three-phase form** (phases a, b, c plus the neutral conductor). A
description of buses, lines, transformers, loads, and power-electronic
converters is compiled into a `sys_dict` that `pydae.core.Builder` then turns
into a fast simulation binary.

```{mermaid}
flowchart LR
    A[network.json / .hjson] --> B[UdsBuilder]
    B --> C[sys_dict]
    C --> D[pydae.core.Builder]
    D --> E[Compiled model]
    E --> F[pydae.core.Model<br/>ini / run / post]
```

It is the successor to the older `urisi` package; the JSON schema is
compatible, only the top-level class name changed to `UdsBuilder`.

## What's in the package

| Subpackage            | Covers                                                    |
|-----------------------|-----------------------------------------------------------|
| `uds.lines`           | Three-phase line models, including DTR-aware variants     |
| `uds.transformers`    | Dyn11, Dyg11, and related 3-phase transformer connections |
| `uds.loads`           | Unbalanced loads                                          |
| `uds.sources`         | Ideal 3-phase sources, DC sources                         |
| `uds.vscs`            | Grid-forming / grid-following VSCs (3-phase, 4-wire)      |
| `uds.vsc_ctrls`       | VSC outer-loop controls (PQ, VSG, GFPI, …)                |
| `uds.vsgs`            | Virtual synchronous generators (`gfpizv`, `gflpfzv`, …)   |
| `uds.ess`             | Battery energy storage systems (BESS) — DC/DC and DC/AC   |
| `uds.fcs`             | Fuel cells (SOFC, PEMFC) and DC/AC stages                 |
| `uds.pvs`             | PV plants with MPPT, DC/DC, DC/AC                         |
| `uds.miscellaneous`   | Breakers, fault models, utilities                         |
| `uds.utils`           | Reporting helpers (`report_v`, `get_v`, `get_i`, …)       |

## Main object: `UdsBuilder`

```python
from pydae.uds import UdsBuilder

grid = UdsBuilder("feeder.json")
grid.checker()
grid.construct("my_feeder")
```

`grid.sys_dict` is passed to the core builder exactly the same way as for
balanced systems; see [pydae-core overview](https://pydae-core.readthedocs.io/en/latest/overview.html).

## Minimal network JSON

```json
{
  "system": {"name": "feeder_test", "S_base": 100e3},
  "buses": [
    {"name": "A", "U_kV": 0.4, "U_kV_n": 0.4},
    {"name": "B", "U_kV": 0.4, "U_kV_n": 0.4}
  ],
  "transformers": [ ... ],
  "lines":        [ ... ],
  "loads":        [ ... ],
  "vscs":         [ ... ]
}
```

The full schema will be published in the [API](api.md) section.

## Where to go next

- Build and simulate a minimal 3-phase feeder:
  [Getting started](getting_started.md).
- Element reference: [API](api.md).
- Solver internals: [pydae-core overview](https://pydae-core.readthedocs.io/en/latest/overview.html).
