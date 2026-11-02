# pydae-emac

**Electric machines and their controllers for pydae.**

`pydae-emac` simulates, controls and helps develop electric machines and
drives. Machines and controllers are built as pydae DAE systems and solved by
pydae's own solvers; a multirate switched-drive solver covers what a
mean-value model cannot show (sampling, carrier, switching edges, dead time,
spatial ripple). Both read the same FEA look-up tables.

```{admonition} Install
:class: tip

    pip install pydae-emac

(brings `pydae` as a dependency automatically)
```

## Documentation sections

```{toctree}
:maxdepth: 2
:caption: Getting started

overview
getting_started
```

```{toctree}
:maxdepth: 1
:caption: Reference

api
```

## Related packages

- [`pydae-core`](https://pydae-core.readthedocs.io) — the DAE solver underneath.
- [`pydae-bps`](https://pydae-bps.readthedocs.io) — balanced power systems.
- [`pydae-uds`](https://pydae-uds.readthedocs.io) — unbalanced distribution systems.

## Indices

- {ref}`genindex`
- {ref}`modindex`
- {ref}`search`
