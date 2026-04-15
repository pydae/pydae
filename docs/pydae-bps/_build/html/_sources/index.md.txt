# pydae-bps

**Balanced power systems builder for pydae.**

`pydae-bps` turns a JSON/HJSON description of a transmission-level power
network — buses, lines, synchronous generators, AVRs, governors, VSCs,
controls — into a `sys_dict` ready for the `pydae.core.Builder`. You write the
network, not the equations.

```{admonition} Install
:class: tip

    pip install pydae-bps

(brings `pydae` as a dependency automatically)
```

```{admonition} Historical note
:class: note

`pydae-bps` was previously called **`bmapu`**. If you have old code, replace
`from pydae.bmapu import bmapu_builder` with `from pydae.bps import BpsBuilder`.
```

## Documentation sections

```{toctree}
:maxdepth: 2
:caption: Getting started

overview
getting_started
```

```{toctree}
:maxdepth: 2
:caption: Models

models/index
```

```{toctree}
:maxdepth: 1
:caption: Reference

api
```

## Related packages

- [`pydae-core`](https://pydae-core.readthedocs.io) — the underlying DAE
  solver.
- [`pydae-uds`](https://pydae-uds.readthedocs.io) — three-phase *unbalanced*
  distribution-system builder for the same solver.

## Indices

- {ref}`genindex`
- {ref}`modindex`
- {ref}`search`
