# pydae-uds

**Unbalanced distribution systems builder for pydae.**

`pydae-uds` targets *three-phase, four-wire, unbalanced* distribution networks
and produces a `sys_dict` suitable for the `pydae.core.Builder`. Power
electronics — VSCs, BESS, PV, fuel cells, grid-forming VSGs — are first-class
citizens alongside lines, transformers and loads.

```{admonition} Install
:class: tip

    pip install pydae-uds

(brings `pydae` as a dependency automatically)
```

```{admonition} Historical note
:class: note

`pydae-uds` was previously called **`urisi`**. If you have old code, replace
`from pydae.urisi.urisi_builder import urisi` with
`from pydae.uds import UdsBuilder`.
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

- [`pydae-core`](https://pydae-core.readthedocs.io) — underlying DAE solver.
- [`pydae-bps`](https://pydae-bps.readthedocs.io) — balanced
  (positive-sequence) transmission-level builder on the same solver.

## Indices

- {ref}`genindex`
- {ref}`modindex`
- {ref}`search`
