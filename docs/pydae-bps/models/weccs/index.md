# WECC Converters (weccs)

WECC-compatible renewable energy generator/converter models.
These models implement the grid-interface layer of the three-module WECC
renewable plant stack and are declared under the ``weccs`` key in the network
HJSON file.

The stack layers, from the network upward:

```
Network
  └── REGC_A  — grid-interface converter            (weccs)
        └── REEC_B  — local electrical controls     (nested via reec key)
              └── REPC_A  — plant controller         (ppcs)
```

``REGC_A`` is always required. ``REEC_B`` is added by including a ``reec``
sub-dict in the ``weccs`` entry, exactly as an AVR is nested inside a ``syns``
entry.

```{toctree}
:maxdepth: 1

regc_a
regc_b
reec_b
regfm_a1
regfm_b1
```
