# Turbine governors

Primary-frequency-control models for hydro, steam, and gas turbines.

| Model | Type | States | Notes |
|-------|------|--------|-------|
| [tgov1](tgov1.md) | Steam, first-order valve | 2 | PSS/E TGOV1 reference implementation |
| [tgov2](tgov2.md) | Steam, rate-limited valve | 2 | TGOV1 + explicit valve rate limits |
| [hygov](hygov.md) | Hydro, inelastic penstock | 4 | Dashpot + gate servo + water column |
| [ggov1](ggov1.md) | Gas / general, PI | 3 | Rate-limited fuel valve, turbine lead-lag |
| [dgov](dgov.md) | Diesel, proportional | 3 | No integral — droop only |
| [agov1](agov1.md) | Thermal + load-sharing | 3+2 | Two-lag turbine, MW integral, AGC port |
| [ieeeg1](ieeeg1.md) | Steam multi-stage | 4 | IEEE Type 1, HP+LP taps, REE NTS defaults |
| [ntsieeeg1](ntsieeeg1.md) | Steam multi-stage | 4 | NTS variant of ieeeg1 (no descriptions()) |

```{toctree}
:maxdepth: 1

tgov1
tgov2
hygov
ggov1
dgov
agov1
ieeeg1
ntsieeeg1
```
