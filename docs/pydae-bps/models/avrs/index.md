# Automatic voltage regulators

Excitation-system models that regulate the field voltage of synchronous machines.

| Model | States | Notes |
|-------|--------|-------|
| [avr_1](avr_1.md) | 0 | Pure-algebraic, no limits, ini/run swap |
| [kundur](kundur.md) | 1 | Textbook proportional + sensor lag, hard limits |
| [kundur_tgr](kundur_tgr.md) | 2 | Kundur + transient gain reduction lead-lag |
| [sexs](sexs.md) | 2 | Simplified excitation system (lead-lag + exciter) |
| [sst1](sst1.md) | 3 | ST1-style with output filter on v_f |
| [st1](st1.md) | 2 | IEEE 421.5 ST1A, REE NTS parameter set |
| [st4b](st4b.md) | 4 | IEEE 421.5 ST4B dual-PI, REE NTS parameter set |
| [ntsst1](ntsst1.md) | 4 | NTS variant of ST1 (legacy, uses xi_v integrator) |
| [ntsst4](ntsst4.md) | 3 | NTS variant of ST4B (legacy) |
| [avr_antiw](avr_antiw.md) | — | Anti-windup AVR |

```{toctree}
:maxdepth: 1

avr_1
kundur
kundur_tgr
sexs
sst1
st1
st4b
avr_antiw
ntsst1
ntsst4
pss_kundur_2
smib_vsc_pq_inf
```
