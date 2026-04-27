# Synchronous machines

Salient- and round-pole synchronous machine models of varying order.

| Model | Order | States | Subtransient repr. | Saturation | Notes |
|-------|-------|--------|--------------------|-----------|-------|
| [milano2ord](milano2ord.md) | 2 | δ, ω | — | — | Classical model, $e'_q$ as input |
| [milano3ord](milano3ord.md) | 3 | δ, ω, e'q | — | scalar $K_{sat}$ | Flux-decay model |
| [milano4ord](milano4ord.md) | 4 | δ, ω, e'q, e'd | — | PSAT 2-point | Two-axis model; default |
| [milano6ord](milano6ord.md) | 6 | δ, ω, e'q, e''q, e'd, e''d | EMF ($e''_q$, $e''_d$) | PSAT 2-point | Full subtransient |
| [pai6ord](pai6ord.md) | 6 | δ, ω, e'q, ψ2d, e'd, ψ2q | Flux linkage ($\psi_{2d}$, $\psi_{2q}$) | — | PAI formulation |

All models share the same **algebraic stator equations** ($i_d$, $i_q$, $p_g$, $q_g$)
and accept $p_m$ and $v_f$ as inputs (supplied by governors and AVRs respectively).

```{toctree}
:maxdepth: 1

milano2ord
milano3ord
milano4ord
milano6ord
pai6ord
```
