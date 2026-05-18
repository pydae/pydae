# Miscellaneous components

This section groups stand-alone helper devices: bus-frequency
estimators (PLL and SOGI variants) and the secondary/tertiary
active-power controllers.

## Bus frequency estimators

Each estimator attaches to a bus via an entry under the top-level
`plls` key in the network hjson. Choose the variant that matches your
problem:

* [PLL](pll.md) — classical PI synchronous-reference-frame PLL. Locks
  to the bus phasor directly in the COI rotating-phasor frame; no AC
  carrier required.
* [SOGI](sogi.md) — Second-Order Generalised Integrator with FLL,
  formulated against the in-frame phase error. Drop-in alternative to
  the PI PLL that operates entirely in the phasor frame.
* [SOGI-FLL](sogi_fll.md) — textbook EMT formulation. Takes an
  instantaneous AC voltage sample (`v_grid`) as a free input; useful
  for EMT-RMS co-simulation.
* [SOGI-PLL](sogi_pll.md) — textbook EMT formulation: SOGI band-pass
  followed by a synchronous-reference-frame PI PLL. Same `v_grid`
  free-input convention as SOGI-FLL.

```{toctree}
:maxdepth: 1

pll
sogi
sogi_fll
sogi_pll
```

## Frequency and active-power controllers

These components implement the secondary and tertiary levels of the
hierarchical active-power / frequency control architecture used in
synchronous-machine networks.

```{toctree}
:maxdepth: 1

load_controller
agc
```

## Control hierarchy

Power-system frequency control is organised in three nested timescales:

```{mermaid}
flowchart TB
    LC["**Load Controller — LC** (tertiary)\np_c_lc → x_lc\nτ ≈ 100 s\nEnsures p_g = p_c_lc at rest"]
    AGC["**AGC** (secondary)\ndp_lc = K_p·ε + K_i·ξ\nτ ≈ 1–30 s\nEliminates Δω"]
    GOV["**Governor** (primary)\ndroop, valve dynamics\nτ ≈ 0.5–10 s\nArrests Δω"]
    MACH["Synchronous machine\np_g into grid"]

    LC -->|"x_lc (base)"| SUM("+")
    AGC -->|"dp_lc (fast)"| SUM
    SUM -->|"ctrl_sym = p_c or p_m"| GOV
    GOV -->|"p_m"| MACH
    MACH -->|"p_g (feedback)"| LC
    MACH -->|"ω (feedback)"| AGC
    MACH -->|"ω (feedback)"| GOV
```

| Level | Component | Timescale | Removes |
|-------|-----------|-----------|---------|
| Primary | Governor (`tgov1`, `hygov`, …) | ms – 10 s | Frequency *rate* of change |
| Secondary | [AGC](agc.md) | 1 – 120 s | Steady-state frequency *error* |
| Tertiary | [Load Controller](load_controller.md) | 1 – 10 min | `p_g ≠ p_c` due to armature losses |
```
