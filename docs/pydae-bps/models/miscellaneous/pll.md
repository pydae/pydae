# PLL (PI bus frequency estimator)

*Synchronous-reference-frame phase-locked loop —
`pydae.bps.miscellaneous.pll`*

---

## Purpose

Estimates the phase and angular frequency of a balanced bus voltage
phasor using a classical PI-based phase-locked loop. Add the block via
the top-level `plls` key in the network hjson; one block per bus where
you need an independent frequency estimate. Other devices (PSS, droop
controllers) can read the resulting `omega_pll_*`, `frequency_pll_*` and
`rocof_pll_*` outputs.

---

## Mathematical model

The bus voltage phasor $V \angle \theta_s$ is projected onto the PLL's
rotating reference $\theta_{\mathrm{pll}}$:

$$v_{sD} = V \sin \theta_s, \quad v_{sQ} = V \cos \theta_s$$
$$v_{sd}^{\mathrm{pll}} =
    v_{sD}\,\cos \theta_{\mathrm{pll}} -
    v_{sQ}\,\sin \theta_{\mathrm{pll}}$$

A PI regulator on $v_{sd}^{\mathrm{pll}}$ produces the angular-frequency
correction:

$$\dot{\xi}_{\mathrm{pll}} = v_{sd}^{\mathrm{pll}}$$
$$\omega_{\mathrm{pll}} = 1 + K_p\, v_{sd}^{\mathrm{pll}}
                            + K_i\, \xi_{\mathrm{pll}}$$

The PLL phase is integrated in the COI-rotating frame so the steady
state is stationary:

$$\dot{\theta}_{\mathrm{pll}} = 2\pi f_n\,
    K_\theta\,(\omega_{\mathrm{pll}} - \omega_{\mathrm{coi}})$$

The output is filtered with a first-order lag of time constant
$T_{\mathrm{pll}}$ and the rate-of-change of frequency exposed as an
algebraic variable plus a smoothed state.

At lock $\omega_{\mathrm{pll}} = \omega_{\mathrm{coi}}$ and
$\theta_{\mathrm{pll}}$ is constant.

---

## Variables

| Symbol | Name | Type | Units |
|---|---|---|---|
| $\theta_{\mathrm{pll}}$ | `theta_pll` | Dynamic state | rad |
| $\xi_{\mathrm{pll}}$ | `xi_pll` | Dynamic state | pu |
| $\omega_{\mathrm{pll}}^{f}$ | `omega_pll_f` | Dynamic state | pu |
| $\dot\omega_{\mathrm{pll}}^{f}$ | `rocof_pll_f` | Dynamic state | pu/s |
| $\dot\omega_{\mathrm{pll}}$ | `rocof_pll` | Algebraic | pu/s |

| Parameter | Key | Default | Description |
|---|---|---|---|
| $K_{p,\mathrm{pll}}$ | `K_p_pll` | 180.0 | PLL proportional gain |
| $K_{i,\mathrm{pll}}$ | `K_i_pll` | 3200.0 | PLL integral gain |
| $T_{\mathrm{pll}}$ | `T_pll` | 0.02 s | Output filter time constant |
| $K_{\theta,\mathrm{pll}}$ | `K_theta_pll` | 1.0 | Phase-integrator scaling |

Outputs available in `model.get_value(...)`:
`omega_pll_{name}`, `omega_pll_f_{name}`, `frequency_pll_{name}` (in Hz),
`theta_pll_{name}`, `rocof_pll_{name}`.

---

## Configuration

```hjson
plls: [{
    bus:        "1",
    K_p_pll:    180.0,
    K_i_pll:    3200.0,
    T_pll:      0.02,
    K_theta_pll: 1.0
}]
```

`bus` is required; `name` defaults to the bus name if not given.

---

## In-module test

`pll.py` ships with `pll.hjson` (a minimal two-bus, two-vsource grid)
and a `test()` function that builds the block with **CasadiBuilder** and
verifies that the PLL is locked to the COI at the rest equilibrium
(`omega_pll_f = 1.0`, `rocof = 0`). Run with::

    uv run python packages/pydae-bps/src/pydae/bps/miscellaneous/pll.py
