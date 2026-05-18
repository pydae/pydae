# SOGI (phasor-frame frequency estimator)

*Second-Order Generalised Integrator with Frequency-Locked Loop —
`pydae.bps.miscellaneous.sogi`*

---

## Purpose

A drop-in alternative to [PLL](pll.md) that estimates the bus phase and
frequency **directly in the COI rotating-phasor frame**. The block
filters the SOGI's own in-frame phase error rather than the raw
alpha-axis bus voltage, so the equilibrium is well posed at zero slip.

---

## Mathematical model

The block carries its own rotating reference $\theta_{\mathrm{sogi}}$
driven by an FLL-adapted frequency $\omega_{\mathrm{sogi}}$. Define the
in-frame phase error:

$$v_{\beta,\mathrm{sogi}} =
    V_s \sin(\theta_{\mathrm{bus}} - \theta_{\mathrm{sogi}})$$

This signal feeds the SOGI band-pass:

$$\varepsilon = v_{\beta,\mathrm{sogi}} - x_v$$
$$\dot{x}_v   = \omega_n\,(k_{\mathrm{sogi}}\,\varepsilon - x_{qv})$$
$$\dot{x}_{qv}= \omega_n\, x_v - \alpha_{\mathrm{leak}}\, x_{qv}$$

with the SOGI centred at the *nominal* angular frequency
$\omega_n = 2\pi F_n$ (not at the estimated frequency), which keeps the
ini Jacobian well-conditioned even when the operating-point slip is
zero. The small leak $\alpha_{\mathrm{leak}}$ removes the marginal pole
of the quadrature integrator at DC.

The FLL drives $\omega_{\mathrm{sogi}}$ to null the filtered phase
error, and the phase integrator closes the loop in the COI frame:

$$\dot{\omega}_{\mathrm{sogi}} = -\gamma_{\mathrm{fll}}\, x_v$$
$$\dot{\theta}_{\mathrm{sogi}} =
    \Omega_b\,(\omega_{\mathrm{sogi}} - \omega_{\mathrm{coi}})$$

At lock $\theta_{\mathrm{sogi}}$ tracks $\theta_{\mathrm{bus}}$,
$x_v = x_{qv} = 0$, and $\omega_{\mathrm{sogi}} = \omega_{\mathrm{coi}}$.

---

## Variables

| Symbol | Name | Type | Units |
|---|---|---|---|
| $x_v$ | `x_v_sogi` | Dynamic state | pu |
| $x_{qv}$ | `x_qv_sogi` | Dynamic state | pu |
| $\omega_{\mathrm{sogi}}$ | `omega_sogi` | Dynamic state | pu |
| $\theta_{\mathrm{sogi}}$ | `theta_sogi` | Dynamic state | rad |
| $\omega_{\mathrm{sogi}}^{f}$ | `omega_sogi_f` | Dynamic state | pu |
| $\dot\omega_{\mathrm{sogi}}^{f}$ | `rocof_sogi_f` | Dynamic state | pu/s |
| $\dot\omega_{\mathrm{sogi}}$ | `rocof_sogi` | Algebraic | pu/s |

| Parameter | Key | Default | Description |
|---|---|---|---|
| $k_{\mathrm{sogi}}$ | `k_sogi` | $\sqrt{2}$ | SOGI damping |
| $\gamma_{\mathrm{fll}}$ | `gamma_fll` | 50.0 | FLL adaptation gain |
| $\alpha_{\mathrm{leak}}$ | `alpha_leak` | 0.5 | Quadrature leak |
| $T_f$ | `T_f` | 0.02 s | Output filter time constant |
| $F_n$ | `F_n` | 50 Hz | Nominal frequency |

Outputs: `omega_sogi_{name}`, `omega_sogi_f_{name}`,
`frequency_sogi_{name}` (Hz), `theta_sogi_{name}`, `rocof_sogi_{name}`,
`rocof_sogi_f_{name}`, plus the SOGI internal states `x_v_sogi_{name}`
and `x_qv_sogi_{name}`.

---

## Configuration

```hjson
plls: [{
    bus:        "1",
    type:       "sogi",
    k_sogi:     1.414,
    gamma_fll:  50.0,
    alpha_leak: 0.5,
    T_f:        0.02,
    F_n:        50.0
}]
```

Keys after `bus` and `type` are optional.

---

## In-module test

`sogi.py` ships with `sogi.hjson` and a `test()` function that builds
the block via **CasadiBuilder** and asserts that the SOGI is locked to
the COI in the rest equilibrium (`omega_sogi_f = 1.0`,
`x_v = x_{qv} = 0`). Run with::

    uv run python packages/pydae-bps/src/pydae/bps/miscellaneous/sogi.py
