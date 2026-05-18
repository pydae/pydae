# SOGI-PLL (textbook EMT formulation)

*Canonical SOGI band-pass with synchronous-reference-frame PI PLL —
`pydae.bps.miscellaneous.sogi_pll`*

---

## Purpose

Implements the textbook SOGI-PLL: a SOGI band-pass produces in-phase
and quadrature voltage estimates which are then Park-transformed into a
PI-locked synchronous reference frame. As with
[SOGI-FLL](sogi_fll.md), the input `v_grid` is an instantaneous AC
sample (EMT-style) registered as a free input.

For a phasor-frame estimator that locks to the bus phasor directly, see
[SOGI](sogi.md).

---

## Mathematical model

$$0 = v_{\mathrm{grid}} - v_p - \varepsilon_v$$
$$0 = (v_p\cos\hat\theta + q v_p\sin\hat\theta) - v_d$$
$$0 = (-v_p\sin\hat\theta + q v_p\cos\hat\theta) - v_q$$
$$0 = (\omega_0 + k_p\,v_q + x_{\mathrm{pi}}) - \hat\omega$$
$$\dot{v}_p       = \hat\omega\,(k_{\mathrm{sogi}}\,\varepsilon_v - q v_p)$$
$$\dot{q v_p}     = \hat\omega\, v_p$$
$$\dot{x}_{\mathrm{pi}} = k_i\, v_q - \epsilon_{\mathrm{reg}}\, x_{\mathrm{pi}}$$
$$\dot{\hat\theta} = \hat\omega - \omega_{\mathrm{ref}}
                    - \epsilon_{\mathrm{reg}}\,\hat\theta$$

**Notes**:

* The textbook phase integrator $\dot{\hat\theta} = \hat\omega$ has no
  equilibrium (phase grows linearly at the grid rate). We work in a
  frame rotating at $\omega_{\mathrm{ref}}$ so a steady state exists;
  by default $\omega_{\mathrm{ref}} = \omega_0$ and the two forms
  coincide once the loop is locked at nominal.
* The two trailing $\epsilon_{\mathrm{reg}}$ terms are leaky-integrator
  regularisations. At the trivial equilibrium $v_p = q v_p = 0$ the
  algebraic Park-transform equations have a zero column for
  $\hat\theta$ (its partial derivatives all multiply $v_p$ or
  $q v_p$), leaving $\hat\theta$ unobservable to the Newton
  initialiser. The leaks resolve this rank deficiency and have
  negligible effect on transient dynamics at the default
  $\epsilon_{\mathrm{reg}} = 10^{-6}$.

At lock: $v_p = v_{\mathrm{grid}}$,
$v_d = \|v_{\mathrm{grid}}\|$, $v_q = 0$,
$\hat\omega = \omega_{\mathrm{grid}}$.

---

## Variables

| Symbol | Name | Type | Units |
|---|---|---|---|
| $v_p$ | `v_p_sogi_pll` | Dynamic state | pu |
| $q v_p$ | `qv_p_sogi_pll` | Dynamic state | pu |
| $x_{\mathrm{pi}}$ | `x_pi_sogi_pll` | Dynamic state | pu |
| $\hat\theta$ | `theta_est_sogi_pll` | Dynamic state | rad |
| $\varepsilon_v$ | `err_v_sogi_pll` | Algebraic | pu |
| $v_d$ | `v_d_sogi_pll` | Algebraic | pu |
| $v_q$ | `v_q_sogi_pll` | Algebraic | pu |
| $\hat\omega$ | `omega_est_sogi_pll` | Algebraic | rad/s |
| $v_{\mathrm{grid}}$ | `v_grid_sogi_pll` | Input | pu |

| Parameter | Key | Default | Description |
|---|---|---|---|
| $k_{\mathrm{sogi}}$ | `k_sogi` | $\sqrt{2}$ | SOGI filter gain |
| $k_p$ | `k_p` | 100.0 | PI proportional gain |
| $k_i$ | `k_i` | 2000.0 | PI integral gain |
| $\omega_0$ | `omega_0` | $2\pi\cdot 50$ rad/s | Nominal angular frequency |
| $\omega_{\mathrm{ref}}$ | `omega_ref` | $\omega_0$ | Reference-frame angular velocity |
| $\epsilon_{\mathrm{reg}}$ | `epsilon_reg` | $10^{-6}$ | Leaky-integrator regularisation |

Outputs: `v_p_sogi_pll_{name}`, `qv_p_sogi_pll_{name}`,
`v_d_sogi_pll_{name}`, `v_q_sogi_pll_{name}`,
`theta_est_sogi_pll_{name}`, `omega_est_sogi_pll_{name}`,
`f_est_sogi_pll_{name}` (in Hz), `x_pi_sogi_pll_{name}`,
`err_v_sogi_pll_{name}`.

---

## Configuration

```hjson
plls: [{
    name:        "pll1",
    type:        "sogi_pll",
    k_sogi:      1.414,
    k_p:         100.0,
    k_i:         2000.0,
    omega_0:     314.159265358979,
    omega_ref:   314.159265358979,
    epsilon_reg: 1.0e-6,
    v_grid:      0.0
}]
```

If a `bus` key is given instead of `name`, the bus name is used as the
namespace identifier; `v_grid` remains a free input.

---

## In-module test

`sogi_pll.py` ships with `sogi_pll.hjson` and a `test()` function that
builds via **CasadiBuilder** and asserts the rest equilibrium
($v_p = q v_p = x_{\mathrm{pi}} = \hat\theta = 0$,
$\hat\omega = \omega_0$, $\hat f = 50$ Hz). Run with::

    uv run python packages/pydae-bps/src/pydae/bps/miscellaneous/sogi_pll.py
