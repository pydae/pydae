# SOGI-FLL (textbook EMT formulation)

*Canonical Second-Order Generalised Integrator with Frequency-Locked
Loop — `pydae.bps.miscellaneous.sogi_fll`*

---

## Purpose

Implements the textbook SOGI-FLL exactly as in Rodriguez et al. (2006 /
2011). The input `v_grid` is an *instantaneous* AC voltage sample
(EMT-style), registered as a free `u_run_dict` input rather than bound
to a bus phasor. This makes the block useful when paired with an
external real-time signal source (e.g. EMT-RMS co-simulation, or a
post-processing loop that re-samples a stored waveform).

For a phasor-frame estimator that locks to the bus phasor directly, see
[SOGI](sogi.md).

---

## Mathematical model

$$0 = v_{\mathrm{grid}} - v_p - \varepsilon_v$$
$$\dot{v}_p   = \hat\omega\,(k_{\mathrm{sogi}}\,\varepsilon_v - q v_p)$$
$$\dot{q v_p} = \hat\omega\, v_p$$
$$\dot{\hat\omega} = -\Gamma\, \varepsilon_v\, q v_p
                    - \epsilon_{\mathrm{reg}}\,(\hat\omega - \omega_0)$$

The trailing $\epsilon_{\mathrm{reg}}$ term is a leaky-integrator
regularisation that gives `omega_est` a non-zero diagonal in the ini
Jacobian. At the trivial equilibrium `err_v = qv_p = 0` the textbook
FLL law has identically zero value and Jacobian, leaving `omega_est`
unobservable to the Newton initialiser. With $\epsilon_{\mathrm{reg}} >
0$ the equilibrium uniquely pins $\hat\omega = \omega_0$. The default
$\epsilon_{\mathrm{reg}} = 10^{-6}$ has negligible effect on the
transient FLL dynamics.

At lock $v_p = v_{\mathrm{grid}}$, $q v_p$ is the quadrature
($90^\circ$-lagged) version of $v_p$, and
$\hat\omega = \omega_{\mathrm{grid}}$.

---

## Variables

| Symbol | Name | Type | Units |
|---|---|---|---|
| $v_p$ | `v_p_sogi_fll` | Dynamic state | pu |
| $q v_p$ | `qv_p_sogi_fll` | Dynamic state | pu |
| $\hat\omega$ | `omega_est_sogi_fll` | Dynamic state | rad/s |
| $\varepsilon_v$ | `err_v_sogi_fll` | Algebraic | pu |
| $v_{\mathrm{grid}}$ | `v_grid_sogi_fll` | Input | pu |

| Parameter | Key | Default | Description |
|---|---|---|---|
| $k_{\mathrm{sogi}}$ | `k_sogi` | $\sqrt{2}$ | SOGI filter gain |
| $\Gamma$ | `Gamma` | 50.0 | FLL adaptation gain |
| $\omega_0$ | `omega_0` | $2\pi\cdot 50$ rad/s | Nominal angular frequency |
| $\epsilon_{\mathrm{reg}}$ | `epsilon_reg` | $10^{-6}$ | Leaky-integrator regularisation |

Outputs: `v_p_sogi_fll_{name}`, `qv_p_sogi_fll_{name}`,
`omega_est_sogi_fll_{name}`, `f_est_sogi_fll_{name}` (in Hz),
`err_v_sogi_fll_{name}`.

---

## Configuration

```hjson
plls: [{
    name:        "fll1",
    type:        "sogi_fll",
    k_sogi:      1.414,
    Gamma:       50.0,
    omega_0:     314.159265358979,
    epsilon_reg: 1.0e-6,
    v_grid:      0.0
}]
```

If a `bus` key is given instead of `name`, the bus name is used as the
namespace identifier; `v_grid` remains a free input — it is **not**
bound to the bus voltage.

---

## Driving `v_grid` for transient tests

To exercise the textbook dynamics, drive `v_grid` from the outer loop
with a small step `Dt` and per-call updates::

    omega_grid = 2 * np.pi * 50.0
    Dt = 1e-4
    for k in range(N):
        t = (k + 1) * Dt
        v = V_amp * np.sin(omega_grid * t)
        model.run(t, {f"v_grid_sogi_fll_{name}": v})

---

## In-module test

`sogi_fll.py` ships with `sogi_fll.hjson` and a `test()` function that
builds via **CasadiBuilder** and asserts the rest equilibrium
($v_p = q v_p = \varepsilon_v = 0$, $\hat\omega = \omega_0$,
$\hat f = 50$ Hz). Run with::

    uv run python packages/pydae-bps/src/pydae/bps/miscellaneous/sogi_fll.py
