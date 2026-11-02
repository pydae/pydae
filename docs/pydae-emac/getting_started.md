# Getting started

## A linear machine under FOC

```python
from pydae.emac import EmacBuilder

data = {
    "machines": [{"type": "pmsm_dq", "name": "1", "R_s": 0.12, "L_d": 2e-3, "L_q": 3.5e-3,
                  "Psi_pm": 0.12, "p": 4, "mech": "imposed", "omega_m": 150.0}],
    "ctrls": [{"type": "foc_pi", "name": "1", "machine": "1", "bandwidth": 500.0,
               "V_dc": 540.0, "i_d_ref": -5.0, "i_q_ref": 20.0}],
}
drive = EmacBuilder(data)
model = drive.build("drive")                    # CasadiModel
model.ini({}, drive.sys_dict["xy_0_dict"])      # steady state at the references
print(model.get_value("T_e_1"))                 # 15.3 N.m
print(model.eval_eigenvalues())                 # -2 pi 500 (x2), -R/L_d, -R/L_q
```

With $K_p = 2\pi f_{bw} L$ and $K_i = K_p R_s / L$ the PI cancels the stator
pole, so the loop poles are exactly $-2\pi f_{bw}$ on a linear machine. A step
in a reference, through pydae's fixed-step solver:

```python
model.Dt = 1e-5
model.set_value("i_q_ref_1", 30.0)
model.recalculate_algebraics()                  # the PI's proportional path jumps
model.run(0.005, method="trapezoidal")
model.post()
```

## A machine from FEA tables

The FEA table is a `FluxMap_dqt.npz` (see `pydae.emac.luts.flux_tables`):

```python
from pydae.emac.luts import load_fea_lut, build_flux_tables
from pydae.emac.utils.analysis import incremental_inductances

lut = load_fea_lut("FluxMap_dqt.npz")
tables = build_flux_tables(lut)
L_d, L_q = incremental_inductances(lut, i_ref)      # i_ref = (i_d, i_q), A peak

data = {
    "machines": [{"type": "pmsm_lut", "name": "m", "tables": tables, "R_s": R,
                  "mech": "imposed", "omega_m": omega_m}],
    "ctrls": [{"type": "foc_pi", "name": "m", "machine": "m", "bandwidth": 400.0,
               "L_d": L_d, "L_q": L_q, "V_dc": V_dc,
               "i_d_ref": i_ref[0], "i_q_ref": i_ref[1]}],
}
```

On a saturated machine the incremental inductance matrix has cross terms, so
tuning on its diagonal places the loop poles off $-2\pi f_{bw}$; the
eigenvalues show by how much.

## The switched drive

```python
import numpy as np
from pydae.emac.drive import build_machine, build_window, build_control_period, simulate, PIController

ode, currents = build_machine(tables, R, J)
Kp = 2 * np.pi * 400.0 * np.array([L_d, L_q])
period = build_control_period(build_window(ode), currents, lut["p"], V_dc, T_s,
                              PIController(Kp, Kp * R / np.array([L_d, L_q])),
                              mode="switched", t_dead=2e-6)
r = simulate(period, currents, x0, T_s, n_periods, i_ref)
```

A controller of your own is any class with `n_state`, `n_ref` and
`step(meas, ref, xc, T_s)` written with CasADi operations;
`pydae.emac.drive.PredictiveController` is a worked example.

Two things to know when measuring: with pole-zero-cancelling gains the loop
rejects its decoupling error only at the machine's own $L/R$, so let a run
settle for about $5L/R$; and the spatial ripple biases any mean not taken over
whole electrical periods.
