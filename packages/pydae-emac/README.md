# pydae-emac

Electric machines and their controllers for the [pydae](https://github.com/pydae/pydae) DAE solver: simulate, control and develop machines and drives.

```bash
pip install pydae-emac   # brings pydae as a dependency
```

## Two solvers, one set of tables

| | mean-value DAE (`EmacBuilder`) | switched drive (`pydae.emac.drive`) |
|---|---|---|
| solved by | pydae's own: `ini()`, `run()` (IDAS or fixed-step trapezoidal), `A_eval()` / eigenvalues | a multirate CasADi solver: one compiled function per control period, chained by `mapaccum` |
| machine | dq, linear (`pmsm_dq`) or from FEA tables with saturation and cross-saturation (`pmsm_lut`), angle-averaged | dq, from the same FEA tables, **with** the spatial ripple (Fourier in the rotor angle) |
| inverter | the voltage asked for, optionally behind a first-order delay | DSP sampling every T_s, computation delay, triangular carrier, each leg's switching edge, dead time |
| controller | DAE blocks (`foc_pi`, built from `pydae.control` `pi` / `lpf1`) | any object with `step(meas, ref, xc, T_s)` written in CasADi |
| for | steady states, transients, eigenvalues, loop design, coupling to a pydae network | ripple, THD, switching, dead time, sampling effects |

Both read the FEA look-up tables through `pydae.emac.luts`.

## Mean-value DAE

```python
from pydae.emac import EmacBuilder

data = {
    "machines": [{"type": "pmsm_dq", "name": "1", "R_s": 0.12, "L_d": 2e-3, "L_q": 3.5e-3,
                  "Psi_pm": 0.12, "p": 4, "mech": "imposed", "omega_m": 150.0}],
    "ctrls": [{"type": "foc_pi", "name": "1", "machine": "1", "bandwidth": 500.0,
               "V_dc": 540.0, "i_d_ref": -5.0, "i_q_ref": 20.0}],
}
drive = EmacBuilder(data)
model = drive.build("drive")                     # a CasadiModel; nothing compiled
model.ini({}, drive.sys_dict["xy_0_dict"])       # steady state at the references
model.Dt = 1e-5
model.run(0.01, {"i_q_ref_1": 30.0}, method="trapezoidal")
model.post()
t, T_e = model.Time, model.get_values("T_e_1")
```

For long transients, `method="rk4"` (pydae-core) reduces machine + controller to a plain ODE (every algebraic variable here is explicit) and runs the whole span in one compiled call: about 3 µs/step against 15–18 for `idas_grid` and 53 for `trapezoidal` on an FEA-table drive.

An FEA machine is the same with `{"type": "pmsm_lut", "lut_file": "FluxMap_dqt.npz", "R_s": ...}`. `mech: "free"` frees the speed: `ini()` then solves the load torque that holds the given speed, and `run()` keeps it (pydae's ini/run swap).

Models carry the usual pydae structure: one module per component with its equations in the docstring, a `descriptions()` list (parameters, inputs, states, outputs) and a builder function; families dispatched by `type`.

| family | models |
|---|---|
| `machines/` | `pmsm_dq` (linear PMSM / IPM / SynRM), `pmsm_lut` (FEA tables); rotor in `mechanics.py` |
| `ctrls/` | `foc_pi` (dq current PI + decoupling, optional inverter delay) |

## Switched drive

```python
from pydae.emac.luts import load_fea_lut, build_flux_tables
from pydae.emac.drive import build_machine, build_window, build_control_period, simulate, PIController

lut = load_fea_lut("FluxMap_dqt.npz")
tables = build_flux_tables(lut, theta_offset_e=0.0)   # pi/2 for a reluctance-frame map
ode, currents = build_machine(tables, R, J)
period = build_control_period(build_window(ode), currents, lut["p"], V_dc, T_s, PIController(Kp, Ki))
r = simulate(period, currents, x0, T_s, n_periods, i_ref)   # one CasADi call
```

**From a Python loop** — `PwmPlant` is the machine with its PWM modulator and inverter, advanced one control period per call. Your loop is the processor: every T_s it reads the sensors with `plant.measurements()` — the phase currents `i_abc`, the encoder angle `theta_m` and the speed, no dq — makes dq itself (`abc_to_dq`), computes the three modulation values (duties; `svpwm_duties` turns a voltage into them) and calls `plant.step(duties)`. The plant compares them with the triangular carrier, switches the legs where they cross and integrates the machine through those instants. A Python loop that does what `PIController` does reproduces `simulate()` (same switching instants, states to 1e-13) at ~27 µs per 50 µs period.

```python
plant = PwmPlant(tables, R, J, V_dc, T_s, x0=x0)       # mode="averaged", t_dead= also available
for k in range(n):
    m = plant.measurements()                            # i_abc, theta_m, omega_m
    i_dq = abc_to_dq(m["i_abc"], p * m["theta_m"])      # Clarke + Park on the processor
    d = my_controller(i_dq, ...)                        # -> three duties in [0, 1]
    plant.step(d)                                       # step(d, T_load=...) frees the speed
r = plant.results()                                     # every switching instant
```

`mode="averaged"` holds the period-mean voltage instead of switching; `t_dead=`, `compensate=` add the inverter's dead time and its compensation. `pydae.emac.drive.PredictiveController` is a worked example of a user controller.

## Layout

```
src/pydae/emac/
  emac_builder.py     EmacBuilder: data -> sys_dict -> CasadiModel
  machines/           DAE machine models
  ctrls/              DAE controllers
  luts/               FEA tables: load, invert, Fourier, pack; synthetic tables with known answers
  drive/              the switched multirate solver and its controllers
  utils/              analysis (harmonics, ripple, inductances), docs helpers
```

## Tests

```bash
uv run pytest -m emac
```
