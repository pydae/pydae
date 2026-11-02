# Overview

## Two solvers, one set of tables

A machine and its drive are studied at two levels of detail, and `pydae-emac`
gives each its proper solver rather than forcing one to do both jobs.

**Mean-value DAE** (`EmacBuilder`). The machine in the rotor's dq frame, its
controller, and the inverter as the voltage it is asked for (optionally behind
a first-order delay). Without the rotor angle the model has a true equilibrium,
so everything pydae-core offers applies: `ini()` to the steady state at the
references, `run()` with IDAS or the fixed-step trapezoid, `A_eval()` and the
eigenvalues for loop design. It can also be coupled to a pydae power-system
model, since it is an ordinary `sys_dict`.

**Switched drive** (`pydae.emac.drive`). The DSP samples the currents every
$T_s$ and writes duties that the PWM hardware compares with a triangular
carrier, so each leg switches exactly where duty and carrier cross. One
control period is a single CasADi function (four constant-voltage pieces, one
RK4 step each), and `mapaccum` chains thousands of them in one call. The FEA's
spatial ripple is kept as a Fourier series in the rotor angle. Dead time and
its compensation are options.

Both read the FEA look-up tables through `pydae.emac.luts`: the
currents-to-fluxes maps the FEA produces, inverted to fluxes-to-currents (the
models' states are the fluxes, so a voltage step never jumps a state), with
the angle dependence as a Fourier series and everything packed for one CasADi
B-spline. The DAE machine uses the series' mean term, the switched drive every
term.

## Models

| family | model | what |
|---|---|---|
| `machines` | `pmsm_dq` | linear PMSM / IPM / SynRM: constant $L_d$, $L_q$, $\Psi_{pm}$ |
| `machines` | `pmsm_lut` | any synchronous machine from FEA tables, with saturation and cross-saturation |
| `machines` | `mechanics` | the rotor: speed imposed, or free with the ini/run swap |
| `ctrls` | `foc_pi` | dq current PI + decoupling, from `pydae.control` `pi` and `lpf1` blocks |

Each model's module docstring holds its equations and its `descriptions()`
list holds its parameters, inputs, states and outputs, as in the sibling
packages.

## Conventions

- **Amplitude-invariant dq, peak values**, the same as the FEA tables.
- **Fluxes are the states** in every machine model.
- **Frame.** The models work in whatever dq frame the table is in. The linear
  model puts the magnet on d. For the switched drive, `theta_offset_e` says
  where the table's d axis sits against phase a at FEM angle 0 (0 for a
  pyleecan PM map, $\pi/2$ for a reluctance-frame map); the angle-averaged DAE
  model does not depend on it.
- **Mechanics `"free"`.** `ini()` is given the speed and solves the load
  torque; `run()` holds that torque and frees the speed. This is pydae's ini/run
  variable swap (as `V ↔ v_ref` in pydae-bps), so a machine starts in
  mechanical equilibrium.
