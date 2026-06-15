# Continuous control blocks

`pydae.control.continuos` is a library of reusable continuous control
blocks that can be dropped into any pydae model. Each block is a
small DAE fragment (a handful of states, an algebraic output, and
some parameters) that follows a uniform builder signature, so they
compose naturally with the power-system component builders in
`pydae-bps` / `pydae-uds`.

## Uniform builder signature

Every block in `pydae.control.continuos` exposes the same public
function:

```python
block(dae, name, backend, data_dict=None, input=None, output=None)
```

| Argument     | Type                          | Purpose                                                                       |
| ------------ | ----------------------------- | ----------------------------------------------------------------------------- |
| `dae`        | `dict`                        | A `sys_dict`-shaped target; the block appends into its `*_list` / `*_dict` keys. |
| `name`       | `str`                         | Instance name; appears as a suffix on every symbol the block creates (e.g. `xi_pi_{name}`). |
| `backend`    | `MathBackend`                 | Symbol factory (SymPy or CasADi). Use the same `grid.backend` instance everywhere so symbol caches stay coherent. |
| `data_dict`  | `dict` or `None`              | Block-specific parameters (gains, time constants, matrices, ...). Missing keys fall back to the `descriptions()` defaults. |
| `input`      | `None` / `str` / `list[str]`  | How the block reads its input(s). See the *Input and output aliasing* section below. |
| `output`     | `None` / `str` / `list[str]`  | How the block exposes its output(s). Same conventions as `input`.            |

The function does not return a value; it mutates `dae` in place. The
shared `pydae.control.control_builder.ControlBuilder` accepts any
block in the library through the same dispatch path -- a HJSON
description of the form

```text
{ control: [
    { type: "pi",  name: "ctrl_1", K_p: 2.0, K_i: 0.5 },
    { type: "lti", name: "plant",  A: [[-1, 1], [0, -2]], B: [[0],[1]],
                                   C: [[1, 0]],          D: [[0]] },
]}
```

is enough to assemble a full DAE.

## Input and output aliasing

`input=` and `output=` make it trivial to wire a block onto signals
that already exist in the surrounding `dae`, without manually
declaring connecting algebraic equations. Resolution rules
(centralised in `pydae.control.continuos._common.resolve_input`):

1. If the name is a key of `dae['h_dict']`, the corresponding
   expression is substituted in directly -- no new free symbol is
   introduced, the constraints that produced the h_dict entry stay
   in force.
2. Otherwise, if the name matches an existing state, algebraic
   variable, input, or parameter, the cached backend symbol is
   aliased.
3. Otherwise the name is created as a new global input, added to
   both `u_ini_dict` and `u_run_dict`.

For `output=`, an aliased name that was *previously a global input*
(e.g. `Qext_5` declared by a REEC_B block) is automatically promoted
to an algebraic variable driven by the block's saturated output --
the same lifecycle used by `stab_1lpf_2wo_3ll` for PSS-like wiring.

**List semantics.** The way a `list[str]` is combined into the
block's input expression depends on the block:

| Block            | `input=list[str]` semantics                                                |
| ---------------- | -------------------------------------------------------------------------- |
| `pi`, `pid`, `lpf1`, `lpf2`, `leadlag`, `washout`, `tf`, `notch`, `saturation`, `deadband`, `rate_limiter`, `integrator`, `stab_1lpf_2wo_3ll` | `u = input[0] - sum(input[1:])` (classic error: measurement minus references). |
| `summation`, `product` | Must be a 2-list `[u_1, u_2]`; bound to the two input channels.       |
| `pi_feedforward` | Must be a 2-list `[e, r]`; first binds to the error `e`, second to the feedforward reference `r` (the two are used in different terms of the PI law, not subtracted). |
| `matrix_gain`    | Length must equal `m` (cols of `K`); each name bound to one column.       |
| `lti`            | Length must equal `m`; each name bound to one column of `B`.              |
| `luenberger`, `kalman` | Length must equal `m + p`; the first `m` bind to the control inputs and the next `p` bind to the measurements. |
| `park`, `park_inv` | Must be a 4-list (`[u_a, u_b, u_c, theta]` for `park`; `[u_d, u_q, u_0, theta]` for `park_inv`); each name binds to one channel. |
| `pll`            | Must be a 3-list `[v_a, v_b, v_c]` of the three-phase voltage signals.     |

## Block catalogue

### Single-input filters

| Module       | Transfer function / description                                          |
| ------------ | ------------------------------------------------------------------------ |
| `lpf1`       | First-order low-pass `1 / (T*s + 1)`.                                    |
| `lpf2`       | Second-order low-pass `omega_n^2 / (s^2 + 2 zeta omega_n s + omega_n^2)`. |
| `leadlag`    | First-order lead-lag `(T_num*s + 1) / (T_den*s + 1)`.                    |
| `washout`    | Washout (high-pass) `T_wo*s / (T_wo*s + 1)`.                             |
| `notch`      | Bandstop `(s^2 + omega_n^2) / (s^2 + (omega_n/Q)*s + omega_n^2)`. Parametrised by physical centre frequency `omega_n` and quality factor `Q`. Useful for SSR damping, torsional-mode rejection, harmonic filtering. |

Each block stores one (or two) dynamic states, exposes its output
through `h_dict`, and follows the same `input=` / `output=` aliasing
conventions.

### Controllers

| Module             | Description                                                                                                  |
| ------------------ | ------------------------------------------------------------------------------------------------------------ |
| `pi`               | PI controller with output saturation and back-calculation anti-windup.                                       |
| `pid`              | PID with filtered derivative (`K_d * s / (1 + s T_f)`), saturation, anti-windup.                             |
| `pi_feedforward`   | PI with an additional reference-feedforward term `K_ff * r`. Two distinct input slots (error `e`, reference `r`); for the same `K_p` / `K_i`, a non-zero `K_ff` produces a larger immediate jump on a step in `r` than the plain PI. |
| `integrator`       | Single-state integrator with output saturation and anti-windup. The I-only piece of `pi` -- use for accumulators / totalizers (energy, charge, integrated frequency error). |

### Signal conditioning and nonlinear

| Module           | Description                                                                                                  |
| ---------------- | ------------------------------------------------------------------------------------------------------------ |
| `saturation`     | Algebraic clamp `z = clip(u, z_min, z_max)`. Declarative replacement for inline `hard_limits`.              |
| `deadband`       | Algebraic deadband, zero inside `[db_low, db_high]` and slope 1 outside (no jump at the edges). Asymmetric bands supported; matches the REEC_B / REPC_A `dbd1` / `dbd2` convention. |
| `rate_limiter`   | First-order tracker with rate clipping `dz/dt = clip((u - z)/T, dz_min, dz_max)`. Behaves like `lpf1` for small input changes and ramps at the limit for large steps; models actuator slew, REGC_A `rrpwr`, REEC_B `dPmax` / `dPmin`. |
| `product`        | Algebraic two-input multiplier `z = K * u_1 * u_2`. Nonlinear-coupling building block: instantaneous power `V * I`, droop products, modulation. |

### Algebraic / multivariable

| Module        | Description                                                                                                  |
| ------------- | ------------------------------------------------------------------------------------------------------------ |
| `summation`   | Two-input weighted sum `z = K_1*u_1 + K_2*u_2`. Algebraic, no state.                                          |
| `matrix_gain` | Generic static map `z = K * u`, with `K` a `p x m` matrix. The MIMO generalisation of `summation`.          |
| `tf`          | Generic transfer function `H(s) = NUM(s) / DEN(s)`, MATLAB convention (descending powers of `s`). Realised in controllable canonical form. |
| `lti`         | Generic state-space block `dx/dt = A x + B u`, `z = C x + D u`. Dimensions derived from matrix shapes; SISO to arbitrary MIMO with the same API. |

### Estimators

| Module        | Description                                                                                                  |
| ------------- | ------------------------------------------------------------------------------------------------------------ |
| `luenberger`  | State observer for an LTI plant, gain matrix `L` chosen by pole placement. Two input streams (`m` control + `p` measurement) concatenated in `input=`. Exposes state estimates as outputs and `y_hat`, `y_tilde` (innovation) signals in `h_dict`. |
| `kalman`      | Steady-state continuous Kalman filter. Structurally identical to `luenberger`; the gain is computed at build time by solving the continuous algebraic Riccati equation from process- and measurement-noise covariances `Q`, `R`. The chosen gain entries are exposed in `h_dict` as `K_ij_kalman_{name}` so the Riccati solution is readable from the built model. |

### Three-phase transforms and grid synchronisation

| Module      | Description                                                                                                  |
| ----------- | ------------------------------------------------------------------------------------------------------------ |
| `park`      | Amplitude-invariant Park transform `(u_a, u_b, u_c, theta) -> (z_d, z_q, z_0)` (`2/3` factor, matches the WECC renewable-converter convention). Purely algebraic. |
| `park_inv`  | Inverse Park `(u_d, u_q, u_0, theta) -> (z_a, z_b, z_c)`. Pair with `park` for round-trip identity. |
| `pll`       | Synchronous-Reference-Frame Phase-Locked Loop. Three-phase voltage in, estimated angle / frequency out. Uses the deviation-form angle dynamics `dθ̂/dt = ω̂ − ω_0` (same convention as syn-machine `δ`) so DAE initialisation at the locked operating point is well-posed. Also exposes the internal `v_d`, `v_q` in `h_dict` for diagnostics. |

### Composite

| Module                | Description                                                                                                  |
| --------------------- | ------------------------------------------------------------------------------------------------------------ |
| `stab_1lpf_2wo_3ll`   | Classic PSS / POD stabiliser chain: `lpf1 -> washout -> washout -> leadlag -> leadlag -> leadlag` with output saturation. Built by composing the basic blocks above; serves as the canonical example for wiring a multi-block composite. |

## Usage examples

### A standalone control loop

A simple PI in unity negative feedback, built from scratch:

```python
from pydae.core.builder.casadi_builder import MathBackend, CasadiBuilder
from pydae.core.model.casadi_model import CasadiModel
from pydae.control.continuos.pi import pi

dae = {'f_list': [], 'x_list': [],
       'g_list': [], 'y_ini_list': [], 'y_run_list': [],
       'u_ini_dict': {}, 'u_run_dict': {},
       'params_dict': {}, 'h_dict': {}, 'xy_0_dict': {}}

backend = MathBackend(use_casadi=True)
# u_pi = r - z_pi (negative-feedback error); the PI's output is z_pi.
pi(dae, 'ctrl', backend, {'K_p': 2.0, 'K_i': 0.5, 'z_min': -10, 'z_max': 10},
   input=['r', 'z_pi_ctrl'])

model = CasadiModel(CasadiBuilder(dae).build())
model.ini({}, dae['xy_0_dict'])
model.run(30.0, {'r': 1.0})
model.post()
```

### Wiring a PSS onto a power-system generator

```python
from pydae.bps import BpsBuilder
from pydae.control.continuos.stab_1lpf_2wo_3ll import stab_1lpf_2wo_3ll

grid = BpsBuilder('mygrid.hjson', use_casadi=True)
grid.construct('mygrid')

# Tap the generator's speed and inject the stabiliser output onto the
# AVR voltage reference. `input='omega_1'` aliases the existing state;
# `output='v_pss_1'` promotes the existing global input to an
# algebraic variable driven by the stabiliser.
stab_1lpf_2wo_3ll(grid.sys_dict, 'pss_1', grid.backend,
                  {'T_lpf': 0.02, 'T_wo1': 10.0, 'T_wo2': 10.0,
                   'T_1': 0.1, 'T_2': 0.02, 'T_3': 0.1, 'T_4': 0.02,
                   'T_5': 0.1, 'T_6': 0.02,
                   'K_stab': 5.0, 'Limit': 0.1},
                  input='omega_1', output='v_pss_1')
```

### Observer + state-feedback controller

A Kalman filter estimating the state of an LTI plant, paired with a
state-feedback gain (`u = -K x_hat`) through `matrix_gain`:

```python
from pydae.control.continuos.lti import lti
from pydae.control.continuos.kalman import kalman
from pydae.control.continuos.matrix_gain import matrix_gain

# Plant
lti(dae, 'plant', backend,
    {'A': A_plant, 'B': B_plant, 'C': C_plant, 'D': D_plant})

# Kalman estimator -- u comes from the plant's input symbol, y from
# the plant's measurement output expression in h_dict.
kalman(dae, 'kf', backend,
       {'A': A_plant, 'B': B_plant, 'C': C_plant, 'D': D_plant,
        'Q': Q, 'R': R},
       input=['u_1_lti_plant', 'z_1_lti_plant'])

# State feedback K: drives the plant's input.
matrix_gain(dae, 'sf', backend,
            {'K': [[-k1, -k2]]},
            input=['x_hat_1_kalman_kf', 'x_hat_2_kalman_kf'],
            output='u_1_lti_plant')
```

## API reference

```{eval-rst}
.. autosummary::
    :toctree: _autosummary
    :nosignatures:

    pydae.control.control_builder
    pydae.control.continuos.pi
    pydae.control.continuos.pid
    pydae.control.continuos.lpf1
    pydae.control.continuos.lpf2
    pydae.control.continuos.leadlag
    pydae.control.continuos.washout
    pydae.control.continuos.summation
    pydae.control.continuos.matrix_gain
    pydae.control.continuos.tf
    pydae.control.continuos.lti
    pydae.control.continuos.luenberger
    pydae.control.continuos.kalman
    pydae.control.continuos.saturation
    pydae.control.continuos.deadband
    pydae.control.continuos.rate_limiter
    pydae.control.continuos.integrator
    pydae.control.continuos.product
    pydae.control.continuos.notch
    pydae.control.continuos.park
    pydae.control.continuos.park_inv
    pydae.control.continuos.pll
    pydae.control.continuos.pi_feedforward
    pydae.control.continuos.stab_1lpf_2wo_3ll
```
