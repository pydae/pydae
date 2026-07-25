# pmsm_pq_ss

*Wind energy conversion systems — pydae-bps model.*

## Model description

PMSM full-converter wind turbine — simplified p/q model.

A reduced version of :func:`pmsm_1` that keeps the **wind-turbine dynamics**
(aerodynamics, 2-mass drivetrain, pitch control, MPPT) but replaces the whole
**electrical stack** (PMSM machine equations, machine-side VSC, DC-link, PLL,
grid-side current control, L-filter) with a purely **algebraic grid-following
p/q injection** in stationary (r-i) coordinates — exactly the simplification
``pv_pq_ss`` applies to ``pv_dq_ss`` on the PV side.

The available active power comes from the wind through the MPPT reference; the
grid injection is current-limited (smooth saturation) and supports a VRT blend
into an arbitrary current-injection mode.  An optional state-space low-pass
filter can shape the reactive-power reference.

**Wind / mechanical side (retained)**

$$\nu_w = \nu_{w,0} + \nu_{w,\text{ramp}}, \qquad
  \lambda = \Lambda_b\,\frac{\omega_t/\Omega_{t,b}}{\nu_w/N_{u_w,b}}$$
$$c_p = C_1(C_2/\lambda_i - C_3\beta - C_4)e^{-C_5/\lambda_i} + C_6\lambda,
  \qquad p_w = K_{pow}\,\frac{c_p}{c_{p,b}}\left(\frac{\nu_w}{N_{u_w,b}}\right)^3$$
$$2H_t\,\dot\omega_t = p_w - K_{tr}\theta_{tr} - D_{tr}(\omega_t-\omega_r), \quad
  2H_r\,\dot\omega_r = K_{tr}\theta_{tr} + D_{tr}(\omega_t-\omega_r) - p_r$$

MPPT sets the electrical (rotor-braking) power reference:

$$p_{mppt} = K_{mppt3}\,\omega_t^3, \qquad
  T_{mppt}\,\dot p_{w,mppt} = p_{mppt} + p_{ref,ext} - p_{w,mppt}, \qquad
  p_r = p_{w,mppt}$$

**Grid injection (algebraic p/q, from pv_pq_ss)**

$$p_{s,ref} = p_{w,mppt}, \qquad q_{s,ref} = q_{s,ppc}$$
$$i_{sr,pq} = \frac{p_{s,ref} v_{sr} + q_{s,ref} v_{si}}{v_{sr}^2+v_{si}^2+\epsilon},
  \quad
  i_{si,pq} = \frac{p_{s,ref} v_{si} - q_{s,ref} v_{sr}}{v_{sr}^2+v_{si}^2+\epsilon}$$
$$i_{mod} = \sqrt{i_{sr}^2+i_{si}^2+\epsilon}, \quad
  i_{mod,sat} = \tfrac12\!\left(i_{mod}+I_{max}-\sqrt{(i_{mod}-I_{max})^2+\epsilon}\right)$$
$$0 = v_{sr} i_{sr} + v_{si} i_{si} - p_s, \qquad
  0 = v_{si} i_{sr} - v_{sr} i_{si} - q_s$$

## Usage

```python
from pydae.bps import BpsBuilder

grid = BpsBuilder("my_network.json")
grid.construct("my_system")
```

The `pmsm_pq_ss` model is instantiated by including an entry in the relevant
section of the network JSON (see [Overview](../../overview.md)).

## Parameters, inputs, states, outputs

### Parameters

| Symbol | Variable | Default | Units | Description |
|---|---|---|---|---|
| $S_n$ | `S_n` | 3000000.0 | VA | Nominal power |
| $F_n$ | `F_n` | 50.0 | Hz | Nominal frequency |
| $H_t$ | `H_t` | 4.0 | s | Turbine inertia constant |
| $H_r$ | `H_r` | 0.8 | s | Rotor (generator) inertia constant |
| $w_{tr}$ | `w_tr` | 2.0 | Hz | Drivetrain torsional frequency |
| $d_{tr}$ | `d_tr` | 0.1 | Hz | Drivetrain torsional damping |
| $K_{mppt3}$ | `K_mppt3` | 0.4 | pu | MPPT cubic gain (p = K_mppt3·ω_t³) |
| $T_{mppt}$ | `T_mppt` | 5.0 | s | MPPT low-pass time constant |
| $I_{max}$ | `I_max` | 1.2 | pu | Maximum current magnitude |
| $T_{lvrt}$ | `T_lvrt` | 0.02 | s | LVRT ramp time constant |
| $\epsilon$ | `Epsilon` | 1e-08 | - | Regularization for soft saturation |

### Inputs

| Symbol | Variable | Default | Units | Description |
|---|---|---|---|---|
| $\nu_{w,0}$ | `nu_w_0` | 10.0 | m/s | Base wind speed |
| $q_{s,ppc}$ | `q_s_ppc` | 0.0 | pu | Reactive power reference from PPC |
| $p_{ref,ext}$ | `p_ref_ext` | 0.0 | pu | External active-power adjustment (≤0 curtails) |
| $\beta_{ext}$ | `beta_ext` | 0.0 | deg | External pitch-angle adder |
| $\text{lvrt}_{ext}$ | `lvrt_ext` | 0.0 | - | External LVRT trigger (0=PQ, 1=arbitrary) |
| $i_{sa,ref}$ | `i_sa_ref` | 0.0 | pu | Active current reference (arbitrary mode) |
| $i_{sr,ref}$ | `i_sr_ref` | 0.0 | pu | Reactive current reference (arbitrary mode) |

### Dynamic States

| Symbol | Variable | Default | Units | Description |
|---|---|---|---|---|
| $\nu_{w,ramp}$ | `nu_w_ramp` |  | m/s | Wind-speed ramp state |
| $\theta_{tr}$ | `theta_tr` |  | rad | Drivetrain torsion angle |
| $\omega_t$ | `omega_t` |  | pu | Turbine speed |
| $\omega_r$ | `omega_r` |  | pu | Rotor (generator) speed |
| $\xi_\beta$ | `xi_beta` |  | - | Pitch PI integrator |
| $\beta$ | `beta` |  | deg | Pitch angle |
| $p_{w,mppt}$ | `p_w_mppt_lpf` |  | pu | Filtered MPPT power reference |

### Algebraic States

| Symbol | Variable | Default | Units | Description |
|---|---|---|---|---|
| $p_s$ | `p_s` |  | pu | Injected active power |
| $q_s$ | `q_s` |  | pu | Injected reactive power |

### Outputs

| Symbol | Variable | Default | Units | Description |
|---|---|---|---|---|
| $\nu_w$ | `nu_w` |  | m/s | Wind speed |
| $p_w$ | `p_w` |  | pu | Aerodynamic power |
| $i_{mod,sat}$ | `i_mod_sat` |  | pu | Saturated current magnitude |


## Source

- Module: `pydae.bps.wecs.pmsm_pq_ss`
- File: [`packages/pydae-bps/src/pydae/bps/wecs/pmsm_pq_ss.py`](https://github.com/pydae/pydae/tree/main/packages/pydae-bps/src/pydae/bps/wecs/pmsm_pq_ss.py)
