# gensal

*Synchronous machines — pydae-bps model.*

Salient-pole 5th-order synchronous machine — IEEE 1110-2019 Model 2.1 /
PSS/E `GENSAL`. The salient-pole counterpart of
[`genrou`](genrou.md). Used for hydro units.

## What differs from `genrou`

| | `genrou` (round rotor) | `gensal` (salient pole) |
|---|---|---|
| States | 6 (δ, ω, e'_q, e'_d, e''_q, e''_d) | 5 (δ, ω, e'_q, e''_q, e''_d) |
| q-axis field winding | present → transient EMF e'_d | **none** — no q-axis transient stage |
| q-axis subtransient | T''_{q0} de''_d/dt = −e''_d + e'_d + (X'_q − X''_q) i_q | T''_{q0} de''_d/dt = −e''_d + (X_q − X''_q) i_q |
| Reactance pairing | X_q ≈ X_d, X'_q present | X_q < X_d (salient), no X'_q |
| Saturation | both axes: S_d = S(ψ_{AT}), S_q = (X_q/X_d) S(ψ_{AT}) | d-axis only: S_d = S(e'_q), S_q = 0 |

Identical to `genrou`:

- Stator algebraic equations (terminal-referred subtransient, no $X_l$
  field).
- Swing equation, COI accumulation, AVR/governor/PSS/LC ports.
- Saturation helper (two-point fit from $S_{1.0}$, $S_{1.2}$).

## Equations

### Auxiliary

$$v_d = V \sin(\delta - \theta), \quad v_q = V \cos(\delta - \theta)$$
$$\tau_e = (v_d + R_a i_d)\,i_d + (v_q + R_a i_q)\,i_q$$
$$\omega_s = \omega_{coi}$$

### Saturation (d-axis only — PSS/E GENSAL convention)

$$\psi_{AT} = \sqrt{e_q'^{\,2} + \epsilon}, \quad
  S(\psi_{AT}) = \frac{B_{sat}\,\max(\psi_{AT} - A_{sat},\,0)^2}{\psi_{AT}}$$
$$S_d = S(\psi_{AT}), \qquad S_q = 0$$

with $B_{sat}$, $A_{sat}$ from the standard $S_{1.0}$ / $S_{1.2}$
two-point fit (same helper as in `genrou`).

### Dynamic equations (5 states)

$$\frac{d\delta}{dt} = \Omega_b (\omega - \omega_s) - K_\delta \delta$$
$$\frac{d\omega}{dt} = \frac{1}{2H}\bigl(\tau_m - \tau_e - D(\omega - \omega_s)\bigr)$$
$$T_{d0}' \frac{de_q'}{dt} = -e_q' - (X_d - X_d')\,i_d - S_d\,e_q' + v_f$$
$$T_{d0}'' \frac{de_q''}{dt} = -e_q'' + e_q' - (X_d' - X_d'')\,i_d$$
$$T_{q0}'' \frac{de_d''}{dt} = -e_d'' + (X_q - X_q'')\,i_q$$

### Algebraic equations

Same as `genrou` (no $X_l$ subtraction):

$$0 = v_q + R_a i_q - e_q'' + X_d''\,i_d$$
$$0 = v_d + R_a i_d - e_d'' - X_q''\,i_q$$
$$0 = i_d v_d + i_q v_q - p_g$$
$$0 = i_d v_q - i_q v_d - q_g$$

## Choosing between gensal and genrou

| | gensal | genrou |
|---|---|---|
| Use for | hydro generators (salient poles) | turbo-alternators (round rotor) |
| Typical $X_q / X_d$ | 0.5 – 0.7 | ≈ 1.0 |
| Q-axis field | none | present |
| Saturation | d-axis only | both axes |

If your parameter source publishes a q-axis transient time constant
$T_{q0}'$ and reactance $X_q'$, the machine is a round rotor — use
`genrou`. Salient-pole data sets characteristically omit both.

## Usage

```python
from pydae.bps import BpsBuilder

grid = BpsBuilder("my_network.json")
grid.construct("my_system")
```

Network HJSON entry:

```hjson
syns: [
  {bus: "1", S_n: 500e6, type: "gensal",
   X_d: 1.10, X_q: 0.70,
   X1d: 0.25, T1d0: 8.0,
   X2d: 0.20, T2d0: 0.03,
   X2q: 0.20, T2q0: 0.05,
   R_a: 0.0, H: 4.0, D: 0.0,
   S_10: 0.10, S_12: 0.30,
   F_n: 50.0, K_sec: 0.0, K_delta: 0.0,
   avr: {...}, gov: {...}, pss: {...}, lc: {...}}
]
```

The AVR / governor / PSS / LC ports are the same as `genrou`; controllers
written for round-rotor machines drop in unchanged.

## Parameters, inputs, states, outputs

### Parameters

| Symbol | Variable | Default | Units | Description |
|---|---|---|---|---|
| $S_n$ | `S_n` | 100000000.0 | VA | Nominal power |
| $F_n$ | `F_n` | 50.0 | Hz | Nominal frequency |
| $H$ | `H` | 4.0 | s | Inertia constant |
| $D$ | `D` | 0.0 | - | Damping coefficient |
| $X_d$ | `X_d` | 1.10 | pu-m | d-axis synchronous reactance |
| $X_q$ | `X_q` | 0.70 | pu-m | q-axis synchronous reactance (salient: $X_q < X_d$) |
| $X'_d$ | `X1d` | 0.25 | pu-m | d-axis transient reactance |
| $X''_d$ | `X2d` | 0.20 | pu-m | d-axis subtransient reactance (terminal-referred) |
| $X''_q$ | `X2q` | 0.20 | pu-m | q-axis subtransient reactance (terminal-referred) |
| $T'_{d0}$ | `T1d0` | 8.0 | s | d-axis open-circuit transient time constant |
| $T''_{d0}$ | `T2d0` | 0.03 | s | d-axis open-circuit subtransient time constant |
| $T''_{q0}$ | `T2q0` | 0.05 | s | q-axis open-circuit subtransient time constant |
| $R_a$ | `R_a` | 0.0 | pu-m | Armature resistance |
| $S_{1.0}$ | `S_10` | 0.0 | - | Saturation factor at E=1.0 |
| $S_{1.2}$ | `S_12` | 0.0 | - | Saturation factor at E=1.2 |
| $K_{\delta}$ | `K_delta` | 0.0 | - | Reference-machine constant |
| $K_{sec}$ | `K_sec` | 0.0 | - | Secondary-frequency control participation |

### Inputs

| Symbol | Variable | Default | Units | Description |
|---|---|---|---|---|
| $p_m$ | `p_m` | 0.5 | pu-m | Mechanical power |
| $v_f$ | `v_f` | 1.0 | pu-m | Field voltage |

### Dynamic States

| Symbol | Variable | Units | Description |
|---|---|---|---|
| $\delta$ | `delta` | rad | Rotor angle |
| $\omega$ | `omega` | pu | Rotor speed |
| $e'_q$ | `e1q` | pu-m | d-axis transient EMF (d-axis flux) |
| $e''_q$ | `e2q` | pu-m | q-axis subtransient EMF |
| $e''_d$ | `e2d` | pu-m | d-axis subtransient EMF |

### Algebraic States

| Symbol | Variable | Units | Description |
|---|---|---|---|
| $i_d$ | `i_d` | pu-m | d-axis current |
| $i_q$ | `i_q` | pu-m | q-axis current |
| $p_g$ | `p_g` | pu-m | Active-power injection (machine base) |
| $q_g$ | `q_g` | pu-m | Reactive-power injection (machine base) |

### Outputs

| Symbol | Variable | Units | Description |
|---|---|---|---|
| $\tau_e$ | `p_e` | pu-m | Air-gap (electrical) torque |
| $v_f$ | `v_f` | pu-m | Field voltage (echo) |
| $v_d$ | `v_d` | pu-m | d-axis terminal voltage |
| $v_q$ | `v_q` | pu-m | q-axis terminal voltage |
| $S_{at}$ | `S_at` | - | Saturation factor $S(\psi_{AT})$ — d-axis only |

## References

- IEEE Std 1110-2019, *IEEE Guide for Synchronous Generator Modeling
  Practices and Parameter Verification with Applications in Power
  System Stability Analyses*, §6.3 (Model 2.1, salient pole).
- IEEE Std 115-2019, *IEEE Guide for Test Procedures for Synchronous
  Machines*.
- PSS/E Model Library, `GENSAL`.
- Kundur, P., *Power System Stability and Control*, §3.6.

## Source

- Module: `pydae.bps.syns.gensal`
- File: [`packages/pydae-bps/src/pydae/bps/syns/gensal.py`](https://github.com/pydae/pydae/tree/main/packages/pydae-bps/src/pydae/bps/syns/gensal.py)
