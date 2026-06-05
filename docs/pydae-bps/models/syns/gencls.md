# gencls

*Synchronous machines — pydae-bps model.*

Classical 2nd-order synchronous machine — PSS/E `GENCLS`. The simplest
useful machine model: just rotor swing dynamics with a **constant
internal voltage** $E$ behind a **single transient reactance** $X'$.
No field winding, no AVR, no damper windings, no saturation, no
saliency.

## When to use

| | gencls | gensal | genrou |
|---|---|---|---|
| Order | 2 | 5 | 6 |
| States | δ, ω | δ, ω, e'_q, e''_q, e''_d | δ, ω, e'_q, e'_d, e''_q, e''_d |
| Field winding | none (E constant) | yes | yes |
| Damper windings | none | one per axis | two per axis |
| Saturation | no | d-axis | both axes |
| AVR / PSS attaches? | **no** (no v_f input) | yes | yes |
| Use for | first-swing studies, distant equivalents, swing-eq benchmarks | hydro (salient) | thermal / turbo (round rotor) |

`gencls` has no $v_f$ input — there is no field winding to drive. Trying
to attach an AVR makes no sense for this model.

## Equations

### Auxiliary

$$v_d = V \sin(\delta - \theta), \quad v_q = V \cos(\delta - \theta)$$
$$\tau_e = (v_d + R_a i_d)\,i_d + (v_q + R_a i_q)\,i_q$$
$$\omega_s = \omega_{coi}$$

### Dynamic equations (2 states)

$$\frac{d\delta}{dt} = \Omega_b (\omega - \omega_s) - K_\delta \delta$$
$$\frac{d\omega}{dt} = \frac{1}{2H}\bigl(\tau_m - \tau_e - D(\omega - \omega_s)\bigr)$$

### Algebraic equations

Constant internal voltage $E = e_q^{(1)}$ behind a single transient
reactance $X'$. No saliency: $X_d' = X_q' = X'$.

$$0 = v_q + R_a i_q - E + X'\,i_d$$
$$0 = v_d + R_a i_d - X'\,i_q$$
$$0 = i_d v_d + i_q v_q - p_g$$
$$0 = i_d v_q - i_q v_d - q_g$$

## Compared with `milano2ord`

`milano2ord` carries distinct $X_d'$ and $X_q'$ — i.e. it allows
*transient saliency* even in a classical-model framework. `gencls` is
the strict PSS/E `GENCLS` form with one reactance. When
$X_q' = X_d'$ they coincide; the test suite enforces this parity to
machine precision.

## Usage

Network HJSON entry:

```hjson
syns: [
  {bus: "1", S_n: 100e6, type: "gencls",
   X1d: 0.30,
   R_a: 0.0, H: 5.0, D: 0.0,
   F_n: 50.0, K_sec: 0.0, K_delta: 0.0,
   e1q: 1.2, p_m: 0.5}
]
```

`e1q` is the constant internal voltage magnitude — pin it to the value
you obtain from a power-flow solution. `p_m` is the mechanical power.
A governor + LC can still drive `p_m`; no AVR/PSS can attach.

## Parameters, inputs, states, outputs

### Parameters

| Symbol | Variable | Default | Units | Description |
|---|---|---|---|---|
| $S_n$ | `S_n` | 100000000.0 | VA | Nominal power |
| $F_n$ | `F_n` | 50.0 | Hz | Nominal frequency |
| $H$ | `H` | 5.0 | s | Inertia constant |
| $D$ | `D` | 0.0 | - | Damping coefficient |
| $X'$ | `X1d` | 0.30 | pu-m | Single transient reactance ($X_d' = X_q' = X'$) |
| $R_a$ | `R_a` | 0.0 | pu-m | Armature resistance |
| $K_{\delta}$ | `K_delta` | 0.0 | - | Reference-machine constant |
| $K_{sec}$ | `K_sec` | 0.0 | - | Secondary-frequency control participation |

### Inputs

| Symbol | Variable | Default | Units | Description |
|---|---|---|---|---|
| $p_m$ | `p_m` | 0.5 | pu-m | Mechanical power |
| $E$ | `e1q` | 1.0 | pu-m | Internal voltage magnitude (constant) |

### Dynamic States

| Symbol | Variable | Units | Description |
|---|---|---|---|
| $\delta$ | `delta` | rad | Rotor angle |
| $\omega$ | `omega` | pu | Rotor speed |

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
| $v_d$ | `v_d` | pu-m | d-axis terminal voltage |
| $v_q$ | `v_q` | pu-m | q-axis terminal voltage |

## References

- PSS/E Model Library, `GENCLS`.
- Anderson, P. M., Fouad, A. A., *Power System Control and Stability*,
  2nd ed., IEEE Press, 2003, §2.6 (Classical model).
- Kundur, P., *Power System Stability and Control*, §13.1.

## Source

- Module: `pydae.bps.syns.gencls`
- File: [`packages/pydae-bps/src/pydae/bps/syns/gencls.py`](https://github.com/pydae/pydae/tree/main/packages/pydae-bps/src/pydae/bps/syns/gencls.py)
