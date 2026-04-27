# ntsieeeg1

*Turbine governors — pydae-bps model.*

## Model description

NTS variant of the IEEEG1 multi-stage steam turbine-governor.

Functionally identical to [ieeeg1](ieeeg1.md) but uses slightly different
code structure (old-style function signature, no `descriptions()` table).
The REE NTS default parameter set applies: $T_1 = T_2 = 0$ (lead-lag unity),
$T_7 = 0$ (fourth lag collapses), and $K_2 = K_4 = K_6 = K_8 = 0$ (pure HP
output).

**Signal path**

Speed deviation:

$$\Delta\omega = 1 - \omega$$

Servo summing junction with secondary control:

$$u_3 = K\,\Delta\omega + p_c - y_g + K_{sec}\,p_{agc}$$

Servo valve (rate/position-limited integrator), $y_g = \mathrm{sat}(x_3, P_{min}, P_{max})$:

$$u_g = \mathrm{sat}\!\left(\frac{u_3}{T_3},\; U_c,\; U_o\right), \qquad
  \frac{d x_3}{dt} = u_g + K_{awu}\,(y_g - x_3)$$

Steam turbine three-stage cascade:

$$\frac{d x_4}{dt} = \frac{y_g - x_4}{T_4}, \quad
  \frac{d x_5}{dt} = \frac{x_4 - x_5}{T_5}, \quad
  \frac{d x_6}{dt} = \frac{x_5 - x_6}{T_6}$$

Mechanical power (HP taps only with NTS defaults):

$$0 = -p_m + K_1\,x_4 + K_3\,x_5 + K_5\,x_6$$

**Steady-state relation**

At $\omega = 1$, $p_{agc} = 0$: $y_g = p_c$, all cascade states equal $p_c$,
$p_m = (K_1 + K_3 + K_5)\,p_c$.  With NTS defaults $K_1 + K_3 + K_5 = 1$:
$p_m = p_c$.

See [ieeeg1](ieeeg1.md) for the full IEEE Type 1 model with LP taps and the
complete parameter table.

## Usage

```hjson
gov: {
  type: "ntsieeeg1",
  K: 20,
  K_1: 0.3, K_3: 0.3, K_5: 0.4, K_7: 0,
  K_2: 0,   K_4: 0,   K_6: 0,   K_8: 0,
  T_1: 0,   T_2: 0,
  T_3: 0.1, T_4: 0.3, T_5: 7.0, T_6: 0.6, T_7: 0,
  U_0: 0.5, U_c: -0.5,
  p_c: 0.8
}
```

## Parameters, inputs, states, outputs

### Parameters

| Symbol | Variable | Default | Units | Description |
|---|---|---|---|---|
| $K$ | `K` | 20.0 | pu | Speed-droop gain ($1/R$). $K=20$ → 5 % droop. |
| $K_1$ | `K_1` | 0.3 | pu | HP fraction at steam-chest tap ($x_4$) |
| $K_3$ | `K_3` | 0.3 | pu | HP fraction at reheater tap ($x_5$) |
| $K_5$ | `K_5` | 0.4 | pu | HP fraction at crossover tap ($x_6$) |
| $K_7$ | `K_7` | 0.0 | pu | HP fraction at LP-turbine tap (NTS: 0) |
| $K_2,K_4,K_6,K_8$ | `K_2`…`K_8` | 0.0 | pu | LP fractions (NTS: 0) |
| $T_1,T_2$ | `T_1`, `T_2` | 0.0 | s | Lead-lag (NTS: 0 → unity) |
| $T_3$ | `T_3` | 0.1 | s | Servo (valve actuator) time constant |
| $T_4$ | `T_4` | 0.3 | s | Steam-chest time constant |
| $T_5$ | `T_5` | 7.0 | s | Reheater time constant |
| $T_6$ | `T_6` | 0.6 | s | Crossover time constant |
| $T_7$ | `T_7` | 0.0 | s | LP-turbine time constant (NTS: 0) |
| $U_o$ | `U_0` | 0.5 | pu/s | Servo opening rate limit |
| $U_c$ | `U_c` | -0.5 | pu/s | Servo closing rate limit |
| $P_{max}$ | `P_max` | 1.0 | pu | Upper valve position limit |
| $P_{min}$ | `P_min` | 0.0 | pu | Lower valve position limit |

### Inputs

| Symbol | Variable | Default | Units | Description |
|---|---|---|---|---|
| $p_c$ | `p_c` | 0.8 | pu | Scheduled dispatch. Equals $p_m$ at steady state when $\omega=1$, $p_{agc}=0$ (with NTS defaults). |

### Dynamic States

| Symbol | Variable | Units | Description |
|---|---|---|---|
| $x_3$ | `x_3_gov` | pu | Servo (valve actuator) integrator state |
| $x_4$ | `x_4_gov` | pu | Steam-chest lag state |
| $x_5$ | `x_5_gov` | pu | Reheater lag state |
| $x_6$ | `x_6_gov` | pu | Crossover lag state |

### Algebraic States

| Symbol | Variable | Units | Description |
|---|---|---|---|
| $p_m$ | `p_m` | pu | Mechanical power delivered to the synchronous machine |

## Source

- Module: `pydae.bps.govs.ntsieeeg1`
- File: [`packages/pydae-bps/src/pydae/bps/govs/ntsieeeg1.py`](https://github.com/pydae/pydae/tree/main/packages/pydae-bps/src/pydae/bps/govs/ntsieeeg1.py)
