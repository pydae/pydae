# ieeeg1

*Turbine governors — pydae-bps model.*

## Model description

IEEE Type 1 (IEEEG1) steam turbine-governor (REE NTS parameter set).

The IEEEG1 model combines a speed-droop governor with a servo-actuated
main valve and a four-lag steam turbine whose output is tapped at four
points by coefficients $K_1 \dots K_8$. The HP fractions $K_1, K_3,
K_5, K_7$ drive the HP shaft and the LP fractions $K_2, K_4, K_6, K_8$
drive the LP shaft. For a single-mass generator (as in REE NTS) the
HP and LP taps are summed into a single mechanical power
$p_m = p_{m,HP} + p_{m,LP}$ that feeds the synchronous machine swing
equation.

**Signal path**

The speed deviation with respect to synchronism is

$$\Delta\omega = 1 - \omega$$

The summing junction combines droop, scheduled dispatch $p_c$, valve
feedback and AGC trim,

$$u_3 = K\,\Delta\omega + p_c - y_g + K_{sec}\, p_{agc}$$

The servo valve is a rate- and position-limited integrator with state
$x_3$,

$$u_g = \mathrm{sat}\!\left(u_3 / T_3,\; U_c,\; U_o\right)$$
$$\frac{d x_3}{dt} = u_g + K_{awu}\,(y_g - x_3)$$

where the anti-windup term drives the integrator state back into the
admissible range when the position limiter saturates,

$$y_g = \mathrm{sat}(x_3,\; P_{min},\; P_{max})$$

With the REE parameter set $T_1 = T_2 = 0$ so the optional lead-lag
block between the servo and the steam chest is a unity pass-through
and is not instantiated.

The steam turbine is a cascade of three first-order lags (the fourth
collapses because $T_7 = 0$ under REE; $K_7$ and $K_8$ coefficients,
if nonzero, are applied directly to $x_6$),

$$\frac{d x_4}{dt} = \frac{y_g - x_4}{T_4}$$
$$\frac{d x_5}{dt} = \frac{x_4 - x_5}{T_5}$$
$$\frac{d x_6}{dt} = \frac{x_5 - x_6}{T_6}$$

The mechanical output is the sum of HP and LP fractions,

$$p_{m,HP} = K_1 x_4 + K_3 x_5 + K_5 x_6 + K_7 x_6$$
$$p_{m,LP} = K_2 x_4 + K_4 x_5 + K_6 x_6 + K_8 x_6$$
$$0 = p_{m,HP} + p_{m,LP} - p_m$$

with $\mathrm{sat}(x, a, b) = \min\{b,\, \max\{a,\, x\}\}$.

**Steady-state relation**

At synchronism ($\omega = 1$, $\Delta\omega = 0$, $p_{agc} = 0$) the
servo integrator requires $u_g = 0 \Rightarrow u_3 = 0$, hence
$y_g = p_c$. The turbine cascade then gives
$x_4 = x_5 = x_6 = p_c$, and the mechanical output is
$p_m = (\sum K_i)\, p_c$ which equals $p_c$ when the HP+LP fractions
sum to unity — as in the REE parameter set
($K_1 + K_3 + K_5 = 1.0$, LP fractions zero).

**No ini/run swap**

Governors regulate mechanical power through speed feedback; they do
not pin a voltage setpoint, so the ``y_ini``/``y_run`` partitions are
identical. ``p_c`` is an input in both phases; ``p_m`` is solved
algebraically in both phases.

**Configuration**

Example data entry (REE NTS defaults)::

    "gov": {"type": "ieeeg1",
            "K": 20.0,
            "K_1": 0.3, "K_3": 0.3, "K_5": 0.4, "K_7": 0.0,
            "K_2": 0.0, "K_4": 0.0, "K_6": 0.0, "K_8": 0.0,
            "T_1": 0.0, "T_2": 0.0, "T_3": 0.1,
            "T_4": 0.3, "T_5": 7.0, "T_6": 0.6, "T_7": 0.0,
            "U_o": 0.5, "U_c": -0.5,
            "P_max": 1.0, "P_min": 0.0,
            "p_c": 0.8}

The ``p_c`` field is the scheduled dispatch — the steady-state
mechanical output when $\omega = 1$ and $p_{agc} = 0$. It is used as
both the ``u_ini`` and ``u_run`` value, and seeds the turbine-cascade
states.

## Usage

```python
from pydae.bps import BpsBuilder

grid = BpsBuilder("my_network.json")
grid.construct("my_system")
```

The `ieeeg1` model is instantiated by including an entry in the relevant
section of the network JSON (see [Overview](../../overview.md)).

## Parameters, inputs, states, outputs

### Parameters

| Symbol | Variable | Default | Units | Description |
|---|---|---|---|---|
| $K$ | `K_gov` | 20.0 | pu | Speed-droop gain (1/R). REE NTS default corresponds to 5% droop. |
| $K_1$ | `K_1_gov` | 0.3 | pu | HP fraction at steam-chest tap (after T_4) |
| $K_3$ | `K_3_gov` | 0.3 | pu | HP fraction at reheater tap (after T_5) |
| $K_5$ | `K_5_gov` | 0.4 | pu | HP fraction at crossover tap (after T_6) |
| $K_7$ | `K_7_gov` | 0.0 | pu | HP fraction at LP-turbine tap (after T_7). Applied directly to x_6 when T_7 = 0. |
| $K_2$ | `K_2_gov` | 0.0 | pu | LP fraction at steam-chest tap |
| $K_4$ | `K_4_gov` | 0.0 | pu | LP fraction at reheater tap |
| $K_6$ | `K_6_gov` | 0.0 | pu | LP fraction at crossover tap |
| $K_8$ | `K_8_gov` | 0.0 | pu | LP fraction at LP-turbine tap |
| $T_1$ | `T_1_gov` | 0.0 | s | Lead-lag numerator. Zero under REE so the lead-lag is unity and is not instantiated. |
| $T_2$ | `T_2_gov` | 0.0 | s | Lead-lag denominator. Zero under REE — lead-lag not instantiated. |
| $T_3$ | `T_3_gov` | 0.1 | s | Servo (valve actuator) time constant |
| $T_4$ | `T_4_gov` | 0.3 | s | Steam-chest time constant |
| $T_5$ | `T_5_gov` | 7.0 | s | Reheater time constant |
| $T_6$ | `T_6_gov` | 0.6 | s | Crossover time constant |
| $T_7$ | `T_7_gov` | 0.0 | s | LP-turbine time constant. Zero under REE — x_7 collapses into x_6. |
| $U_o$ | `U_o_gov` | 0.5 | pu/s | Servo opening rate limit |
| $U_c$ | `U_c_gov` | -0.5 | pu/s | Servo closing rate limit |
| $P_{max}$ | `P_max_gov` | 1.0 | pu | Upper valve position limit |
| $P_{min}$ | `P_min_gov` | 0.0 | pu | Lower valve position limit |

### Inputs

| Symbol | Variable | Default | Units | Description |
|---|---|---|---|---|
| $p_c$ | `p_c` | 0.8 | pu | Scheduled dispatch (load reference). Equals p_m at steady state when omega = 1 and p_agc = 0. |

### Dynamic States

| Symbol | Variable | Default | Units | Description |
|---|---|---|---|---|
| $x_3$ | `x_3_gov` |  | pu | Servo (valve actuator) integrator state |
| $x_4$ | `x_4_gov` |  | pu | Steam-chest lag state |
| $x_5$ | `x_5_gov` |  | pu | Reheater lag state |
| $x_6$ | `x_6_gov` |  | pu | Crossover lag state |

### Algebraic States

| Symbol | Variable | Default | Units | Description |
|---|---|---|---|---|
| $p_m$ | `p_m` |  | pu | Mechanical power delivered to the synchronous machine swing equation (HP + LP tap sum). |


## Source

- Module: `pydae.bps.govs.ieeeg1`
- File: [`packages/pydae-bps/src/pydae/bps/govs/ieeeg1.py`](https://github.com/pydae/pydae/tree/main/packages/pydae-bps/src/pydae/bps/govs/ieeeg1.py)
