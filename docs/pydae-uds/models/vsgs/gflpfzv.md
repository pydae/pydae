# gflpfzv

*Virtual synchronous generators — pydae-uds model.*

## Model description

Grid-following + Primary-Frequency-Response VSG with LPF and per-phase
Q-PI control.

Hosts inside an `ac_3ph_4w` VSC entry (see HJSON below). The VSG generates
the converter terminal EMFs $e_{ao}, e_{bo}, e_{co}$ from a rotating frame
$\phi(t)$ plus a per-phase magnitude that's slowly tracked by an outer-loop
PI on the **positive-sequence** reactive power.

**Symmetrical-component decomposition** of the grid-side voltage and
current ($\alpha = e^{j2\pi/3}$, in real form for backend portability):

$$\begin{bmatrix}v_0 \\ v_+ \\ v_-\end{bmatrix} =
\frac{1}{3}\begin{bmatrix}
 1 & 1 & 1 \\
 1 & \alpha & \alpha^2 \\
 1 & \alpha^2 & \alpha
\end{bmatrix}
\begin{bmatrix}v_a \\ v_b \\ v_c\end{bmatrix}$$

$$p_+ = 3\,\Re(v_+ \overline{i_+}), \qquad
  q_+ = 3\,\Im(v_+ \overline{i_+})$$

**Dynamic states** — swing equation with droop + power LPF:

$$\dot{\phi} = \Omega_b (\omega - \omega_{coi}) - K_\delta \phi$$
$$\dot{\omega} = \frac{1}{2H}\bigl(p_m - p_{ef} - D(\omega - 1)\bigr)$$
$$\dot{p_{ef}} = \frac{1}{T_e}\!\left(\frac{p_+}{S_n} - p_{ef}\right)$$
$$\dot{p_{cf}} = \frac{1}{T_c}\!\bigl(p_c - p_{cf}\bigr)$$
$$\dot{p_{pfr}} = \frac{1}{T_{pfr}}\!\left(\frac{1}{R}(\omega - \omega_{ref}) - p_{pfr}\right)$$

**Reactive PI with saturation + conditional-integration anti-windup**:

$$\varepsilon_q = q_{ref} - q_+/S_n$$
$$\Delta_{q,\text{nosat}} = K_{qp}\varepsilon_q + K_{qi}\xi_q$$
$$\Delta_q = \text{clip}(\Delta_{q,\text{nosat}}, -0.05\,U_n, +0.05\,U_n)$$
$$\dot{\xi_q} = K_{qaw}\varepsilon_q - (1 - K_{qaw})\xi_q$$

where $K_{qaw} \in \{0, 1\}$ is the anti-windup indicator (0 once saturated).
The saturation uses `bk.hard_limits` (so it codegens cleanly on both
backends); the indicator stays on `bk.Piecewise` because it's a true step.

**Per-phase EMF amplitude** — each phase has its own $\Delta e_{*o,m}$
state tracking $v_{*r} + \Delta_q$ with first-order lag $T_v$:

$$\dot{\Delta e_{ao,m}} = \frac{1}{T_v}\bigl(v_{ra} + \Delta_q - \Delta e_{ao,m}\bigr)$$

**Virtual-impedance terminal voltage** (writes the EMFs back into the
host VSC's $v_{t*}$ algebraic vars):

$$0 = e_{*o} - (R_v + jX_v) i_{s\varphi} - v_{t\varphi}$$

**HJSON snippet** (nested under an `ac_3ph_4w` VSC):

```hjson
vsg: {type: "gflpfzv", bus: "A3", S_n: 100e3, U_n: 400,
      R_v: 0, X_v: 0.1,
      H: 5.0, D: 0.1, T_e: 0.1, T_c: 0.1, T_v: 0.1, T_pfr: 0.2,
      Droop: 0.05, K_qp: 0.01, K_qi: 0.01,
      K_agc: 0.0, K_delta: 0.0, K_sec: 0.0}
```

## Usage

```python
from pydae.uds import UdsBuilder

grid = UdsBuilder("my_network.hjson")
grid.construct("my_system")
```

The `gflpfzv` model is instantiated by including an entry in the relevant
section of the network HJSON (see [Overview](../../overview.md)).

## Parameters, inputs, states, outputs

### Parameters

| Symbol | Variable | Default | Units | Description |
|---|---|---|---|---|
| $S_n$ | `S_n_{name}` |  | VA | Nominal apparent power |
| $U_n$ | `U_n_{name}` |  | V | Nominal line-to-line voltage |
| $H$ | `H_{name}` |  | s | Virtual inertia constant |
| $D$ | `D_{name}` |  | - | Damping coefficient |
| $T_e$ | `T_e_{name}` |  | s | Power-measurement LPF time constant |
| $T_c$ | `T_c_{name}` |  | s | Command LPF time constant |
| $T_v$ | `T_v_{name}` |  | s | Voltage-amplitude LPF time constant |
| $T_{pfr}$ | `T_pfr_{name}` |  | s | PFR LPF time constant |
| $R$ | `Droop_{name}` |  | - | PFR droop (pu/pu) |
| $K_{qp}$ | `K_qp_{name}` |  | - | Reactive-PI proportional gain |
| $K_{qi}$ | `K_qi_{name}` |  | 1/s | Reactive-PI integral gain |
| $K_{agc}$ | `K_agc_{name}` |  | - | AGC participation factor |
| $K_\delta$ | `K_delta_{name}` |  | 1/s | Angle reference-pull gain |
| $R_v$ | `R_v_{name}` |  | pu | Virtual resistance |
| $X_v$ | `X_v_{name}` |  | pu | Virtual reactance |

### Inputs

| Symbol | Variable | Default | Units | Description |
|---|---|---|---|---|
| $p_c$ | `p_c_{name}` | 0.0 | pu | Active-power command |
| $q_{ref}$ | `q_ref_{name}` | 0.0 | pu | Reactive-power reference |
| $\omega_{ref}$ | `omega_{name}_ref` | 1.0 | pu | Frequency reference for PFR |
| $e_{\varphi o}^m$ | `e_{a,b,c}o_m_{name}` | V_n/\sqrt{3} | V | Per-phase EMF magnitude (set-point) |
| $\phi_\varphi$ | `phi_{a,b,c,n}_{name}` | 0.0 | rad | Per-phase angle offset |
| $v_{r\varphi}$ | `v_r{a,b,c,n}_{name}` | 0.0 | V | Per-phase voltage residual injection |

### Dynamic States

| Symbol | Variable | Default | Units | Description |
|---|---|---|---|---|
| $\phi$ | `phi_{name}` |  | rad | Internal swing angle |
| $\omega$ | `omega_{name}` | 1.0 | pu | Rotor (virtual) speed |
| $\xi_q$ | `xi_q_{name}` |  | pu·s | Reactive-PI integrator state |
| $p_{ef}$ | `p_ef_{name}` |  | pu | Filtered positive-sequence active power |
| $p_{cf}$ | `p_cf_{name}` |  | pu | Filtered active-power command |
| $p_{pfr}$ | `p_pfr_{name}` |  | pu | Filtered PFR output |
| $\Delta e_{\varphi o,m}$ | `De_{a,b,c,n}o_m_{name}` |  | V | Per-phase EMF amplitude correction |

### Algebraic States

| Symbol | Variable | Default | Units | Description |
|---|---|---|---|---|
| $p_m$ | `p_m_{name}` |  | pu | Mechanical power balance variable |
| $v_{t\varphi}^{r,i}$ | `v_t{a,b,c}_{r,i}_{name}` |  | V | Per-phase terminal voltage (host VSC's input turned into state) |

### Outputs

| Symbol | Variable | Default | Units | Description |
|---|---|---|---|---|
| $p_+$ | `p_pos_{name}` |  | W | Positive-sequence active power |
| $q_+$ | `q_pos_{name}` |  | var | Positive-sequence reactive power |
| $p_-$ | `p_neg_{name}` |  | W | Negative-sequence active power |
| $p_0$ | `p_zer_{name}` |  | W | Zero-sequence active power |


## Source

- Module: `pydae.uds.vsgs.gflpfzv`
- File: [`packages/pydae-uds/src/pydae/uds/vsgs/gflpfzv.py`](https://github.com/pydae/pydae/tree/master/packages/pydae-uds/src/pydae/uds/vsgs/gflpfzv.py)
