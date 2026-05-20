# REGFM_B1

*WECC Virtual Synchronous Machine Grid-Forming Inverter model B1 —
`pydae.bps.weccs.regfm_b1`*

---

## Purpose

REGFM_B1 is the **Virtual Synchronous Machine (VSM)** variant of the WECC
grid-forming inverter family. Unlike the droop-based [REGFM_A1](regfm_a1.md),
it contains an explicit swing equation and emulates a synchronous generator's
inertial response.

| Feature | REGFM_A1 | REGFM_B1 |
|---|---|---|
| Active power control | P-f droop (algebraic ω) | VSM swing equation + damping |
| Virtual inertia | Optional (`Mvir`) | Always active (`H`, `D1`, `D2`) |
| PLL | No | Yes — tracks bus angle |
| VSM angle | $\delta$ (single integrator) | $\delta_{VSM} = \delta_{IT} + \delta_{PLL}$ |
| States | 4 | 11–12 |
| Reference | PNNL-32278 | NREL/TP-5D00-90260 |

---

## Architecture — Four Control Subsystems

```
1. VSM Active Power / Frequency Control
   ωref − ω ──► [1/mp] ──► [1/(1+s·Tp)] ──► ΔP
                                               │
   Pref + ΔP − Pinv − D1·Δωm − D2·washout ──► [1/(2H·s)] ──► Δωm
                                                                 │
                         ω0·(Δωm + ΔωPLL) ──────────────────────► [1/s] ──► δIT

2. Voltage Control
   Q or Iq (droop) ──► Vcmd ──► [kpv + kiv/s] ──► EVSM ──(×f_cl)──► EVSM_lim

3. PLL
   V·sin(θ−δPLL) = Vq ──► [kpPLL + kiPLL/s] ──► ΔωPLL ──► [ω0/s] ──► δPLL

4. FCL (Fault Current Limiting)
   |I| > ImaxF  →  f_cl = ImaxF/|I| < 1  (scale EVSM down)

   δVSM = δIT + δPLL  ──► [voltage source E∠δVSM behind XL] ──► network
```

---

## Mathematical model

### Measurement filters (eqs. 1–5 of spec)

$$\dot P_{inv}   = (P_e   - P_{inv})  / T_{pf}, \quad
  \dot I_{d,inv} = (i_d   - I_{d,inv}) / T_{if}$$

$$\dot Q_{inv}   = (Q_e   - Q_{inv})  / T_{Qf}, \quad
  \dot V_{inv}   = (V     - V_{inv})  / T_{Vf}, \quad
  \dot I_{q,inv} = (i_q   - I_{q,inv}) / T_{if}$$

### P-f droop filter (when $F_{flag}=1$)

$$\dot x_{\Delta P} = \left(\frac{1-\omega_{in}}{m_p} - x_{\Delta P}\right) / T_p,
  \qquad \Delta P = x_{\Delta P}$$

where $\omega_{in} = \omega_{coi}$ ($\omega_{flag}=0$) or $\omega_{PLL}$ ($\omega_{flag}=1$),
and $T_p = 0$ bypasses the filter.

### VSM swing equation and D2 washout

$$\dot{x}_{D2}    = \omega_D\,(\Delta\omega_m - x_{D2})$$

$$\dot\Delta\omega_m = \frac{P_{cond} - P_{inv}
                              - D_1\Delta\omega_m
                              - D_2(\Delta\omega_m - x_{D2})}{2H}$$

where $P_{cond} = P_{ref} + \Delta P$. $\Delta\omega_m$ is clamped to
$[\Delta\omega_{min}, \Delta\omega_{max}]$.

### VSM angle integrator

$$\dot\delta_{IT} = \omega_0\,(\Delta\omega_m + \Delta\omega_{PLL})$$

Clamped to $[\delta_{IT,min}, \delta_{max}]$ where:

$$\delta_{max} = \arcsin(X_L \cdot I_{maxSS})$$
$$\delta_{IT,min} = \begin{cases} -\delta_{max} & ES_{flag}=1\text{ (battery)} \\ 0 & ES_{flag}=0 \end{cases}$$

### PLL

$$V_q^{PLL} = V\sin(\theta - \delta_{PLL})$$

$$\dot x_{PLL} = k_{i,PLL}\,V_q^{PLL} \cdot f_{frz}, \qquad
  \Delta\omega_{PLL} = \mathrm{clip}(k_{p,PLL}\,V_q^{PLL} + x_{PLL},\;
                                      \Delta\omega_{PLL,min},\; \Delta\omega_{PLL,max})$$

$$\dot\delta_{PLL} = \omega_0\,\Delta\omega_{PLL} \cdot f_{frz}$$

where $f_{frz}=0$ when $V < V_{PLL,frz}$ (freeze during deep voltage dip).

### VSM angle and Q-V droop

$$\delta_{VSM} = \delta_{IT} + \delta_{PLL}$$

$$V_{cmd} = \begin{cases}
  V_{ref} + m_q(Q_{ref} - Q_{inv}) & V_{drp,flag}=0\\
  V_{ref} - m_q I_{q,inv}          & V_{drp,flag}=1
\end{cases}$$

### Voltage PI controller with dynamic limits

$$\dot x_{iv} = k_{iv}(V_{cmd} - V_{inv})$$

$$E_{min} = \sqrt{(V_{inv} - I_{q,maxSS}\,X_L)^2 + (I_{d,inv}\,X_L)^2}$$
$$E_{max} = \sqrt{(V_{inv} + I_{q,maxSS}\,X_L)^2 + (I_{d,inv}\,X_L)^2}$$

$$E_{VSM} = \mathrm{clip}(k_{pv}(V_{cmd} - V_{inv}) + x_{iv},\; E_{min},\; E_{max})$$

### PQ priority algorithm — steady-state current limits

| $PQ_{flag}$ | Priority | $I_{d,maxSS}$ | $I_{q,maxSS}$ |
|---|---|---|---|
| 0 | Q | $\sqrt{I_{maxSS}^2 - I_{q,inv}^2}$ | $k_f I_{maxSS}$ |
| 1 | P | $k_f I_{maxSS}$ | $\sqrt{I_{maxSS}^2 - I_{d,inv}^2}$ |

### Fault current limiting

$$f_{cl} = \min(I_{maxF}/|I|,\;1), \qquad E_{VSM,lim} = E_{VSM} \cdot f_{cl}$$

### Network algebraic equations ($R_e = 0$ default)

$$0 = E_{VSM,lim} - X_L i_d - v_q, \quad 0 = X_L i_q - v_d$$

where $v_d = V\sin(\delta_{VSM}-\theta)$, $v_q = V\cos(\delta_{VSM}-\theta)$.

---

## Initialization

| State | Initial value |
|---|---|
| $\delta_{IT}$ | $\arctan(X_L P_{ref} / (E_0 V_0))$ |
| $\Delta\omega_m$ | 0 |
| $x_{D2}$ | 0 |
| $P_{inv}$ | $P_{ref}$ |
| $I_{d,inv}$ | $(E_0 - V_0)/X_L$ |
| $Q_{inv}$ | $Q_{ref}$ |
| $V_{inv}$ | $V_0$ |
| $I_{q,inv}$ | $V_0\sin\delta_0/X_L$ |
| $x_{iv}$ | $E_0$ |
| $\delta_{PLL}$ | 0 (PLL locked to bus at $\theta \approx 0$) |
| $x_{PLL}$ | 0 |
| $x_{\Delta P}$ | 0 |

---

## HJSON configuration

```hjson
weccs: [{
  type:    "regfm_b1",  bus: "1",  S_n: 100e6,  F_n: 50.0,

  XL: 0.1,  Re: 0.0,           // coupling impedance

  // VSM
  mp: 0.02,  Tp: 0.0,           // P-f droop gain and filter
  H: 0.5,                        // inertia constant (s)
  D1: 0.0,  D2: 100.0,  omegaD: 50.0,   // damping
  Dωmax: 0.05,  Dωmin: -0.05,

  // Voltage
  mq: 0.05,  kpv: 0.0,  kiv: 5.0,
  Vref: 1.0,  Qref: 0.0,  Pref: 0.8,

  // Limits
  ImaxSS: 1.0,  ImaxF: 1.5,  kf: 0.9,  Ke: 1.0,

  // Filters
  Tpf: 0.02,  TQf: 0.02,  TVf: 0.02,  Tif: 0.02,

  // PLL
  kpPLL: 0.265,  kiPLL: 2.65,
  DwPLLmax: 0.2,  DwPLLmin: -0.2,
  VPLLfrz: 0.05,

  // Flags
  omegaFlag: 0,  VdrpFlag: 0,  QVFlag: 1,
  PQFlag: 1,  FFlag: 1,  ESFlag: 1,

  V_ini: 1.0
}]
```

---

## Parameters

| Symbol | Variable | Default | Units | Description |
|---|---|---|---|---|
| $X_L$ | `XL` | 0.1 | pu | Coupling reactance (0.04–0.40) |
| $R_e$ | `Re` | 0.0 | pu | Coupling resistance (0 ≤ Re ≤ XL/4) |
| $m_p$ | `mp` | 0.02 | pu/pu | P-f droop gain (=1/mr) |
| $T_p$ | `Tp` | 0.0 | s | Droop filter time constant (0=bypass) |
| $H$ | `H` | 0.5 | s | Virtual inertia constant |
| $D_1$ | `D1` | 0.0 | pu | Algebraic damping coefficient |
| $D_2$ | `D2` | 100.0 | pu | Transient (washout) damping coefficient |
| $\omega_D$ | `omegaD` | 50.0 | rad/s | Washout corner frequency |
| $\Delta\omega_{max}$, $\Delta\omega_{min}$ | `Dωmax`, `Dωmin` | ±0.05 | pu | VSM speed limits |
| $m_q$ | `mq` | 0.05 | pu | Q-V droop gain (or virtual impedance) |
| $k_{pv}$ | `kpv` | 0.0 | pu | Voltage PI proportional gain |
| $k_{iv}$ | `kiv` | 5.0 | pu/s | Voltage PI integral gain |
| $I_{maxSS}$ | `ImaxSS` | 1.0 | pu | Steady-state current limit (≤0→1/XL) |
| $I_{maxF}$ | `ImaxF` | 1.5 | pu | Transient current limit |
| $k_f$ | `kf` | 0.9 | — | Priority factor (0→reset to 1) |
| $K_e$ | `Ke` | 1.0 | — | Negative current scalar |
| $k_{p,PLL}$ | `kpPLL` | 0.265 | pu | PLL proportional gain |
| $k_{i,PLL}$ | `kiPLL` | 2.65 | pu/s | PLL integral gain |
| $V_{PLL,frz}$ | `VPLLfrz` | 0.05 | pu | PLL freeze threshold (≤0=disabled) |

## Flags

| Flag | 0 | 1 |
|---|---|---|
| `omegaFlag` | Use $\omega_{coi}$ for droop | Use $\omega_{PLL}$ |
| `VdrpFlag` | Q droop ($m_q$ in pu/pu) | $I_q$ droop ($m_q$ = virtual impedance) |
| `QVFlag` | Plant ctrl changes $Q_{ref}$ | Plant ctrl changes $V_{ref}$ |
| `PQFlag` | Q priority | P priority |
| `FFlag` | Droop disabled | Droop enabled |
| `ESFlag` | Non-battery ($\delta_{IT,min}=0$) | Battery ($\delta_{IT,min}=-\delta_{max}$) |

## Dynamic states

| Symbol | Variable | Units | Description |
|---|---|---|---|
| $\delta_{IT}$ | `deltaIT` | rad | VSM angle integrator |
| $\Delta\omega_m$ | `Domegam` | pu | VSM speed deviation |
| $x_{D2}$ | `x_D2` | pu | D2 washout filter state |
| $P_{inv}$ | `Pinv` | pu | Filtered active power |
| $I_{d,inv}$ | `Idinv` | pu | Filtered d-axis current |
| $Q_{inv}$ | `Qinv` | pu | Filtered reactive power |
| $V_{inv}$ | `Vinv` | pu | Filtered terminal voltage |
| $I_{q,inv}$ | `Iqinv` | pu | Filtered q-axis current |
| $x_{iv}$ | `xiv` | pu | Voltage PI integrator |
| $\delta_{PLL}$ | `deltaPLL` | rad | PLL angle |
| $x_{PLL}$ | `xPLL` | pu | PLL PI integrator |
| $x_{\Delta P}$ | `x_DP` | pu | P-f droop filter (FFlag=1 only) |

## Outputs

| Variable | Units | Description |
|---|---|---|
| `p_g`, `q_g` | pu | Active/reactive power injection |
| `EVSM` | pu | Internal voltage magnitude |
| `deltaVSM` | rad | Total VSM angle ($\delta_{IT}+\delta_{PLL}$) |
| `deltaIT`, `deltaPLL` | rad | Individual angle components |
| `Domegam` | pu | VSM speed deviation (inertia response) |
| `DomegaPLL` | pu | PLL frequency deviation |
| `Imag` | pu | Output current magnitude |
| `f_cl` | — | Fault current limiter factor |
| `Vcmd` | pu | Voltage command |

---

## Source

- Module: `pydae.bps.weccs.regfm_b1`
- File: [`packages/pydae-bps/src/pydae/bps/weccs/regfm_b1.py`](https://github.com/pydae/pydae/tree/main/packages/pydae-bps/src/pydae/bps/weccs/regfm_b1.py)
- Reference: NREL/TP-5D00-90260, *Virtual Synchronous Machine Grid-Forming
  Inverter Model Specification (REGFM_B1)*, June 2024, UNIFI-2024-6-1
