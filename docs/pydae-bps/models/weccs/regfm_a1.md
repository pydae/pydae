# REGFM_A1

*WECC Renewable Energy Grid-Forming Inverter model A1 —
`pydae.bps.weccs.regfm_a1`*

---

## Purpose

REGFM_A1 represents a **droop-controlled grid-forming (GFM) inverter** — a
fundamentally different operating principle from the grid-following converters
REGC_A and REGC_B.

| Feature | Grid-Following (REGC_A/B) | Grid-Forming (REGFM_A1) |
|---|---|---|
| Network interface | Current source | Voltage source behind $X_L$ |
| Synchronisation | PLL tracks grid voltage | Sets own frequency via droop |
| Islanded operation | No | Yes |
| Inertia contribution | None | Optional virtual inertia ($M_{vir}$) |
| Control inputs | $I_{pcmd}$, $I_{qcmd}$ | $P_{ref}$, $Q_{ref}$, $V_{ref}$ |

Because the network interface is a voltage source behind a coupling reactance,
the algebraic equations are identical to those of a synchronous machine —
REGFM_A1 can be viewed as a "synchronous generator" whose internal dynamics
are replaced by droop controllers.

---

## Signal chain

```
REPC_A  (plant level, optional — provides Pref)
    │
REGFM_A1  (standalone weccs entry, no reec layer)
    │  Virtual voltage source E∠δ behind XL
    │
Network bus
```

REGFM_A1 does **not** use the ``reec`` sub-dict pattern because it does not
separate grid-interface and electrical-control layers — both are integrated
into a single model, like a synchronous machine.

---

## Block diagram

```
Pref ──►─────────────────────────────────────────────────────┐
         │                                                    │
Pe ──► [1/(1+s·TPI)] ──► x_Pe ──► (−mP) ──►(+)─► clip ──► omega_droop
                                             ▲                │
                                           1.0 ─────────────►│
                                                              ▼
                                                       [Omega_b·(·−omega_coi)]
                                                              │
                                                       d(delta)/dt
                                                              │
                                                          delta ──► network
                                                                      interface

Qref ──►────────────────────────────────────────────────────┐
Vref ──►(+kpv)──►                                           │
         │                                                   │
Qe ──► [1/(1+s·TQI)] ──► x_Qe ──► (−nQ) ──►(+)─► clip ──► E_droop
                                             ▲                │
                                           Vref ─────────────►│
                                                              ▼
                                                    E_droop·f_cl ──► [1/(1+s·Tv)] ──► E
                                                         ▲
                               f_cl = min(ImaxF/|I|, 1) ─┘  (fault current limit)

E, delta ──► [voltage source behind XL] ──► i_d, i_q, p_g, q_g ──► network
```

---

## Mathematical model

### dq-frame terminal voltage

$$v_d = V\sin(\delta-\theta), \qquad v_q = V\cos(\delta-\theta)$$

### P-f droop: virtual frequency and angle

$$\omega_{droop} = \mathrm{clip}\!\left(
    1 - m_P(x_{Pe} - P_{ref}),\;
    \omega_{min},\; \omega_{max}\right)$$

$$\dot{\delta} = \Omega_b(\omega_{droop} - \omega_{coi})
              - K_\delta(\delta - \delta_{ref})$$

The optional $K_\delta$ term prevents unbounded angle drift in isolated
systems. In grid-connected operation it can be set to zero.

### Active power filter

$$\dot{x}_{Pe} = (P_e - x_{Pe}) / T_{PI}, \qquad P_e = i_d v_d + i_q v_q$$

### Q-V droop: internal voltage reference

$$E_{droop} = \mathrm{clip}\!\left(
    V_{ref} + k_{pv}(V_{ref} - V) - n_Q(x_{Qe} - Q_{ref}),\;
    E_{min},\; E_{max}\right)$$

When $k_{pv} = 0$ (default): pure reactive power droop. When $k_{pv} > 0$:
additional proportional feedback from the terminal voltage deviation.

### Reactive power filter

$$\dot{x}_{Qe} = (Q_e - x_{Qe}) / T_{QI}, \qquad Q_e = i_d v_q - i_q v_d$$

### Fault Current Limiting (FCL)

$$|I| = \sqrt{i_d^2 + i_q^2 + \varepsilon}, \qquad
  f_{cl} = \min\!\left(\frac{I_{maxF}}{|I|},\; 1\right)$$

$$E_{ref} = E_{droop} \cdot f_{cl}$$

When the output current exceeds $I_{maxF}$, the internal voltage reference is
scaled down proportionally, naturally limiting the fault current.

### Voltage control lag

$$\dot{E} = (E_{ref} - E) / T_v$$

### Coupling reactance algebraic equations ($R_a = 0$)

$$0 = E - X_L\,i_d - v_q \qquad \text{(q-axis)}$$
$$0 = X_L\,i_q - v_d        \qquad \text{(d-axis, }E_d = 0\text{)}$$
$$0 = i_d v_d + i_q v_q - p_g$$
$$0 = i_d v_q - i_q v_d - q_g$$

Solving explicitly (for reference):

$$i_d = \frac{E - v_q}{X_L}, \qquad i_q = \frac{v_d}{X_L}$$

$$p_g = \frac{E\,V\sin(\delta-\theta)}{X_L}, \qquad
  q_g = \frac{E\,V\cos(\delta-\theta) - V^2}{X_L}$$

### Network injection

$$P = p_g\,S_n, \qquad Q = q_g\,S_n$$

---

## Initialization

At $t=0$ (power flow solved, $V = V_0$, $\theta = 0$):

| State | Initial value |
|---|---|
| $\delta$ | $\arctan(X_L P_{ref} / (E_0 V_0))$ |
| $x_{Pe}$ | $P_{ref}$ (filter at steady state = droop setpoint) |
| $x_{Qe}$ | $Q_{ref}$ |
| $E$ | $V_0 + X_L Q_{ref} / V_0$ (from Q-V droop, small-angle approx.) |
| $i_d$ | $(E_0 - V_0\cos\delta_0)/X_L$ |
| $i_q$ | $V_0\sin\delta_0/X_L$ |
| $p_g$ | $P_{ref}$ |
| $q_g$ | $Q_{ref}$ |

At steady state, $\omega_{droop} = 1$ (frequency deviation is zero), which
requires $x_{Pe} = P_{ref}$ — satisfied by the initial condition.

---

## HJSON configuration

```hjson
weccs: [
  {
    type:  "regfm_a1",
    bus:   "1",
    S_n:   100e6,
    F_n:   50.0,

    XL:    0.1,       // coupling reactance (pu, typical 0.05–0.25)

    // P-f droop
    mP:    0.04,      // 4% frequency deviation per unit power deviation
    TPI:   0.05,      // active power filter time constant (s)

    // Q-V droop
    nQ:    0.05,      // reactive power droop gain
    kpv:   0.0,       // voltage feedback gain (0 = disabled, pure Q droop)
    TQI:   0.05,      // reactive power filter time constant (s)

    // Voltage control
    Tv:    0.02,      // voltage control lag (s)
    Vref:  1.0,       // voltage reference (pu)

    // Setpoints
    Pref:  0.8,       // active power setpoint (pu on S_n)
    Qref:  0.0,       // reactive power reference (pu on S_n)

    Pmax:  1.0,   Pmin: 0.0,
    Qmax:  0.4,   Qmin: -0.4,
    Emax:  1.2,   Emin: 0.8,

    ImaxF: 1.5,       // fault current limit (pu, typical 1.5–3.0)

    omega_max: 1.05,
    omega_min: 0.95,

    Mvir:     0.0,    // virtual inertia (s); 0 = pure droop
    K_delta:  0.0001, // angle stabiliser (0 in fully grid-connected operation)
    V_ini:    1.0
  }
]
```

---

## Parameters

| Symbol | Variable | Default | Units | Description |
|---|---|---|---|---|
| $S_n$ | `S_n` | 100e6 | VA | Converter MVA base |
| $F_n$ | `F_n` | 50.0 | Hz | Nominal frequency |
| $X_L$ | `XL` | 0.1 | pu | Coupling reactance |
| $m_P$ | `mP` | 0.04 | pu/pu | P-f droop gain |
| $n_Q$ | `nQ` | 0.05 | pu/pu | Q-V droop gain |
| $k_{pv}$ | `kpv` | 0.0 | pu/pu | Terminal voltage proportional feedback |
| $T_{PI}$ | `TPI` | 0.05 | s | Active power filter time constant |
| $T_{QI}$ | `TQI` | 0.05 | s | Reactive power filter time constant |
| $T_v$ | `Tv` | 0.02 | s | Voltage control time constant |
| $V_{ref}$ | `Vref` | 1.0 | pu | Voltage reference |
| $P_{ref}$ | `Pref` | 0.8 | pu | Active power setpoint |
| $Q_{ref}$ | `Qref` | 0.0 | pu | Reactive power reference |
| $P_{max}$, $P_{min}$ | `Pmax`, `Pmin` | 1.0, 0.0 | pu | Active power limits |
| $Q_{max}$, $Q_{min}$ | `Qmax`, `Qmin` | ±0.4 | pu | Reactive power limits |
| $E_{max}$, $E_{min}$ | `Emax`, `Emin` | 1.2, 0.8 | pu | Internal voltage limits |
| $I_{maxF}$ | `ImaxF` | 1.5 | pu | Fault current limit |
| $\omega_{max}$, $\omega_{min}$ | `omega_max`, `omega_min` | 1.05, 0.95 | pu | Frequency limits |
| $M_{vir}$ | `Mvir` | 0.0 | s | Virtual inertia (0 = pure droop) |
| $K_\delta$ | `K_delta` | 0.0 | pu | Angle stabiliser |

## Dynamic states

| Symbol | Variable | Units | Description |
|---|---|---|---|
| $\delta$ | `delta` | rad | Virtual rotor angle |
| $x_{Pe}$ | `x_Pe` | pu | Filtered active power ($T_{PI}$ lag) |
| $x_{Qe}$ | `x_Qe` | pu | Filtered reactive power ($T_{QI}$ lag) |
| $E$ | `E` | pu | Internal voltage magnitude ($T_v$ lag) |

## Algebraic states

| Symbol | Variable | Units | Description |
|---|---|---|---|
| $i_d$ | `i_d` | pu | d-axis current |
| $i_q$ | `i_q` | pu | q-axis current |
| $p_g$ | `p_g` | pu | Active power output (pu on $S_n$) |
| $q_g$ | `q_g` | pu | Reactive power output (pu on $S_n$) |

## Outputs

| Variable | Units | Description |
|---|---|---|
| `p_g` | pu | Active power injection |
| `q_g` | pu | Reactive power injection |
| `E` | pu | Internal voltage magnitude |
| `delta` | rad | Virtual angle |
| `omega_d` | pu | Droop frequency ($= 1 - m_P(x_{Pe}-P_{ref})$) |
| `E_droop` | pu | Voltage droop reference (before FCL) |
| `Imag` | pu | Output current magnitude ($\sqrt{i_d^2+i_q^2}$) |
| `f_cl` | — | Fault current limiter factor ($\leq 1$) |

---

## Source

- Module: `pydae.bps.weccs.regfm_a1`
- File: [`packages/pydae-bps/src/pydae/bps/weccs/regfm_a1.py`](https://github.com/pydae/pydae/tree/main/packages/pydae-bps/src/pydae/bps/weccs/regfm_a1.py)
- Reference: PNNL-32278, *Model Specification of Droop-Controlled Grid-Forming
  Inverters (REGFM_A1)*, Pacific Northwest National Laboratory, September 2023
