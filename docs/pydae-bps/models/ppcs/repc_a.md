# REPC_A

*WECC Renewable Energy Plant Control model A —
`pydae.bps.ppcs.repc_a`*

---

## Purpose

REPC_A is the optional **plant-level supervisory controller** of the WECC
renewable model stack. It monitors the point of interconnection (POI) and
outputs:

- $Q_{ext}$ — reactive power / voltage reference to one or more REEC_B units
- $P_{ref}$ — active power reference to one or more REEC_B units

It implements plant-level closed-loop voltage or reactive power regulation
(Volt/VAR control) and optionally a frequency-droop governor response.

Unlike REEC_B — which is nested inside a ``weccs`` entry — REPC_A is
declared in a separate top-level ``ppcs`` section and lists the converter
names it governs. A single REPC_A can command multiple REGC_A/REEC_B units
simultaneously.

---

## Signal chain

```
Network
  │  Vreg (POI bus V), Qbranch, Pbranch, omega_coi
  ▼
┌───────────────────────────────────────────────────────────────┐
│  REPC_A                                                       │
│                                                               │
│  ── Volt / VAR path ──────────────────────────────────────── │
│                                                               │
│  Vreg_eff ──► [1/(1+s·Tfltr)] ──► Vreg_flt                  │
│  Qbranch  ──► [1/(1+s·Tfltr)] ──► Qbrn_flt                  │
│                                                               │
│  error = Vref − Vreg_flt   (RefFlag=1, V control)            │
│  error = Qref − Qbrn_flt  (RefFlag=0, Q control)             │
│                                                               │
│  error ──► deadband(dbd) ──► clip(emin,emax)                  │
│         ──► freeze(Vfrz) ──► K_q integral ──► (PI_Q output)  │
│         ──► [lead-lag (Tft,Tfv)] ──► clip(Qmin,Qmax) ──► Qext│
│                                                               │
│  ── Active power / frequency droop path ───────────────────── │
│                                                               │
│  Pbranch ──► [1/(1+s·Tp)] ──► Pbrn_flt                      │
│                                                               │
│  freq_dev ──► dual deadband(fdbd1,fdbd2)                      │
│           ──► Ddn (overfreq) + Dup (underfreq) = f_droop     │
│                                                               │
│  Pref_r − Pbrn_flt + f_droop ──► clip(femin,femax)           │
│   ──► K_ig integral ──► (PI_P output) ──► [1/(1+s·Tlag)]     │
│   ──► clip(Pmin,Pmax) ──► Pref   (Freq_flag=1)               │
│   ──► Pref_r (constant)           (Freq_flag=0)               │
└───────────────────────────────────────────────────────────────┘
  Qext, Pref → each REEC_B unit in the weccs list
```

---

## Mathematical model

### Volt / VAR path

**Regulated voltage** (with reactive droop or line drop compensation):

$$V_{reg,eff} = V_{reg} - K_c\, Q_{brn}$$

**Filters**:

$$\dot{V}_{reg,flt} = \frac{V_{reg,eff} - V_{reg,flt}}{T_{fltr}}, \qquad
  \dot{Q}_{brn,flt} = \frac{Q_{brn} - Q_{brn,flt}}{T_{fltr}}$$

**Error** (selected by $Ref_{flag}$):

$$e_{QV} = \begin{cases}
  V_{ref} - V_{reg,flt} & Ref_{flag} = 1 \text{ (voltage control)} \\
  Q_{ref} - Q_{brn,flt} & Ref_{flag} = 0 \text{ (Q control)}
\end{cases}$$

**Deadband** (signals in $[-dbd,\, dbd]$ are zeroed):

$$e_{db} = \max(e_{QV} - dbd,\,0) + \min(e_{QV} + dbd,\,0)$$
$$e_{clip} = \mathrm{clip}(e_{db},\; e_{min},\; e_{max})$$

**Volt/VAR PI integrator** (frozen when $V_{reg,flt} < V_{frz}$):

$$\dot{x}_{Kq} = K_q\, e_{clip} \cdot f_{frz}, \qquad
  f_{frz} = \mathrm{clip}(V_{reg,flt} - V_{frz},\; 0,\; 1)$$

$$Q_{PI,out} = \mathrm{clip}(K_p\, e_{clip} + x_{Kq},\; Q_{min},\; Q_{max})$$

**Lead-lag on Q output** (with time constants $T_{ft}$, $T_{fv}$):

$$\dot{x}_{Tfv} = \frac{Q_{PI,out} - x_{Tfv}}{T_{fv}}$$

$$Q_{ext,raw} = Q_{PI,out}\,\frac{T_{ft}}{T_{fv}}
               + x_{Tfv}\!\left(1 - \frac{T_{ft}}{T_{fv}}\right)$$

$$Q_{ext} = \mathrm{clip}(Q_{ext,raw},\; Q_{min},\; Q_{max})$$

### Active power / frequency droop path

**Active power filter**:

$$\dot{P}_{brn,flt} = \frac{P_{brn} - P_{brn,flt}}{T_p}$$

**Frequency droop** (dual deadband on $\Delta\omega = \omega_{coi}-1$):

$$f_{droop} =
  D_{dn}\,\max(\Delta\omega - f_{dbd1},\,0)
+ D_{up}\,\min(\Delta\omega - f_{dbd2},\,0)$$

**P error and PI**:

$$e_P = P_{ref,r} - P_{brn,flt} + f_{droop}$$
$$e_{P,clip} = \mathrm{clip}(e_P,\; f_{emin},\; f_{emax})$$

$$\dot{x}_{Kig} = K_{ig}\, e_{P,clip}$$
$$P_{PI,out} = \mathrm{clip}(K_{pg}\, e_{P,clip} + x_{Kig},\; P_{min},\; P_{max})$$

**Output lag and Pref** (selected by $Freq_{flag}$):

$$\dot{P}_{ref,lag} = \frac{P_{PI,out} - P_{ref,lag}}{T_{lag}}$$

$$P_{ref} = \begin{cases}
  \mathrm{clip}(P_{ref,lag},\; P_{min},\; P_{max}) & Freq_{flag} = 1 \\
  P_{ref,r} & Freq_{flag} = 0 \text{ (constant)}
\end{cases}$$

### Per-converter connection

REPC_A exposes a single $Q_{ext}$ and $P_{ref}$. These are broadcast to all
converters in the ``weccs`` list via algebraic equality constraints:

$$Q_{ext,gen_i} = Q_{ext}, \quad P_{ref,gen_i} = P_{ref}
  \qquad \forall\, i \in \mathrm{weccs}$$

Each $Q_{ext,gen_i}$ and $P_{ref,gen_i}$ is promoted from a u_run input
(set by REEC_B) to a y_run algebraic state, so the code generator resolves
them as array entries instead of free identifiers.

---

## Initialization

At $t=0$ (power flow solved):

| State | Initial value |
|---|---|
| $V_{reg,flt}$ | $V_{reg,0}$ from power flow |
| $Q_{brn,flt}$ | $Q_{brn,0}$ from power flow (≈ $\sum I_{q,i} V_i$) |
| $x_{Kq}$ | $Q_{ext,0}$ (no error at $t=0$ so PI is settled) |
| $x_{Tfv}$ | $Q_{ext,0}$ |
| $P_{brn,flt}$ | $P_{ref,r}$ from HJSON |
| $x_{Kig}$ | $P_{ref,r}$ |
| $P_{ref,lag}$ | $P_{ref,r}$ |
| $Q_{ext}$ (algebraic) | $Q_{ext,0}$ |
| $P_{ref}$ (algebraic) | $P_{ref,r}$ |

---

## HJSON configuration

```hjson
ppcs: [
  {
    type:    "repc_a",
    name:    "repc1",        // unique name for this plant controller
    reg_bus: "POI",          // regulated (POI) bus
    weccs:   ["gen1", "gen2"], // converter names commanded by this controller

    // Flags
    RefFlag:   1,            // 1 = voltage control at reg_bus
    VcompFlag: 0,            // 0 = reactive droop  1 = line drop comp.
    Freq_flag: 0,            // 0 = no governor     1 = frequency droop

    // Volt/VAR PI
    Tfltr: 0.02,  Tp: 0.02,
    Kp: 0.0,   Kq: 0.05,
    Kc: 0.0,                 // reactive droop gain (VcompFlag=0)
    dbd: 0.0,                // error deadband half-width
    emax: 0.3, emin: -0.3,
    Qmax: 0.4, Qmin: -0.4,
    Vfrz: 0.7,               // integrator freeze voltage (pu)
    Tft: 0.0,  Tfv: 0.15,   // lead-lag time constants

    // P / frequency droop
    Kpg: 0.0,  Kig: 0.05,
    Pmax: 1.0, Pmin: 0.0,
    Tlag: 0.15,
    fdbd1: 0.01, fdbd2: -0.01, // over/under-freq deadbands (pu)
    Ddn: 20.0,   Dup:  0.0,    // droop gains (pu power / pu freq)
    femax: 0.3, femin: -0.3,

    // Initial setpoints (auto-initialized from power flow)
    Vref:   1.0,  // target POI voltage (pu)
    Qref:   0.0,  // target branch Q (pu on S_base)
    Pref_r: 0.8,  // target branch P (pu on S_base)
    Qext:   1.0   // initial Q/V reference sent to REEC_B units
  }
]
```

---

## Parameters

| Symbol | Variable | Default | Units | Description |
|---|---|---|---|---|
| $T_{fltr}$ | `Tfltr` | 0.02 | s | V and Q filter time constant |
| $T_p$ | `Tp` | 0.02 | s | Active power filter time constant |
| $K_p$ | `Kp` | 0.0 | pu/pu | Volt/VAR PI proportional gain |
| $K_q$ | `Kq` | 0.05 | pu/pu·s | Volt/VAR PI integral gain |
| $K_c$ | `Kc` | 0.0 | pu/pu | Reactive droop gain (VcompFlag=0) |
| $dbd$ | `dbd` | 0.0 | pu | Error deadband half-width |
| $e_{max}$ | `emax` | 0.3 | pu | Volt/VAR error upper limit |
| $e_{min}$ | `emin` | −0.3 | pu | Volt/VAR error lower limit |
| $Q_{max}$ | `Qmax` | 0.4 | pu | Plant Q command upper limit |
| $Q_{min}$ | `Qmin` | −0.4 | pu | Plant Q command lower limit |
| $V_{frz}$ | `Vfrz` | 0.7 | pu | Voltage for Volt/VAR integrator freeze |
| $T_{ft}$ | `Tft` | 0.0 | s | Q output lead time constant |
| $T_{fv}$ | `Tfv` | 0.15 | s | Q output lag time constant |
| $K_{pg}$ | `Kpg` | 0.0 | pu/pu | P droop PI proportional gain |
| $K_{ig}$ | `Kig` | 0.05 | pu/pu·s | P droop PI integral gain |
| $P_{max}$ | `Pmax` | 1.0 | pu | Plant P command upper limit |
| $P_{min}$ | `Pmin` | 0.0 | pu | Plant P command lower limit |
| $T_{lag}$ | `Tlag` | 0.15 | s | P output lag time constant |
| $f_{dbd1}$ | `fdbd1` | 0.01 | pu | Over-frequency governor deadband |
| $f_{dbd2}$ | `fdbd2` | −0.01 | pu | Under-frequency governor deadband |
| $D_{dn}$ | `Ddn` | 20.0 | pu/pu | Down-regulation droop gain |
| $D_{up}$ | `Dup` | 0.0 | pu/pu | Up-regulation droop gain |
| $f_{emax}$ | `femax` | 0.3 | pu | P droop error upper limit |
| $f_{emin}$ | `femin` | −0.3 | pu | P droop error lower limit |

## Flags

| Flag | Values | Effect |
|---|---|---|
| `RefFlag` | 0 / 1 | 0 = regulate branch Q; 1 = regulate POI voltage |
| `VcompFlag` | 0 / 1 | 0 = reactive droop ($V_{eff} = V_{reg} - K_c Q_{brn}$); 1 = line drop compensation |
| `Freq_flag` | 0 / 1 | 0 = $P_{ref}$ constant at $P_{ref,r}$; 1 = frequency droop active |

## Dynamic states

| Symbol | Variable | Units | Description |
|---|---|---|---|
| $V_{reg,flt}$ | `Vreg_flt` | pu | Filtered regulated bus voltage |
| $Q_{brn,flt}$ | `Qbrn_flt` | pu | Filtered branch reactive power |
| $x_{Kq}$ | `x_Kq` | pu | Volt/VAR PI integrator state |
| $x_{Tfv}$ | `x_Tfv` | pu | Q output lead-lag lag state |
| $P_{brn,flt}$ | `Pbrn_flt` | pu | Filtered branch active power |
| $x_{Kig}$ | `x_Kig` | pu | P droop PI integrator state |
| $P_{ref,lag}$ | `Pref_lag` | pu | P command output lag state |

## Algebraic states (plant-level outputs)

| Symbol | Variable | Units | Description |
|---|---|---|---|
| $Q_{ext}$ | `Qext_{name}` | pu | Plant reactive/voltage command (shared by all converters) |
| $P_{ref}$ | `Pref_{name}` | pu | Plant active power command (shared by all converters) |

Per-converter connectors $Q_{ext,i}$ and $P_{ref,i}$ are added automatically
for each entry in ``weccs`` and are constrained equal to the plant-level outputs.

---

## Source

- Module: `pydae.bps.ppcs.repc_a`
- File: [`packages/pydae-bps/src/pydae/bps/ppcs/repc_a.py`](https://github.com/pydae/pydae/tree/main/packages/pydae-bps/src/pydae/bps/ppcs/repc_a.py)
- Reference: WECC Solar Plant Dynamic Modeling Guidelines, April 2014, Section 5.3
