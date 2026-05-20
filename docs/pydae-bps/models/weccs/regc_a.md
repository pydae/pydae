# REGC_A

*WECC Renewable Energy Generator/Converter model A —
`pydae.bps.weccs.regc_a`*

---

## Purpose

REGC_A is the **grid-interface layer** of the WECC three-module renewable
energy plant model. It receives real ($I_{pcmd}$) and reactive ($I_{qcmd}$)
current commands — either as fixed external inputs or driven by REEC_B — and
injects the corresponding active and reactive power into the network bus.

The same module is used identically for PV plants, Type 3, and Type 4 generic
wind turbines.

---

## Signal chain

```
REEC_B
  │  Ipcmd, Iqcmd
  ▼
┌─────────────────────────────────────────────────────────┐
│  REGC_A                                                 │
│                                                         │
│  Vt ──► [1/(1+s·Tfltr)] ──► V_flt                     │
│                                   │                     │
│              LVPL piecewise ◄─────┤   Low-V gain ◄─────┤
│                   │                        │            │
│  Ipcmd ──► min(·,LVPL) ──► ×gain ──► [1/(1+s·Tg)] ──► Ip
│                                                         │
│  Iqcmd ──► (−1) ──► rate_limit ──► [1/(1+s·Tg)] ──► Iq
│                                            │            │
│                     HV clamp: Khv·max(Vt−Volim,0) ─────┘
│                                            ▼            │
│                                          Iq_out         │
└─────────────────────────────────────────────────────────┘
  Ip, Iq_out → network:  P = Ip·Vt·S_n,  Q = Iq_out·Vt·S_n
```

---

## Mathematical model

### Voltage filter (for LVPL and low-voltage management)

$$\dot{V}_{flt} = \frac{V_t - V_{flt}}{T_{fltr}}$$

### Low Voltage Active Current Management gain

$$\mathrm{gain} =
  \mathrm{clip}\!\left(\frac{V_{flt} - l_{vpnt0}}{l_{vpnt1} - l_{vpnt0}},\;
                        0,\; 1\right)$$

The gain is 0 below $l_{vpnt0}$ (active current injection cut off) and 1
above $l_{vpnt1}$ (full injection). This approximates PLL behaviour during
voltage dips.

### Low Voltage Power Logic (LVPL, when $L_{vplsw}=1$)

$$\mathrm{LVPL} =
  \mathrm{clip}\!\left(\frac{V_{flt} - Z_{erox}}{B_{rkpt} - Z_{erox}}
                        \,L_{vpl1},\; 0,\; L_{vpl1}\right)$$

The effective active current command entering the lag:

$$I_{p,eff} = \min(I_{pcmd},\;\mathrm{LVPL})\cdot\mathrm{gain}$$

When $L_{vplsw}=0$ the LVPL is bypassed: $I_{p,eff} = I_{pcmd}\cdot\mathrm{gain}$.

### Active current lag

$$\dot{I}_p =
  \mathrm{clip}\!\left(\frac{I_{p,eff} - I_p}{T_g},\;
                        -r_{rpwr},\; r_{rpwr}\right)$$

The up/down rate limit $r_{rpwr}$ is applied symmetrically here (a common
simulation simplification; the standard specifies only an upward limit on
voltage recovery).

### Reactive current lag

$$\dot{I}_q =
  \mathrm{clip}\!\left(\frac{-I_{qcmd} - I_q}{T_g},\;
                        I_{qr\min},\; I_{qr\max}\right)$$

Note the sign inversion on $I_{qcmd}$: positive reactive injection
($I_q > 0$) requires $I_{qcmd} < 0$.

### High-voltage reactive current clamp

$$I_{q,out} =
  \mathrm{clip}\!\left(I_q + K_{hv}\,\max(V_t - V_{olim},\,0),\;
                        I_{olim},\; +\infty\right)$$

When $V_t > V_{olim}$ the clamp injects a positive correction that reduces
reactive injection (absorbs reactive power), bounded below by $I_{olim} < 0$.

### Network injection

$$P = I_p \, V_t \, S_n, \qquad Q = I_{q,out} \, V_t \, S_n$$

---

## Initialization

At $t = 0$ (power flow solved, $V_t = V_0$, $I_{pcmd,0}$, $I_{qcmd,0}$ given):

| State | Initial value |
|---|---|
| $V_{flt}$ | $V_0$ |
| $I_p$ | $I_{pcmd,0}$ (gain = 1, LVPL inactive at nominal voltage) |
| $I_q$ | $-I_{qcmd,0}$ (steady-state of the lag with sign inversion) |

---

## HJSON configuration

```hjson
weccs: [
  {
    type:    "regc_a",
    bus:     "1",
    S_n:     100e6,        // converter MVA base

    Tg:      0.02,         // current regulator lag (s)
    Tfltr:   0.02,         // LVPL voltage filter (s)

    Lvplsw:  1,            // 1 = LVPL enabled
    Lvpl1:   1.22,         // current at Brkpt (pu)
    Zerox:   0.4,          // LVPL zero-crossing voltage (pu)
    Brkpt:   0.9,          // LVPL breakpoint voltage (pu)
    rrpwr:   10.0,         // active current ramp rate (pu/s)

    lvpnt0:  0.4,          // low-V gain lower breakpoint (pu)
    lvpnt1:  0.8,          // low-V gain upper breakpoint (pu)

    Volim:   1.2,          // high-V clamp threshold (pu)
    Iolim:   -1.3,         // high-V clamp lower limit (<0, pu)
    Khv:     0.7,          // high-V clamp gain (pu/pu)

    Iqrmax:  999.0,        // max reactive current rate (pu/s)
    Iqrmin: -999.0,        // min reactive current rate (pu/s)

    Ipcmd:   0.8,          // initial active current command (pu)
    Iqcmd:   0.0,          // initial reactive current command (pu)
    V_ini:   1.0           // initial terminal voltage guess (pu)
  }
]
```

Add a ``reec`` sub-dict to attach REEC_B local controls (see [REEC_B](reec_b.md)).

---

## Parameters

| Symbol | Variable | Default | Units | Description |
|---|---|---|---|---|
| $S_n$ | `S_n` | 100e6 | VA | Converter MVA base |
| $T_g$ | `Tg` | 0.02 | s | Current regulator lag time constant |
| $T_{fltr}$ | `Tfltr` | 0.02 | s | Terminal voltage filter time constant (LVPL) |
| $L_{vplsw}$ | `Lvplsw` | 1 | — | Enable (1) / disable (0) LVPL |
| $L_{vpl1}$ | `Lvpl1` | 1.22 | pu | LVPL current at and above $B_{rkpt}$ |
| $Z_{erox}$ | `Zerox` | 0.4 | pu | LVPL zero-crossing voltage |
| $B_{rkpt}$ | `Brkpt` | 0.9 | pu | LVPL breakpoint voltage |
| $r_{rpwr}$ | `rrpwr` | 10.0 | pu/s | Active current ramp rate limit |
| $l_{vpnt0}$ | `lvpnt0` | 0.4 | pu | Low-V gain lower breakpoint (gain = 0 below) |
| $l_{vpnt1}$ | `lvpnt1` | 0.8 | pu | Low-V gain upper breakpoint (gain = 1 above) |
| $V_{olim}$ | `Volim` | 1.2 | pu | High-voltage clamp threshold |
| $I_{olim}$ | `Iolim` | −1.3 | pu | High-voltage clamp lower limit (< 0) |
| $K_{hv}$ | `Khv` | 0.7 | pu/pu | High-voltage clamp gain |
| $I_{qr\max}$ | `Iqrmax` | 999.0 | pu/s | Maximum reactive current rate of change |
| $I_{qr\min}$ | `Iqrmin` | −999.0 | pu/s | Minimum reactive current rate of change |

## Inputs

| Symbol | Variable | Default | Units | Description |
|---|---|---|---|---|
| $I_{pcmd}$ | `Ipcmd` | 0.8 | pu | Active current command (from REEC_B or external) |
| $I_{qcmd}$ | `Iqcmd` | 0.0 | pu | Reactive current command (from REEC_B or external) |

## Dynamic states

| Symbol | Variable | Units | Description |
|---|---|---|---|
| $V_{flt}$ | `V_flt` | pu | Filtered terminal voltage (LVPL filter) |
| $I_p$ | `Ip` | pu | Actual active current injection |
| $I_q$ | `Iq` | pu | Actual reactive current (before HV clamp) |

## Outputs

| Symbol | Variable | Units | Description |
|---|---|---|---|
| $p_g$ | `p_g` | pu | Active power injection ($= I_p V_t$) |
| $q_g$ | `q_g` | pu | Reactive power injection ($= I_{q,out} V_t$) |
| $\mathrm{gain}$ | `gain` | — | Low-voltage active current management gain |
| $V_{flt}$ | `V_flt` | pu | Filtered terminal voltage |

---

## Source

- Module: `pydae.bps.weccs.regc_a`
- File: [`packages/pydae-bps/src/pydae/bps/weccs/regc_a.py`](https://github.com/pydae/pydae/tree/main/packages/pydae-bps/src/pydae/bps/weccs/regc_a.py)
- Reference: WECC Solar Plant Dynamic Modeling Guidelines, April 2014, Section 5.1
