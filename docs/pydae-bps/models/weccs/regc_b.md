# REGC_B

*WECC Renewable Energy Generator/Converter model B —
`pydae.bps.weccs.regc_b`*

---

## Purpose

REGC_B is a simplified variant of [REGC_A](regc_a.md). Both serve as the
grid-interface layer of the WECC renewable plant stack, injecting active and
reactive current into the network bus. REGC_B removes the Low Voltage Power
Logic (LVPL) and the low-voltage active current management gain, replacing
them with a single **combined apparent-current limit**.

| Feature | REGC_A | REGC_B |
|---|---|---|
| LVPL block | ✓ (`Lvplsw`, `Lvpl1`, `Zerox`, `Brkpt`) | ✗ removed |
| Low-voltage gain | ✓ (`lvpnt0`, `lvpnt1`) | ✗ removed |
| Combined $I_{max}$ limit | ✗ | ✓ `Imax` parameter |
| $r_{rpwr}$ direction | symmetric (approximation) | upward only |
| High-voltage clamp | ✓ | ✓ (unchanged) |
| Current lags $T_g$ | ✓ | ✓ (unchanged) |

Use REGC_B when a simpler current-source model suffices and the detailed
LVPL voltage-recovery behaviour of REGC_A is not required.

---

## Signal chain

```
REEC_B
  │  Ipcmd, Iqcmd
  ▼
┌──────────────────────────────────────────────────────────────┐
│  REGC_B                                                      │
│                                                              │
│  Vt ──► [1/(1+s·Tfltr)] ──► V_flt  (for HV clamp only)     │
│                                                              │
│  Iq ──► Ipmax = sqrt(Imax²−Iq²)                             │
│                   │                                          │
│  Ipcmd ──► clip(0, Ipmax) ──► [1/(1+s·Tg)] ──► Ip           │
│                                   ↑ upward rate: rrpwr       │
│                                                              │
│  Iqcmd ──► (−1) ──► rate_limit(Iqrmin,Iqrmax)               │
│                  ──► [1/(1+s·Tg)] ──► Iq                    │
│                                    │                         │
│               Iq_out = clip(Iq + Khv·max(Vt−Volim,0), Iolim)│
└──────────────────────────────────────────────────────────────┘
  P = Ip·Vt·S_n,   Q = Iq_out·Vt·S_n
```

---

## Mathematical model

### Voltage filter (for high-voltage clamp only)

$$\dot{V}_{flt} = \frac{V_t - V_{flt}}{T_{fltr}}$$

Unlike REGC_A, this filter is used solely for measuring the terminal voltage
for the high-voltage clamp; there is no LVPL or low-voltage gain path.

### Combined apparent-current limit

$$I_{pmax} = \sqrt{\max(I_{max}^2 - I_q^2,\; \varepsilon)}$$

The reactive current $I_q$ (state, not command) is used to avoid algebraic
circular dependence. At unity power factor ($I_q = 0$) the full $I_{max}$
is available for active current.

### Effective active current command

$$I_{p,eff} = \mathrm{clip}(I_{pcmd},\; 0,\; I_{pmax})$$

### Active current lag (upward rate limited)

$$\dot{I}_p = \mathrm{clip}\!\left(\frac{I_{p,eff} - I_p}{T_g},\;
                                    -999,\; r_{rpwr}\right)$$

The $-999$ lower bound leaves the downward rate effectively unconstrained
(the converter can reduce active current instantaneously). The upward
bound $r_{rpwr}$ limits how fast the converter recovers active current
after a voltage dip.

### Reactive current lag

$$\dot{I}_q = \mathrm{clip}\!\left(\frac{-I_{qcmd} - I_q}{T_g},\;
                                    I_{qr\min},\; I_{qr\max}\right)$$

### High-voltage reactive current clamp

$$I_{q,out} = \mathrm{clip}\!\left(
    I_q + K_{hv}\,\max(V_t - V_{olim},\,0),\;
    I_{olim},\; +\infty\right)$$

### Network injection

$$P = I_p\, V_t\, S_n, \qquad Q = I_{q,out}\, V_t\, S_n$$

---

## Initialization

| State | Initial value |
|---|---|
| $V_{flt}$ | $V_0$ (terminal voltage from power flow) |
| $I_p$ | $I_{pcmd,0}$ (lag settled at steady state) |
| $I_q$ | $-I_{qcmd,0}$ (sign inversion in lag equation) |

---

## HJSON configuration

```hjson
weccs: [
  {
    type:  "regc_b",
    bus:   "1",
    S_n:   100e6,

    Tg:    0.02,      // current regulator lag (s)
    Tfltr: 0.02,      // voltage filter for HV clamp (s)

    Imax:  1.1,       // combined apparent current limit (pu)
    rrpwr: 10.0,      // active current up-ramp rate (pu/s)

    Volim: 1.2,       // high-V clamp threshold (pu)
    Iolim: -1.3,      // high-V clamp lower bound (<0, pu)
    Khv:   0.7,       // high-V clamp gain

    Iqrmax:  999.0,
    Iqrmin: -999.0,

    Ipcmd: 0.8,
    Iqcmd: 0.0,
    V_ini: 1.0
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
| $T_{fltr}$ | `Tfltr` | 0.02 | s | Terminal voltage filter time constant |
| $I_{max}$ | `Imax` | 1.1 | pu | Combined apparent current limit |
| $r_{rpwr}$ | `rrpwr` | 10.0 | pu/s | Active current up-ramp rate limit |
| $V_{olim}$ | `Volim` | 1.2 | pu | High-voltage clamp threshold |
| $I_{olim}$ | `Iolim` | −1.3 | pu | High-voltage clamp lower bound (< 0) |
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
| $V_{flt}$ | `V_flt` | pu | Filtered terminal voltage |
| $I_p$ | `Ip` | pu | Actual active current injection |
| $I_q$ | `Iq` | pu | Actual reactive current (before HV clamp) |

## Outputs

| Symbol | Variable | Units | Description |
|---|---|---|---|
| $p_g$ | `p_g` | pu | Active power injection ($= I_p V_t$) |
| $q_g$ | `q_g` | pu | Reactive power injection ($= I_{q,out} V_t$) |
| $I_{pmax}$ | `Ipmax` | pu | Available active current ($= \sqrt{I_{max}^2 - I_q^2}$) |

---

## Source

- Module: `pydae.bps.weccs.regc_b`
- File: [`packages/pydae-bps/src/pydae/bps/weccs/regc_b.py`](https://github.com/pydae/pydae/tree/main/packages/pydae-bps/src/pydae/bps/weccs/regc_b.py)
- Reference: WECC Second-Generation Renewable Energy Model Library (post-2015)
