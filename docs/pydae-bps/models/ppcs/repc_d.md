# REPC_D

*WECC Renewable Energy Plant Controller model D —
`pydae.bps.ppcs.repc_d`*

---

## Purpose

REPC_D is the **recommended replacement for REPC_B** (per the WECC document
"Convert REPC_B to REPC_D", June 2024). It extends [REPC_A](repc_a.md) with:

| Feature | REPC_A | REPC_D |
|---|---|---|
| Bus attachment | Generator-attached | **Own bus** (standalone in `ppcs`) |
| P/Q references | Deviations from initial | **Absolute values** |
| Dispatch | Shared Qext/Pref to all | **Per-device** with weights `Kw_n`/`Kz_n` and limits |
| Per-device states | None | Q filter (`Tw_n`) + P filter (`Tz_n`) per device |
| Max devices | Unlimited (shared) | Up to 50 with individual parameters |
| Q deadband | Symmetric `±dbd` | **Asymmetric** `dbd1`/`dbd2` |
| Voltage ref limits | None | `Vrefmax`/`Vrefmin` |
| Frequency filter | None | `Tfrq` |
| Reactive-current comp. | None | `Kc`/`Tc` |
| Active P feedback flag | Always on | `Pefd_Flag` |
| P feedforward flag | None | `Ffwrd_Flag` |
| Q controller enable | Always on | `QVFlag` |
| PI limits | `Qmin/Qmax` only | + `pimax/pimin`, `qvrmax/qvrmin`, `prmax/prmin` |

---

## Architecture

```
Network (POI bus)
  │  V_reg, Q_brn, P_brn, Freq
  ▼
┌───────────────────────────────────────────────────────────────┐
│  REPC_D                                                       │
│                                                               │
│  ── Q/V channel ──────────────────────────────────────────── │
│  V_reg_eff ──► [1/(1+s·Tfltr)] ──► Vreg_flt                 │
│  Q_brn     ──► [1/(1+s·Tfltr)] ──► Qbrn_flt                 │
│  error = Vref − Vreg_flt   (RefFlg=1)                        │
│        = Qref − Qbrn_flt  (RefFlg=0)                        │
│  ──► asym. deadband(dbd1,dbd2) ──► PI(Kp,Ki) ──► lead-lag   │
│  ──► Qext_plant (plant-level Q/V command)                    │
│                                                               │
│  ── P / frequency channel ─────────────────────────────────  │
│  Freq ──► [1/(1+s·Tfrq)] ──► droop(fdbd1,fdbd2,Ddn,Dup)     │
│  P_brn ──► [1/(1+s·Tp)] ──► P_feedback  (Pefd_Flag=1)       │
│  ──► PI(Kpg,Kig) ──► feedfwd ──► [1/(1+s·Tlag)]             │
│  ──► Pref_plant (plant-level P command)                       │
│                                                               │
│  ── Per-device dispatch (for each n in weccs) ──────────────  │
│  Qext_plant ──► Qo_n + Kw_n·(Qext_plant − Qext_o)           │
│             ──► [1/(1+s·Tw_n)] ──► clip(Qmin_n,Qmax_n)       │
│             ──► Qext_{gen_n}  (to REEC_B or REGFM_A1)        │
│                                                               │
│  Pref_plant ──► Po_n + Kz_n·(Pref_plant − Pref_o)           │
│             ──► [1/(1+s·Tz_n)] ──► clip(Pmin_n,Pmax_n)       │
│             ──► Pref_{gen_n}  (to REEC_B)                    │
└───────────────────────────────────────────────────────────────┘
```

---

## Mathematical model

### Q/V channel

$$\dot V_{reg,flt} = (V_{reg,eff} - V_{reg,flt}) / T_{fltr}, \qquad
  \dot Q_{brn,flt} = (Q_{brn}     - Q_{brn,flt}) / T_{fltr}$$

$$V_{ref,c} = \mathrm{clip}(V_{ref},\; V_{ref,min},\; V_{ref,max})$$

$$e_{QV} = \begin{cases}
  V_{ref,c} - V_{reg,flt} & Ref_{flag}=1\\
  Q_{ref}   - Q_{brn,flt} & Ref_{flag}=0
\end{cases}$$

**Asymmetric deadband** (REPC_D replaces symmetric `±dbd` of REPC_A):

$$e_{db} = \max(e_{QV}-dbd_2,0) + \min(e_{QV}-dbd_1,0), \qquad
  e_{clip} = \mathrm{clip}(e_{db},\;e_{min},\;e_{max})$$

$$\dot x_{Kq} = K_i\,e_{clip}\cdot f_{frz}, \qquad
  Q_{PI,out} = \mathrm{clip}(K_p\,e_{clip}+x_{Kq},\;Q_{min},\;Q_{max})$$

$$\dot x_{Tfv} = (Q_{PI,out} - x_{Tfv})/T_{fv}$$

$$Q_{ext,plant} = \mathrm{clip}\!\left(
    Q_{PI,out}\,\frac{T_{ft}}{T_{fv}}
    + x_{Tfv}\!\left(1-\frac{T_{ft}}{T_{fv}}\right),\;Q_{min},\;Q_{max}\right)
    \quad (QV_{flag}=1)$$

### P / frequency channel

$$\dot P_{brn,flt} = (P_{brn} - P_{brn,flt})/T_p, \qquad
  \dot F_{flt} = (\Delta\omega - F_{flt})/T_{frq}$$

$$f_{droop} = D_{dn}\,\max(F_{flt}-f_{dbd2},0)
            + D_{up}\,\min(F_{flt}-f_{dbd1},0)$$

$$e_P = P_{ref,r} - P_{fb} + f_{droop}, \qquad
  P_{fb} = \begin{cases} P_{brn,flt} & Pe_{fd}=1 \\ 0 & Pe_{fd}=0 \end{cases}$$

$$\dot x_{Kig} = K_{ig}\,\mathrm{clip}(e_P,\;f_{emin},\;f_{emax})$$

$$P_{PI} = \mathrm{clip}(K_{pg}\,e_P + x_{Kig},\;pi_{min},\;pi_{max})$$

$$P_{ff} = P_{PI} + F_{fwd}\,P_{brn,flt}, \qquad
  \dot P_{ref,lag} = (\mathrm{clip}(P_{ff},\;P_{min},\;P_{max}) - P_{ref,lag})/T_{lag}$$

$$P_{ref,plant} = \mathrm{clip}(P_{ref,lag},\;P_{min},\;P_{max}) \quad (F_{flag}=1)$$

### Per-device dispatch (for each device $n$)

$$Q_{device,n} = Q_{o,n} + K_{w,n}\,(Q_{ext,plant} - Q_{ext,o})$$

$$\dot x_{Q,n} = (Q_{device,n} - x_{Q,n})/T_{w,n}$$

$$Q_{ext,n} = \mathrm{clip}(x_{Q,n},\;Q_{min,n},\;Q_{max,n})$$

$$P_{device,n} = P_{o,n} + K_{z,n}\,(P_{ref,plant} - P_{ref,o})$$

$$\dot x_{P,n} = (P_{device,n} - x_{P,n})/T_{z,n}$$

$$P_{ref,n} = \mathrm{clip}(x_{P,n},\;P_{min,n},\;P_{max,n})$$

where $Q_{ext,o}$ and $P_{ref,o}$ are the initial plant Q and P references.

At steady state ($Q_{ext,plant} = Q_{ext,o}$) each device gets its initial
setpoint: $Q_{ext,n} = Q_{o,n}$, $P_{ref,n} = P_{o,n}$.

---

## HJSON configuration

```hjson
ppcs: [{
  type:    "repc_d",
  name:    "repc1",
  reg_bus: "POI",
  weccs:   ["gen1", "gen2"],

  devices: [
    {name: "gen1",
     Kw: 0.5,  Kz: 0.5,       // Q and P dispatch weights
     Tw: 0.02, Tz: 0.02,       // Q and P dispatch filter time constants (s)
     Qmax: 0.4, Qmin: -0.4,    // per-device Q limits (pu on S_base)
     Qo:   0.0,                 // initial Q setpoint (pu on S_base)
     Pmax: 1.0, Pmin: 0.0,     // per-device P limits
     Po:   0.8},                // initial P setpoint (pu on S_base)
    {name: "gen2",
     Kw: 0.5,  Kz: 0.5,
     Tw: 0.02, Tz: 0.02,
     Qmax: 0.4, Qmin: -0.4, Qo: 0.0,
     Pmax: 0.8, Pmin: 0.0,  Po: 0.6}
  ],

  // Flags
  RefFlg: 1,  VcmpFlg: 0,  Freq_flag: 1,
  QVFlag: 1,  Pefd_Flag: 1,  Ffwrd_Flag: 0,

  // Q/V controller
  Tfltr: 0.02, Tc: 0.02, Tfrq: 0.02,
  Kp: 0.5, Ki: 3.0, Kc: 0.0,
  Tft: 0.0, Tfv: 0.05,
  emax: 0.3, emin: -0.3,
  dbd1: -0.01, dbd2: 0.01,
  Qmax: 0.4, Qmin: -0.4,
  Vfrz: 0.7, Vrefmax: 1.1, Vrefmin: 0.9,

  // P / frequency controller
  Tp: 0.02,
  Kpg: 0.5, Kig: 0.25,
  Pmax: 1.0, Pmin: 0.0, Tlag: 0.7,
  fdbd1: -0.0006, fdbd2: 0.0006,
  Ddn: 103.33, Dup: 103.33,
  femax: 999.0, femin: -999.0,
  pimax: 1.5, pimin: -0.5,

  Vref: 1.0, Qref: 0.0, Pref_r: 1.4
}]
```

---

## Parameters

| Symbol | Variable | Default | Units | Description |
|---|---|---|---|---|
| $T_{fltr}$ | `Tfltr` | 0.02 | s | V and Q filter time constant |
| $T_p$ | `Tp` | 0.02 | s | Active power filter time constant |
| $T_{frq}$ | `Tfrq` | 0.02 | s | Frequency filter time constant |
| $T_c$ | `Tc` | 0.02 | s | Reactive-current compensation filter |
| $K_p$, $K_i$ | `Kp`, `Ki` | 0.5, 3.0 | pu | Q/V PI gains |
| $K_c$ | `Kc` | 0.0 | pu | Reactive-current compensation gain |
| $dbd_1$, $dbd_2$ | `dbd1`, `dbd2` | ±0.01 | pu | Asymmetric Q/V deadband |
| $e_{max}$, $e_{min}$ | `emax`, `emin` | ±0.3 | pu | Q/V error limits |
| $Q_{max}$, $Q_{min}$ | `Qmax`, `Qmin` | ±0.4 | pu | Plant Q command limits (absolute) |
| $V_{refmax}$, $V_{refmin}$ | `Vrefmax`, `Vrefmin` | 1.1, 0.9 | pu | Voltage reference limits |
| $K_{pg}$, $K_{ig}$ | `Kpg`, `Kig` | 0.5, 0.25 | pu | P PI gains |
| $P_{max}$, $P_{min}$ | `Pmax`, `Pmin` | 1.0, 0.0 | pu | Plant P command limits (absolute) |
| $T_{lag}$ | `Tlag` | 0.7 | s | P output lag time constant |
| $f_{dbd1}$, $f_{dbd2}$ | `fdbd1`, `fdbd2` | ±0.0006 | pu | Frequency droop deadbands |
| $D_{dn}$, $D_{up}$ | `Ddn`, `Dup` | 103.33 | pu/pu | Droop gains |
| $pi_{max}$, $pi_{min}$ | `pimax`, `pimin` | 1.5, −0.5 | pu | P PI output limits |

## Per-device parameters (`devices` list)

| Symbol | Key | Description |
|---|---|---|
| $K_{w,n}$ | `Kw` | Q dispatch weight |
| $K_{z,n}$ | `Kz` | P dispatch weight |
| $T_{w,n}$ | `Tw` | Q dispatch filter time constant (s) |
| $T_{z,n}$ | `Tz` | P dispatch filter time constant (s) |
| $Q_{max,n}$, $Q_{min,n}$ | `Qmax`, `Qmin` | Per-device Q limits (pu on S_base) |
| $Q_{o,n}$ | `Qo` | Initial Q setpoint for device n |
| $P_{max,n}$, $P_{min,n}$ | `Pmax`, `Pmin` | Per-device P limits |
| $P_{o,n}$ | `Po` | Initial P setpoint for device n |

## Flags

| Flag | 0 | 1 |
|---|---|---|
| `RefFlg` | Q control ($Q_{ref}$ − $Q_{brn,flt}$) | V control ($V_{ref}$ − $V_{reg,flt}$) |
| `VcmpFlg` | Reactive droop | Line drop compensation |
| `Freq_flag` | $P_{ref}$ = constant | Frequency droop active |
| `QVFlag` | Q controller disabled | Q controller active |
| `Pefd_Flag` | Bypass P measurement | Use P feedback |
| `Ffwrd_Flag` | No feedforward | Add $P_{brn,flt}$ after PI |

---

## Source

- Module: `pydae.bps.ppcs.repc_d`
- File: [`packages/pydae-bps/src/pydae/bps/ppcs/repc_d.py`](https://github.com/pydae/pydae/tree/main/packages/pydae-bps/src/pydae/bps/ppcs/repc_d.py)
- Reference: WECC *REPC_B.pdf* — "Convert REPC_B to REPC_D", June 2024
  (recommends retiring REPC_B in favour of REPC_D)
