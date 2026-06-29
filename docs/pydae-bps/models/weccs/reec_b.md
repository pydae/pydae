# REEC_B

*WECC Renewable Energy Electrical Control model B —
`pydae.bps.weccs.reec_b`*

---

## Purpose

REEC_B is the **local electrical control** layer of the WECC renewable model
stack. It sits between the plant controller (REPC_A) and the grid-interface
converter (REGC_A), translating active and reactive power references into the
current commands $I_{pcmd}$ and $I_{qcmd}$ that drive REGC_A.

REEC_B is activated by including a ``reec`` sub-dict in the HJSON ``weccs``
entry — exactly as an AVR is nested inside a ``syns`` entry.

---

## Signal chain

```
REPC_A (or external)
  │  Pref, Qext
  ▼
┌──────────────────────────────────────────────────────────────┐
│  REEC_B                                                      │
│                                                              │
│  Vt ──► [1/(1+s·Trv)] ──► Vt_flt                           │
│                                 │                            │
│  Qext ──► (Vref_ctrl) ◄─── Q-PI (Vflag=0)                  │
│                  │                                           │
│                  ▼                                           │
│            inner V-PI ──► [1/(1+s·Tiq)] ──► Vl_flt          │
│                                                 │            │
│  VRT: Kqv·db(Vt_flt−Vref0) = Iqinj ────────────►(+)         │
│                                                 ▼            │
│                                              Iqcmd ──► REGC_A│
│                                                              │
│  Pref ──► [1/(1+s·Tpord)] ──► P_flt ÷ Vt_flt ──► Ipcmd     │
│                  rate-limits: dPmin, dPmax                   │
└──────────────────────────────────────────────────────────────┘
```

---

## Mathematical model

### Terminal voltage filter

$$\dot{V}_{t,flt} = \frac{V_t - V_{t,flt}}{T_{rv}}$$

### Active power filter

$$\dot{x}_{Pe} = \frac{I_p V_t - x_{Pe}}{T_p}$$

Used when $PF_{flag}=1$ to compute the constant-PF reactive reference.

### Q / voltage control path

**Reactive reference** (selected by $PF_{flag}$):

$$Q_{ref} = \begin{cases}
  Q_{ext} & PF_{flag} = 0 \\
  x_{Pe}\,\tan(\phi_{ref}) & PF_{flag} = 1
\end{cases}$$

**Voltage reference** (selected by $V_{flag}$):

$$V_{ref,ctrl} = \begin{cases}
  Q_{ext} & V_{flag} = 1
    \text{ (Qext acts as voltage reference directly)} \\
  \mathrm{clip}(K_{qp}(Q_{ref} - Q_e) + x_{Kqi},\; V_{min},\; V_{max})
    & V_{flag} = 0
\end{cases}$$

where $Q_e = I_q V_t$ is the measured reactive power.

**Q-PI integrator** (active when $V_{flag}=0$):

$$\dot{x}_{Kqi} = K_{qi}(Q_{ref} - Q_e)$$

**Inner voltage PI** (active when $Q_{flag}=1$):

$$\dot{x}_{Kvi} = K_{vi}(V_{t,flt} - V_{ref,ctrl})$$
$$K_{vi,out} = \mathrm{clip}(K_{vp}(V_{t,flt} - V_{ref,ctrl}) + x_{Kvi},\;
                              I_{q\min},\; I_{q\max})$$

Note the sign: positive voltage error (over-voltage) produces a positive
output that drives $I_q$ toward zero (less reactive injection).

**Reactive current lag**:

$$\dot{V}_{l,flt} =
  \mathrm{clip}\!\left(\frac{K_{vi,out} - V_{l,flt}}{T_{iq}},\;
                        dP_{min},\; dP_{max}\right)$$

### VRT supplementary reactive injection

During voltage ride-through events, a fast reactive injection supplements
the PI path:

$$\mathrm{db}_{out} = \max(V_{t,flt} - V_{ref0} - dbd_2, 0)
                    + \min(V_{t,flt} - V_{ref0} - dbd_1, 0)$$

$$I_{q,inj} = \mathrm{clip}(K_{qv}\,\mathrm{db}_{out},\; I_{qll},\; I_{qhl})$$

At nominal voltage ($V_{t,flt} \approx V_{ref0}$) the deadband zeroes this
injection.

### Current limit logic

Available current is shared between $I_{pcmd}$ and $I_{qcmd}$ according to
$Pq_{flag}$:

| $Pq_{flag}$ | Priority | $I_{pmax}$ | $I_{q\max}$, $I_{q\min}$ |
|---|---|---|---|
| 0 | Reactive | $\sqrt{\max(I_{max}^2 - V_{l,flt}^2,\,\varepsilon)}$ | $\pm I_{max}$ |
| 1 | Active | $I_{max}$ | $\pm\sqrt{\max(I_{max}^2 - I_{p,flt}^2,\,\varepsilon)}$ |

### Algebraic outputs

$$I_{pcmd} = \mathrm{clip}\!\left(\frac{P_{flt}}{\max(V_{t,flt},\,0.01)},\;
                                   P_{min},\; I_{pmax}\right)$$

$$I_{qcmd} = \mathrm{clip}(V_{l,flt} + I_{q,inj},\; I_{q\min},\; I_{q\max})$$

### Active power lag

$$\dot{P}_{flt} =
  \mathrm{clip}\!\left(\frac{P_{ref} - P_{flt}}{T_{pord}},\;
                        dP_{min},\; dP_{max}\right)$$

---

## Initialization

At $t=0$, all state derivatives are zero:

| State | Initial value |
|---|---|
| $V_{t,flt}$ | $V_0$ (terminal voltage from power flow) |
| $x_{Pe}$ | $I_{p,0} V_0$ (filtered active power = steady-state) |
| $x_{Kqi}$ | **mode-dependent.** $V_{flag}=1$ (voltage control): 0 — the Q-PI is decoupled and integrates an irrelevant error. $V_{flag}=0$ (Q control / PQ mode): $V_{t,0}$ — the integrator output is the $V_{ref,ctrl}$ that the inner V-PI matches to $V_t$, so seeding it to 0 puts the model far from steady state and the Newton init either NaNs out or settles on $Q\!\approx\!0$ instead of $Q_{ext}$. |
| $x_{Kvi}$ | $I_{qcmd,0} = -I_{q,0}$ |
| $V_{l,flt}$ | $I_{qcmd,0} = -I_{q,0}$ |
| $P_{flt}$ | $P_{ref,0}$ |
| $I_{pcmd}$ (algebraic) | $I_{p,0}$ |
| $I_{qcmd}$ (algebraic) | $-I_{q,0}$ |

---

## The ini/run variable swap

REEC_B promotes $I_{pcmd}$ and $I_{qcmd}$ from external inputs (set by
the user before REEC_B is attached) to **algebraic states** driven by its
own residuals. After calling ``reec_b()``, those two symbols no longer
appear in ``u_run_dict``; instead they appear in ``y_ini`` and ``y_run``
with their own algebraic constraints.

New inputs added to ``u_run_dict``: ``Pref_{name}`` and ``Qext_{name}``.
These are overridden by REPC_A when a ``ppcs`` section is present.

---

## HJSON configuration

```hjson
weccs: [
  {
    type: "regc_a",
    bus:  "1",
    // ... REGC_A parameters ...

    reec: {
      // Flags (Python values — clean expression tree)
      Vflag:  1,    // 0 = Q control  1 = voltage control
      Qflag:  1,    // 0 = bypass inner V-PI  1 = engage
      PFflag: 0,    // 0 = constant Q / Vref  1 = constant PF
      Pqflag: 0,    // 0 = Q priority  1 = P priority

      // Time constants
      Trv:   0.02,  // terminal voltage filter (s)
      Tp:    0.02,  // active power filter (s)
      Tiq:   0.02,  // reactive current lag (s)
      Tpord: 0.1,   // active power order lag (s)

      // Inner V-PI gains
      Kvp:   1.0,   // proportional gain
      Kvi:  40.0,   // integral gain

      // Q-PI gains (Vflag=0 only)
      Kqp:   0.0,
      Kqi:   0.0,

      // VRT reactive injection
      Vref0:  1.0,
      dbd1:  -0.05, // deadband lower limit (overvoltage, < 0)
      dbd2:   0.05, // deadband upper limit (undervoltage, > 0)
      Kqv:    2.0,
      Iqhl:   1.05, // VRT Iqinj upper limit (pu)
      Iqll:  -1.05, // VRT Iqinj lower limit (pu)

      // Limits
      Vmax:  1.1,   Vmin: 0.9,
      Imax:  1.1,
      Pmax:  1.0,   Pmin:  0.0,
      dPmax: 999.0, dPmin: -999.0,

      // Initial references (overridden by REPC_A if present)
      Pref:  0.8,
      Qext:  1.0    // voltage reference when Vflag=1
    }
  }
]
```

---

## Parameters

| Symbol | Variable | Default | Units | Description |
|---|---|---|---|---|
| $T_{rv}$ | `Trv` | 0.02 | s | Terminal voltage filter time constant |
| $T_p$ | `Tp` | 0.02 | s | Active power filter time constant |
| $T_{iq}$ | `Tiq` | 0.02 | s | Reactive current regulator lag time constant |
| $T_{pord}$ | `Tpord` | 0.1 | s | Active power order lag time constant |
| $K_{vp}$ | `Kvp` | 1.0 | pu/pu | Inner voltage PI proportional gain |
| $K_{vi}$ | `Kvi` | 40.0 | pu/pu·s | Inner voltage PI integral gain |
| $K_{qp}$ | `Kqp` | 0.0 | pu/pu | Q-PI proportional gain (Vflag=0) |
| $K_{qi}$ | `Kqi` | 0.0 | pu/pu·s | Q-PI integral gain (Vflag=0) |
| $V_{ref0}$ | `Vref0` | 1.0 | pu | Initial voltage reference |
| $dbd_1$ | `dbd1` | −0.05 | pu | VRT deadband lower bound (overvoltage, < 0) |
| $dbd_2$ | `dbd2` | 0.05 | pu | VRT deadband upper bound (undervoltage, > 0) |
| $K_{qv}$ | `Kqv` | 2.0 | pu/pu | VRT reactive current injection gain |
| $I_{qhl}$ | `Iqhl` | 1.05 | pu | VRT $I_{q,inj}$ upper limit |
| $I_{qll}$ | `Iqll` | −1.05 | pu | VRT $I_{q,inj}$ lower limit (< 0) |
| $V_{max}$ | `Vmax` | 1.1 | pu | Q-PI output upper voltage limit |
| $V_{min}$ | `Vmin` | 0.9 | pu | Q-PI output lower voltage limit |
| $I_{max}$ | `Imax` | 1.1 | pu | Maximum apparent current magnitude |
| $P_{max}$ | `Pmax` | 1.0 | pu | Maximum active power |
| $P_{min}$ | `Pmin` | 0.0 | pu | Minimum active power |
| $dP_{max}$ | `dPmax` | 999.0 | pu/s | Active power up-ramp rate limit |
| $dP_{min}$ | `dPmin` | −999.0 | pu/s | Active power down-ramp rate limit |

## Flags

| Flag | Values | Effect |
|---|---|---|
| `Vflag` | 0 / 1 | 0 = Q control (Q-PI engaged: $K_{qp}/K_{qi}$ drive $V_{ref,ctrl}$); 1 = voltage control ($Q_{ext}$ acts directly as the V reference) |
| `Qflag` | 0 / 1 | 0 = bypass inner V-PI; 1 = engage inner V-PI ($K_{vp}/K_{vi}$ drive $I_{qcmd}$) |
| `PFflag` | 0 / 1 | 0 = use $Q_{ext}$ as Q reference; 1 = constant power factor (Q derived from $x_{Pe}$) |
| `Pqflag` | 0 / 1 | 0 = Q priority in current limits; 1 = P priority |

```{warning}
**The pydae `Vflag` / `Qflag` convention is the *opposite* of the
PSSE/PSLF WECC standard.** Configurations transplanted from PSSE
without flipping these flags will silently end up in the wrong mode
(typically "voltage control with no Q tracking").
```

---

## Operating modes

The four flags select which loops in the reactive path are closed and
what each one tracks. Three combinations are supported by this
implementation; the fourth is structurally singular and pydae will
abort at model build.

| `Vflag` | `Qflag` | `PFflag` | What's tracked | Use case |
|:---:|:---:|:---:|---|---|
| 1 | 1 | 0 | $V_t \to Q_{ext}$ (Qext is the *voltage* setpoint) | local voltage regulator |
| 0 | 1 | 0 | $Q \to Q_{ext}$ via cascaded $K_{qp}/K_{qi}$ → $K_{vp}/K_{vi}$ | **PQ generator** (constant-Q injection) |
| 0 | 1 | 1 | $Q \to x_{Pe}\tan(\phi)$ | constant power factor |
| 0 | 0 | — | — | **structurally rank-deficient** — $x_{Kvi}$ has no defining equation, `rootfinder` aborts with `sprank(J) < n` |

Disable the VRT (voltage-ride-through) supplementary injection for all
three working modes by setting `Kqv = 0` (and widening `dbd1` / `dbd2`
defensively); otherwise an off-nominal initial $V_t$ adds an unmodeled
$I_{q,inj}$ during the power-flow initialization.

### Example — PQ generator (track $Q_{ext}$)

The supported way to make an inverter behave as a fixed-PQ source is the
**cascaded** mode: the outer Q-PI produces a $V_{ref,ctrl}$ that the
inner V-PI then tracks to a corresponding $I_{qcmd}$.

```python
wecc = {
    "type": "regc_a",
    "name": "PV1",
    "bus":  "LV_PV1",
    "S_n":  3_000_000.0,    # inverter nameplate (VA)

    # ---- regc_a internals (defaults shown) ----
    "Tg": 0.02, "Tfltr": 0.02,
    "Lvplsw": 1, "Lvpl1": 1.22,
    "Zerox": 0.4, "Brkpt": 0.9,
    "rrpwr": 10.0,
    "lvpnt0": 0.4, "lvpnt1": 0.8,
    "Volim": 1.2,  "Iolim": -1.3, "Khv": 0.7,
    "Iqrmax":  999.0, "Iqrmin": -999.0,

    # outer regc_a current-state seeds — MUST match reec.Pref / reec.Qext
    "Ipcmd":  0.5,
    "Iqcmd":  0.1,
    "V_ini":  1.0,

    "reec": {
        # ---- PQ-mode flags ----
        "Vflag":  0,        # Q control engaged (Q-PI active)
        "Qflag":  1,        # inner V-PI engaged (consumes V_ref_ctrl)
        "PFflag": 0,        # Qext as Q setpoint (not power factor)
        "Pqflag": 1,        # P priority on the current limit

        # measurement / order filters
        "Trv": 0.02, "Tp": 0.02, "Tiq": 0.02, "Tpord": 0.1,

        # both PI gains active in cascade
        "Kvp": 1.0,  "Kvi": 40.0,
        "Kqp": 1.0,  "Kqi": 10.0,

        # VRT disabled — pure PQ source, no voltage-droop FRT support
        "Vref0":  1.0,
        "dbd1":  -0.99, "dbd2": 0.99,
        "Kqv":    0.0,
        "Iqhl":   2.0,  "Iqll": -2.0,

        # protection envelope — kept loose so the PF init has headroom
        "Vmax": 1.1, "Vmin": 0.9, "Imax": 2.0,
        "Pmax": 1.0, "Pmin": -0.1,
        "dPmax": 999.0, "dPmin": -999.0,

        # setpoints (pu of S_n)
        "Pref": 0.5,
        "Qext": 0.1,
    },
}
```

Matching `xy_0` seed entries for one inverter (see *Initialization* below
for the per-state derivation):

```python
def reec_b_xy0_pq(inv_id: str, Pref: float, Qext: float,
                  Vt0: float = 1.0) -> dict:
    """xy_0 seed for one REEC_B inverter running in PQ mode (Vflag=0, Qflag=1).

    At steady state with V_t ≈ Vt0:
        I_q  ≈ -Qext / Vt0     (sign per pydae's REGC_A convention)
        x_Kqi ≈ Vt0            (V_ref_ctrl that the inner V-PI matches to V_t)
        x_Kvi ≈ I_qcmd = -I_q
    """
    Iq = -Qext / Vt0
    return {
        f"V_flt_{inv_id}":  Vt0,
        f"Vt_flt_{inv_id}": Vt0,
        f"Vl_flt_{inv_id}": -Iq,
        f"Ip_{inv_id}":     Pref,
        f"Iq_{inv_id}":     Iq,
        f"Ipcmd_{inv_id}":  Pref,
        f"Iqcmd_{inv_id}":  -Iq,
        f"P_flt_{inv_id}":  Pref,
        f"x_Pe_{inv_id}":   Pref,
        f"x_Kqi_{inv_id}":  Vt0,    # !! NOT 0 in Q-control mode
        f"x_Kvi_{inv_id}":  -Iq,
    }
```

The critical seed — and the one most likely to trip up a hand-written
`xy_0` — is **`x_Kqi`**. In voltage-control mode (`Vflag=1`) the Q-PI is
decoupled and `x_Kqi = 0` is fine; in Q-control mode (`Vflag=0`) the Q-PI
is the upstream regulator and its integrator output *is* the
$V_{ref,ctrl}$ that the inner V-PI matches to $V_t$. Seeding it to 0
puts the model far from steady state and the Newton rootfinder either
diverges to NaN or settles on $Q \approx 0$ instead of $Q = Q_{ext}$.

## Inputs

| Symbol | Variable | Default | Units | Description |
|---|---|---|---|---|
| $P_{ref}$ | `Pref` | 0.8 | pu | Active power reference (from REPC_A or external) |
| $Q_{ext}$ | `Qext` | 1.0 | pu | Reactive power / voltage reference (from REPC_A or external) |

## Dynamic states

| Symbol | Variable | Units | Description |
|---|---|---|---|
| $V_{t,flt}$ | `Vt_flt` | pu | Filtered terminal voltage |
| $x_{Pe}$ | `x_Pe` | pu | Filtered active power |
| $x_{Kqi}$ | `x_Kqi` | pu | Q-PI integrator state (active when Vflag=0) |
| $x_{Kvi}$ | `x_Kvi` | pu | Inner V-PI integrator state |
| $V_{l,flt}$ | `Vl_flt` | pu | Reactive current regulator output lag |
| $P_{flt}$ | `P_flt` | pu | Active power order (Tpord lag) |

## Algebraic states (outputs to REGC_A)

| Symbol | Variable | Units | Description |
|---|---|---|---|
| $I_{pcmd}$ | `Ipcmd` | pu | Active current command |
| $I_{qcmd}$ | `Iqcmd` | pu | Reactive current command |

---

## Source

- Module: `pydae.bps.weccs.reec_b`
- File: [`packages/pydae-bps/src/pydae/bps/weccs/reec_b.py`](https://github.com/pydae/pydae/tree/main/packages/pydae-bps/src/pydae/bps/weccs/reec_b.py)
- Reference: WECC Solar Plant Dynamic Modeling Guidelines, April 2014, Section 5.2
