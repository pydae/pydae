# st4b

*Automatic voltage regulators — pydae-bps model.*

## Model description

IEEE Std 421.5 type ST4B excitation system with PV-bus initialisation.

The model follows the block diagram of IEEE Std 421.5 type **ST4B**.
Default parameters match those mandated by the REE *Normas Técnicas
de Supervisión* (NTS) for grid-code compliance simulations.

**Signal path**

The compensated terminal voltage passes through a first-order sensor
with time constant $T_R$:

$$\frac{d v_c}{dt} = \frac{V - v_c}{T_R}$$

The voltage error feeds a proportional-integral regulator with hard
limits $V_{RMIN}$, $V_{RMAX}$. The PI is written so that the
state $\xi_r$ carries the *integral contribution to the output*
(rather than the time integral of the error itself), so setting the
integral gain to zero freezes the state cleanly:

$$\frac{d \xi_r}{dt} = K_{IR}\,(v^{\star} - v_c + v_s) - \epsilon_{leak}\,\xi_r$$
$$v_r^{nosat} = K_{PR}\,(v^{\star} - v_c + v_s) + \xi_r$$
$$v_r = \mathrm{sat}(v_r^{nosat}, V_{RMIN}, V_{RMAX})$$

The $\epsilon_{leak} = 10^{-6}$ term is a tiny self-decay that
guarantees a unique equilibrium for $\xi_r$ when $K_{IR} = 0$;
with any reasonable $K_{IR} > 0$ it is negligible (time constant
$10^6$ s).

A first-order lag of time constant $T_A$ separates the outer and
inner loops:

$$\frac{d x_a}{dt} = \frac{v_r - x_a}{T_A}$$

The inner PI regulates field current; its error $\varepsilon_m = x_a
- K_G v_f$ reduces to $x_a$ in the REE configuration ($K_G = 0$).
Same integral-state convention (with the same leakage term) as the
outer loop:

$$\frac{d \xi_m}{dt} = K_{IM}\,\varepsilon_m - \epsilon_{leak}\,\xi_m$$
$$v_m^{nosat} = K_{PM}\,\varepsilon_m + \xi_m$$
$$v_m = \mathrm{sat}(v_m^{nosat}, V_{MMIN}, V_{MMAX})$$

The leakage term matters here: the REE default $K_{IM} = 0$ would
otherwise leave $\xi_m$ as an unconstrained state (zero row in the
ini Jacobian, effectively singular). The leakage pins $\xi_m$ to 0
at steady state when $K_{IM} = 0$.

The exciter voltage $V_E$ and rectifier function $F_{EX}$ are given
in the standard as

$$V_E = \left|\, j K_P \bar{V}_T + \left(K_I + j K_P X_L e^{j\theta_P}\right)\,\bar{I}_T\,\right|$$
$$F_{EX}(I_N) = \text{piecewise}, \quad I_N = \frac{K_C I_{FD}}{V_E}$$

For the REE default parameters ($K_I = 0$, $X_L = 0$, $\theta_P = 0$)
the first expression collapses to $V_E = K_P V$; and $K_C = -0.08 < 0$
makes $I_N \le 0$ so $F_{EX} \equiv 1$ for all physical operating
points. This module implements those simplifications:

$$V_E = K_P V, \qquad F_{EX} = 1$$

Using the full vector $V_E$ and a non-trivial $F_{EX}$ requires
wiring the machine's field current $I_{FD}$ and the bus complex
current $\bar{I}_T$ into the AVR — not needed for the REE compliance
simulations this model targets.

The field-voltage command is produced with a hard upper limit
$V_{BMAX}$ (the rectifier ceiling) and returned as the algebraic
variable $v_f$:

$$e_{fd}^{nosat} = v_m V_E$$
$$0 = \min(V_{BMAX},\; e_{fd}^{nosat}) - v_f$$

The lower bound on $v_f$ is set indirectly by $V_{MMIN}$: with $v_m
\ge V_{MMIN}$ and $V_E > 0$, $e_{fd}^{nosat} \ge V_{MMIN} V_E$.

**The ini/run variable swap**

For a generator bus held at a voltage setpoint, the initialisation
problem is PV (active power and voltage magnitude specified, reactive
power and voltage reference unknown) while the subsequent time-domain
simulation is reference-driven (the reference is an input and the
voltage magnitude is solved from the network). ``pydae`` supports
this by allowing the ``y`` (algebraic) and ``u`` (input) partitions
to differ between the ``ini`` and ``run`` phases while sharing the
same set of residual equations $g$.

The table below shows how each quantity is classified in each phase:

                    ini                 run
    v_f             y_ini               y_run      (algebraic, always solved)
    v_ref           y_ini               u_run      (unknown in ini, input in run)
    V_bus           u_ini               y_run      (pinned in ini, solved in run)

List cardinalities are preserved: ``|y_ini| == |y_run|`` and ``|g|``
is unchanged. The swap is performed in place at the existing
``y_ini`` index of ``V_bus`` (added earlier by the bus builder) so
that downstream components that reference ``y_ini`` by integer
index — for example ``vsource``'s ``g[idx_V] = ...`` override —
continue to target the correct equations.

No artificial ``xi_v`` integrator is used: the PV-bus initialisation
is handled entirely by the swap.

**Value transfer between phases**

After ``ini()`` converges, ``ini2run()`` automatically copies solved
values to the run-phase state:

- ``v_ref`` is in ``y_ini`` and appears in ``u_run``: its solved
  value becomes the run-phase input, so the operating point is
  preserved without the user having to pass ``v_ref`` manually.
- ``V_bus`` is in ``u_ini`` and appears in ``y_run``: its pinned
  setpoint is used as the starting value of the run-phase solver.

**Configuration**

Example data entry (REE NTS ST4B defaults)::

    "avr": {"type": "st4b",
            "T_R": 0.02,
            "K_PR": 3.15, "K_IR": 3.15,
            "V_RMAX": 1.0, "V_RMIN": -0.87,
            "T_A": 0.02,
            "K_PM": 1.0, "K_IM": 0.0,
            "V_MMAX": 1.0, "V_MMIN": -0.87,
            "K_G": 0.0, "K_P": 6.5,
            "V_BMAX": 8.0,
            "v_ref": 1.0}

The ``v_ref`` field serves two roles: it is the **bus voltage setpoint
during ini()** (since ``v_ref`` itself is unknown there) and the
**initial run-phase input** (overwritten by the value solved during
ini via ``ini2run``). The optional ``bus`` key selects a remote bus
whose voltage is regulated; if omitted the generator bus is used.

Parameters $K_I$, $X_L$, $\theta_P$, $K_C$ from the full IEEE ST4B
spec are currently fixed to the simplified form documented above; if
needed in the future, pass them through ``avr_data`` and extend the
signal path.

## Usage

```python
from pydae.bps import BpsBuilder

grid = BpsBuilder("my_network.json")
grid.construct("my_system")
```

The `st4b` model is instantiated by including an entry in the relevant
section of the network JSON (see [Overview](../../overview.md)).

## Parameters, inputs, states, outputs

### Parameters

| Symbol | Variable | Default | Units | Description |
|---|---|---|---|---|
| $T_R$ | `T_R` | 0.02 | s | Terminal-voltage sensor time constant |
| $K_{PR}$ | `K_PR` | 3.15 | pu | Outer voltage-regulator proportional gain |
| $K_{IR}$ | `K_IR` | 3.15 | pu/s | Outer voltage-regulator integral gain |
| $V_{RMAX}$ | `V_RMAX` | 1.0 | pu | Upper limit on outer-regulator output |
| $V_{RMIN}$ | `V_RMIN` | -0.87 | pu | Lower limit on outer-regulator output |
| $T_A$ | `T_A` | 0.02 | s | Inter-stage lag time constant |
| $K_{PM}$ | `K_PM` | 1.0 | pu | Inner field-current regulator proportional gain |
| $K_{IM}$ | `K_IM` | 0.0 | pu/s | Inner field-current regulator integral gain (REE doc writes K_IN; interpreted as K_IM per IEEE 421.5 ST4B) |
| $V_{MMAX}$ | `V_MMAX` | 1.0 | pu | Upper limit on inner-regulator output |
| $V_{MMIN}$ | `V_MMIN` | -0.87 | pu | Lower limit on inner-regulator output |
| $K_G$ | `K_G` | 0.0 | pu | Field-voltage feedback gain into inner loop. Default 0 disables the feedback (REE NTS default). |
| $K_P$ | `K_P` | 6.5 | pu | Exciter voltage gain. With X_L = K_I = theta_P = 0 (REE default), V_E = K_P V. |
| $V_{BMAX}$ | `V_BMAX` | 8.0 | pu | Rectifier ceiling on field-voltage command |

### Inputs

| Symbol | Variable | Default | Units | Description |
|---|---|---|---|---|
| $v^{\star}$ | `v_ref` | 1.0 | pu | Voltage reference. Acts as the PV-bus setpoint during ini (where it is solved for) and as an input during run. |
| $v_s$ | `v_pss` | 0.0 | pu | Supplementary stabilising input (PSS output), added to the voltage error. |

### Dynamic States

| Symbol | Variable | Default | Units | Description |
|---|---|---|---|---|
| $v_c$ | `v_c` |  | pu | Sensed terminal voltage (T_R lag output) |
| $\xi_r$ | `xi_r` |  | pu | Outer-PI integrator — integral contribution to v_r^{nosat}. |
| $x_a$ | `x_a` |  | pu | Inter-stage lag output |
| $\xi_m$ | `xi_m` |  | pu | Inner-PI integrator — integral contribution to v_m^{nosat}. Pinned to 0 by leakage when K_IM = 0. |

### Algebraic States

| Symbol | Variable | Default | Units | Description |
|---|---|---|---|---|
| $v_f$ | `v_f` |  | pu | Field-voltage command sent to the synchronous machine exciter (post rectifier ceiling). |


## Source

- Module: `pydae.bps.avrs.st4b`
- File: [`packages/pydae-bps/src/pydae/bps/avrs/st4b.py`](https://github.com/pydae/pydae/tree/main/packages/pydae-bps/src/pydae/bps/avrs/st4b.py)
