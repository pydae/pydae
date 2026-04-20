# st1

*Automatic voltage regulators — pydae-bps model.*

## Model description

IEEE 421.5 ST1A static excitation system (REE NTS parameter set).

This controller implements the bus-fed static excitation system IEEE
type ST1A (IEEE Std 421.5). The model is a lead-lag compensator
followed by a high-gain amplifier with hard output limits; the field
voltage is $E_{fd} = V_R$ (no rectifier loading since $K_C = 0$). With
the REE NTS parameter set the amplifier time constant $T_A = 0$ (pure
gain, no state) and the rate-feedback stabiliser is disabled
($K_F = 0$, no washout state), so the dynamic order is reduced to two.

**Signal path**

The terminal voltage passes through a first-order transducer with
state $v_c$,

$$\frac{d v_c}{dt} = \frac{V - v_c}{T_R}$$

The summing junction forms the voltage error

$$\varepsilon = v^{\star} - v_c + v_s$$

where $v^{\star}$ is the voltage reference and $v_s$ is the
supplementary PSS input. The error is clipped by the input limiter,

$$\varepsilon_{lim} = \mathrm{sat}(\varepsilon,\; V_{Imin},\; V_{Imax})$$

and fed to the lead-lag compensator $(1 + s T_C)/(1 + s T_B)$ with
internal state $x_{lead}$,

$$\frac{d x_{lead}}{dt} = \frac{\varepsilon_{lim} - x_{lead}}{T_B}$$
$$y_{lead} = (\varepsilon_{lim} - x_{lead}) \frac{T_C}{T_B} + x_{lead}$$

With $T_A = 0$ the amplifier stage reduces to a pure gain,

$$V_R^{nosat} = K_A \, y_{lead}$$

and the field voltage command is produced by the output limiter,

$$V_R = \mathrm{sat}(V_R^{nosat},\; V_{Rmin},\; V_{Rmax})$$

The field voltage is returned as the algebraic variable $v_f$ via the
residual

$$0 = V_R - v_f$$

with $\mathrm{sat}(x, a, b) = \min\{b,\, \max\{a,\, x\}\}$.

**Simplifications under the REE parameter set**

- $T_A = 0$ — amplifier is a pure gain; no state $V_A$.
- $K_F = 0$ — rate-feedback stabiliser disabled; no washout state.
- $K_C = 0$ — no rectifier loading.
- $V_{Imax} = V_{Rmax} = 999$ and $V_{Imin} = V_{Rmin} = -999$ — limits
  effectively inactive, but kept as residuals so the model is
  structurally complete if a future fixture tightens them.

**The ini/run variable swap**

For a generator bus held at a voltage setpoint, the initialisation
problem is PV (active power and voltage magnitude specified, reactive
power and voltage reference unknown) while the subsequent time-domain
simulation is reference-driven (the reference is an input and the
voltage magnitude is solved from the network). ``pydae`` supports this
by allowing the ``y`` and ``u`` partitions to differ between phases
while sharing the same residual equations $g$.

                    ini                 run
    v_f             y_ini               y_run      (algebraic, always solved)
    v_ref           y_ini               u_run      (unknown in ini, input in run)
    V_bus           u_ini               y_run      (pinned in ini, solved in run)

The swap is performed in place at the existing ``y_ini`` index of
``V_bus`` so downstream components that reference ``y_ini`` by integer
index — for example ``vsource``'s ``g[idx_V] = ...`` override —
continue to target the correct equations.

**Configuration**

Example data entry (REE NTS defaults):

    "avr": {"type": "st1",
            "T_R": 0.01, "T_B": 10.0, "T_C": 1.0,
            "K_A": 200.0, "T_A": 0.0,
            "V_Imax": 999.0, "V_Imin": -999.0,
            "V_Rmax": 999.0, "V_Rmin": -999.0,
            "v_ref": 1.0}

The ``v_ref`` field serves two roles: it is the **bus voltage setpoint
during ini()** and the **initial run-phase input**. The optional
``bus`` key selects a remote bus whose voltage is regulated; if
omitted the generator bus is used.

## Usage

```python
from pydae.bps import BpsBuilder

grid = BpsBuilder("my_network.json")
grid.construct("my_system")
```

The `st1` model is instantiated by including an entry in the relevant
section of the network JSON (see [Overview](../../overview.md)).

## Parameters, inputs, states, outputs

### Parameters

| Symbol | Variable | Default | Units | Description |
|---|---|---|---|---|
| $T_R$ | `T_R` | 0.01 | s | Terminal voltage transducer time constant |
| $T_B$ | `T_B` | 10.0 | s | Lead-lag denominator time constant |
| $T_C$ | `T_C` | 1.0 | s | Lead-lag numerator time constant |
| $K_A$ | `K_A` | 200.0 | pu | AVR amplifier gain |
| $T_A$ | `T_A` | 0.0 | s | Amplifier time constant (0 under REE NTS — pure gain, no state) |
| $V_{Imax}$ | `V_Imax` | 999.0 | pu | Upper input-error limit |
| $V_{Imin}$ | `V_Imin` | -999.0 | pu | Lower input-error limit |
| $V_{Rmax}$ | `V_Rmax` | 999.0 | pu | Upper regulator output limit |
| $V_{Rmin}$ | `V_Rmin` | -999.0 | pu | Lower regulator output limit |

### Inputs

| Symbol | Variable | Default | Units | Description |
|---|---|---|---|---|
| $v^{\star}$ | `v_ref` | 1.0 | pu | Voltage reference. Acts as the PV-bus setpoint during ini and as an input during run. |
| $v_s$ | `v_pss` | 0.0 | pu | Supplementary stabilising input (PSS output), added to the voltage error. |

### Dynamic States

| Symbol | Variable | Default | Units | Description |
|---|---|---|---|---|
| $v_c$ | `v_c` |  | pu | Terminal voltage transducer state |
| $x_{lead}$ | `x_lead` |  | pu | Lead-lag compensator internal state |

### Algebraic States

| Symbol | Variable | Default | Units | Description |
|---|---|---|---|---|
| $v_f$ | `v_f` |  | pu | Field-voltage command sent to the synchronous machine exciter (saturated). |


## Source

- Module: `pydae.bps.avrs.st1`
- File: [`packages/pydae-bps/src/pydae/bps/avrs/st1.py`](https://github.com/pydae/pydae/tree/main/packages/pydae-bps/src/pydae/bps/avrs/st1.py)
