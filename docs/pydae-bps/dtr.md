# Dynamic Thermal Rating (DTR) lines

A **DTR line** adds a thermal model of the conductor to an ordinary
π-section, so the conductor temperature becomes a state of the DAE and the
line rating follows the real operating and weather conditions instead of a
fixed nameplate ampacity.

Two thermal models are provided, picked per line:

| Model | Standard | Use for | Module |
|---|---|---|---|
| **overhead** (default) | IEEE-738 | bare conductors in air | {mod}`pydae.bps.lines.line_dtr` |
| **cable** | IEC 60287 / 60853 | buried XLPE cables | {mod}`pydae.bps.lines.line_dtr_cable` |

## Quick start

Tag a line with `dtr` and (optionally) reference a reusable conductor *code*:

```hjson
dtr_codes: [
  // Overhead bare aluminium conductor (IEEE-738)
  {name: "ohl_A1", D_conductor: 0.0102, Cp: 900.0, Mass_linear: 0.17, K_dtr: 1.0,
   t_ambient: 25.0, v_wind: 1.0, angle_wind: 90.0, irradiance: 1000.0},
  // Buried XLPE cable (IEC 60853 two-node)
  {name: "ug_XLPE", model: "cable", dc: 0.0124, t_ins: 0.0055, De: 0.0342,
   rho_soil: 1.0, L_depth: 0.8, t_soil: 15.0}
]
lines: [
  {bus_j: "B12", bus_k: "B13", R_km: 0.51,  X_km: 0.37, Bs_km: 3.2e-6, km: 3.0, dtr: "ohl_A1"},
  {bus_j: "B01", bus_k: "B02", R_km: 0.501, X_km: 0.716, Bs_km: 4.7e-5, km: 2.8,
   dtr: {code: "ug_XLPE", v_wind: 0.0}}   // a code with per-line overrides
]
```

The `dtr` field of a line may be:

- `true` — use the built-in defaults;
- an inline `{...}` dict — those values;
- `"code_name"` — copy a named entry from `dtr_codes`;
- `{code: "code_name", ...}` — that code with per-line overrides.

Defining each cable type once in `dtr_codes` keeps the `lines` section compact.

## Overhead model (IEEE-738)

A single conductor node with the steady/transient heat balance

$$ m\,C_p\,\frac{dT}{dt} = (q_j + q_s) - (q_c + q_r), $$

where $q_j = I^2 R_{ac}(T)$ is Joule heating, $q_s$ the absorbed solar gain,
$q_c$ forced+natural convective cooling (Reynolds/Rayleigh/Nusselt correlations)
and $q_r$ the Stefan-Boltzmann radiative loss. Weather (`t_ambient`, `v_wind`,
`angle_wind`, `irradiance`) are runtime-settable inputs. See
{doc}`models/lines/line_dtr` for the full parameter list.

## Cable model (IEC 60853, two-node)

A buried cable is conduction-cooled. The transient response reduces to a
two-loop RC ladder (Van Wormer split):

```
   W_c + p·W_d            (1-p)·W_d + λ1·W_c
        │                          │
      [T_c]──T1──[T_e]──────T4──────[T_soil]
       Q1          Q2
```

$$ Q_1\,\dot T_c = (W_c + p W_d) - (T_c - T_e)/T_1, $$
$$ Q_2\,\dot T_e = (T_c - T_e)/T_1 + (1-p)W_d + \lambda_1 W_c - (T_e - T_{soil})/T_4. $$

The thermal resistances `T1` (insulation) and `T4` (soil) follow IEC 60287;
the capacitances `Q1`/`Q2` and Van Wormer factor `p` are derived from the cable
geometry and material heat capacities. Soil dynamics are lumped over a finite
shell (`soil_shell`) — a transparent simplification of the IEC 60853
exponential-integral soil response. See {doc}`models/lines/line_dtr_cable`.

## Closed-loop temperature-dependent resistance

The series admittance of a DTR line is **not** a constant parameter. It is an
algebraic variable evaluated from the temperature-dependent resistance

$$ R(T) = R_{20}\,\bigl(1 + \alpha_R\,(T-20)\bigr), $$

so conductor heating feeds back into the network resistance, ohmic losses and
voltage drop — and that extra loss feeds back into more heating. One resistance
per line drives **both** the heat balance and the network admittance: `R_20`
comes from the line's `R_km` and a single coefficient `alpha_R` (default
`0.004`, aluminium). Set `alpha_R: 0` to recover a constant-resistance
(open-loop) network while keeping the thermal model.

```{note}
For cables, `R_km` is the positive-sequence resistance (it includes
screen/sheath return-path losses), so using it for conductor heating is a slight
over-estimate. Add `R_ac_20` to a cable's `dtr_codes` entry to use the pure
conductor AC resistance for heating while the network still uses `R_km`.
```

## Reporting

{mod}`pydae.bps.utils.dtr_report` summarises the thermal state after `ini()`
(or `run()` + `post()`):

```python
from pydae.bps.utils.dtr_report import report_dtr
report_dtr(model, "my_network.hjson")
```

It prints overhead lines (temperature, heat balance, convection) and cable lines
(conductor/sheath/soil temperatures) as Markdown tables, including a **ΔR (%)**
column — the closed-loop resistance rise `α·(T−20)`, i.e. the extra series
resistance the network carries because the line is hot.

## References

- IEEE Std 738 — *Calculation of the Current-Temperature Relationship of Bare
  Overhead Conductors.*
- IEC 60287 — *Electric cables — Calculation of the current rating* (steady
  state, thermal resistances).
- IEC 60853 — *Calculation of the cyclic and emergency current rating of
  cables* (transient, two-loop reduction, Van Wormer factor).
- CIGRE TB 601 / TB 880 — real-time thermal rating of overhead lines / cables.
```
