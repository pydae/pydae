# -*- coding: utf-8 -*-
"""
Dynamic Thermal Rating for **underground cables** — IEC 60853 two-node model,
or an n-stage Cauer ladder when two nodes are not enough.

This is the buried-cable counterpart of the overhead IEEE-738 model in
:mod:`pydae.bps.lines.line_dtr`. A cable is not air-cooled: heat is *conducted*
outward through the insulation, sheath and jacket into the surrounding soil.
IEC 60287 gives the steady-state thermal resistances; IEC 60853 reduces the
transient response to a two-loop (two-node) lumped RC ladder via the Van Wormer
capacitance-splitting factor.

Network (per unit length)::

    W_c + p·W_d                     (1-p)·W_d + λ1·W_c
         │                                  │
       [θ_c]───T1───[θ_e]───────T4────────[θ_soil]
        Q1            Q2

    θ_c   conductor temperature                 [°C]   (state)
    θ_e   external / sheath-region temperature  [°C]   (state)
    θ_soil  ambient soil temperature            [°C]   (input)
    W_c   conductor loss  I²·R_ac(θ_c)          [W/m]
    W_d   dielectric loss                       [W/m]
    T1    insulation thermal resistance         [K·m/W]  (IEC 60287)
    T4    external/soil thermal resistance       [K·m/W]  (IEC 60287)
    Q1,Q2 lumped thermal capacitances           [J/(m·K)] (Van Wormer split)
    p     Van Wormer factor                      [-]
    λ1    sheath loss factor                      [-]

State equations::

    Q1·dθ_c/dt = (W_c + p·W_d)            − (θ_c − θ_e)/T1
    Q2·dθ_e/dt = (θ_c − θ_e)/T1 + (1−p)·W_d + λ1·W_c − (θ_e − θ_soil)/T4

T1, T4, Q1, Q2 and p are derived from the cable geometry/materials in Python
(:func:`_cable_thermal_network`) and registered as **per-line parameters** with
those numeric defaults; any of them can be overridden directly in the ``dtr``
data (or a ``dtr_codes`` entry). The temperature-dependent conductor resistance,
the network current and the soil temperature remain symbolic.

Selected by ``dtr: {model: "cable", ...}`` (or a ``dtr_codes`` entry whose
``model`` is ``"cable"``) on a ``lines[]`` entry.

n-stage ladder form
-------------------
Two nodes cannot span the insulation and the soil at once. Their time constants
here are ~2 min and ~1.5 days, and a daily load cycle falls straight into the
gap between them: on the CIGRE MV benchmark the two-node model reproduces the
steady state exactly but **under-predicts the daily conductor swing by about
2x**, which under-predicts the peak — the unsafe direction for a rating study.

Supplying ``R`` and ``C`` as lists instead of ``T1/T4/Q1/Q2`` builds an
n-stage Cauer ladder, which fixes that::

    dtr: {model: "cable", R: [0.070, 0.478, 0.409, 0.561, 0.099],
                          C: [5.1e2, 9.5e2, 8.6e4, 9.4e5, 3.4e7],
                          lambda1: 0.4822, t_soil: 18.94, ...}

Network (per unit length)::

    W_tot --+--[R1]--+--[R2]--+-- ... --+--[R_{n-1}]--+--[Rn]--+
            |        |        |         |             |        |
           C1       C2       C3        ...           Cn        |
            |        |        |         |             |        |
            +--------+--------+---------+-------------+--------+  theta_soil

    theta_1 = conductor temperature (state, named t_conductor_{line})
    theta_i = i-th ladder node       (state, named t_ladder_{i}_{line})
    W_tot   = (1 + lambda1)*W_c + W_d   total loss, injected at node 1

State equations::

    C1 dtheta_1/dt = W_tot                        - (theta_1-theta_2)/R1
    Ci dtheta_i/dt = (theta_{i-1}-theta_i)/R_{i-1} - (theta_i-theta_{i+1})/Ri
    Cn dtheta_n/dt = (theta_{n-1}-theta_n)/R_{n-1} - (theta_n-theta_soil)/Rn

``sum(R)`` is the steady conductor-to-soil thermal resistance per unit of total
loss, so the DC behaviour is set by the R list alone.

Note the one modelling difference from the two-node form: the ladder takes the
**total** loss at node 1, rather than splitting W_c into the conductor and
lambda1*W_c into the sheath. That is not an approximation being added — it is
where the R/C values come from. :func:`pydae.bps.lines.cable_fem.identify_ladder`
drives the finite-element model with the real spatial loss distribution (W_c in
the conductor, W_s at the screen) and reduces the result, so the identified
R and C already carry the split. Re-injecting it here would double-count it.

``R``/``C`` and ``T1``/``T4``/``Q1``/``Q2``/``p`` are mutually exclusive; giving
both raises, rather than silently honouring one and dropping the other.

:mod:`pydae.bps.lines.cable_fem` derives either form from a 2D finite-element
solution of the real cable geometry — see its ``TwoNodeFit.as_dtr()`` and
``ladder_dtr()``.

.. note::
   The soil capacitance ``Q_soil`` is lumped over a finite shell of outer
   diameter ``soil_shell × De`` — a transparent approximation of the
   semi-infinite-medium response. IEC 60853 proper uses the exponential-integral
   soil response; the two-node reduction trades that for a single dominant soil
   time constant, which is the standard simplification for real-time / control
   studies. Tune ``soil_shell`` (or override ``Q2``/``T4``) to match a reference.

Default geometry follows the CIGRE-575 European MV cable NA2XS2Y 120 mm²
(Table 6.11); soil resistivity, burial depth and material heat capacities use
IEC typical values (not given by the benchmark).
"""

import math

import numpy as np

from pydae.bps.lines.branch_current import (line_currents_pu,
                                            current_magnitude,
                                            current_magnitude_sq)


# Material / installation defaults (SI). Overridable via the dtr data dict.
_DEFAULTS = {
    'dc':        0.0124,   # conductor diameter [m]            (NA2XS2Y dc = 1.24 cm)
    't_ins':     0.0055,   # insulation thickness [m]          (ti = 5.5 mm)
    'De':        0.0342,   # cable overall diameter [m]        (dov = 34.2 mm)
    'rho_ins':   3.5,      # insulation thermal resistivity [K·m/W]  (XLPE)
    'rho_soil':  1.0,      # soil thermal resistivity [K·m/W]
    'L_depth':   0.8,      # laying depth to cable axis [m]
    'cv_cond':   2.5e6,    # conductor volumetric heat capacity [J/(m³·K)] (Al)
    'cv_ins':    2.4e6,    # insulation volumetric heat capacity [J/(m³·K)] (XLPE)
    'cv_soil':   2.0e6,    # soil volumetric heat capacity [J/(m³·K)]
    'soil_shell': 3.0,     # soil-capacitance shell, outer Ø = soil_shell·De [-]
    'R_ac_20':   2.64e-4,  # AC resistance at 20 °C [Ω/m]      (120 mm² Al)
    'alpha_R':   0.004,    # resistance temperature coefficient [1/K] (Al)
    'lambda1':   0.0,      # sheath loss factor [-]
    'Wd':        0.0,      # dielectric loss [W/m]  (negligible at MV)
    'K_dtr':     1.0,      # network-current → conductor-current scaling [-]
    't_soil':    15.0,     # ambient soil temperature [°C] (input default)
}


def descriptions():
    """
    Single source of truth for the underground-cable DTR parameters, inputs,
    states and outputs. Consumed by the docs autosummary generator.

    Geometry/material values are supplied per line through the ``dtr`` field
    (inline dict, ``dtr_codes`` reference, or reference with overrides) whose
    ``model: "cable"`` routes here; the ``data`` column is the corresponding key.
    The lumped network coefficients T1/T4/Q1/Q2/p are derived from the geometry
    (see :func:`_cable_thermal_network`) but may also be supplied directly.

    ``R``/``C`` select the n-stage ladder instead, in which case T1/T4/Q1/Q2/p
    and the ``t_sheath`` output do not exist and the ladder-node states
    ``t_ladder_{i}`` take their place.
    """
    d = []

    # Parameters — cable geometry
    d += [{"type": "Parameter", "tex": "d_c",      "data": "dc",      "model": "dc",      "default": 0.0124, "description": "Conductor diameter", "units": "m"}]
    d += [{"type": "Parameter", "tex": "t_i",      "data": "t_ins",   "model": "t_ins",   "default": 0.0055, "description": "Insulation thickness", "units": "m"}]
    d += [{"type": "Parameter", "tex": "D_e",      "data": "De",      "model": "De",      "default": 0.0342, "description": "Cable overall diameter", "units": "m"}]
    # Parameters — materials / installation
    d += [{"type": "Parameter", "tex": "\\rho_{ins}",  "data": "rho_ins",  "model": "rho_ins",  "default": 3.5,  "description": "Insulation thermal resistivity", "units": "K·m/W"}]
    d += [{"type": "Parameter", "tex": "\\rho_{soil}", "data": "rho_soil", "model": "rho_soil", "default": 1.0,  "description": "Soil thermal resistivity", "units": "K·m/W"}]
    d += [{"type": "Parameter", "tex": "L",            "data": "L_depth",  "model": "L_depth",  "default": 0.8,  "description": "Laying depth to cable axis", "units": "m"}]
    d += [{"type": "Parameter", "tex": "c_{v,c}",      "data": "cv_cond",  "model": "cv_cond",  "default": 2.5e6, "description": "Conductor volumetric heat capacity", "units": "J/(m³·K)"}]
    d += [{"type": "Parameter", "tex": "c_{v,i}",      "data": "cv_ins",   "model": "cv_ins",   "default": 2.4e6, "description": "Insulation volumetric heat capacity", "units": "J/(m³·K)"}]
    d += [{"type": "Parameter", "tex": "c_{v,s}",      "data": "cv_soil",  "model": "cv_soil",  "default": 2.0e6, "description": "Soil volumetric heat capacity", "units": "J/(m³·K)"}]
    d += [{"type": "Parameter", "tex": "k_{soil}",     "data": "soil_shell","model": "soil_shell","default": 3.0, "description": "Soil-capacitance shell, outer Ø = k·De", "units": "-"}]
    # Parameters — electrical / loss
    d += [{"type": "Parameter", "tex": "R_{ac,20}", "data": "R_ac_20", "model": "R_ac_20", "default": 2.64e-4, "description": "AC resistance at 20 °C (defaults to R_km/1000)", "units": "Ω/m"}]
    d += [{"type": "Parameter", "tex": "\\alpha_R", "data": "alpha_R", "model": "alpha_R", "default": 0.004,   "description": "Resistance temperature coefficient", "units": "1/K"}]
    d += [{"type": "Parameter", "tex": "\\lambda_1","data": "lambda1", "model": "lambda1", "default": 0.0,     "description": "Sheath loss factor", "units": "-"}]
    d += [{"type": "Parameter", "tex": "W_d",       "data": "Wd",      "model": "Wd",      "default": 0.0,     "description": "Dielectric loss", "units": "W/m"}]
    d += [{"type": "Parameter", "tex": "K_{dtr}",   "data": "K_dtr",   "model": "K_dtr",   "default": 1.0,     "description": "Network- to conductor-current scaling", "units": "-"}]
    # Parameters — derived lumped network (computed from geometry; overridable)
    d += [{"type": "Parameter", "tex": "T_1", "data": "T1", "model": "T1", "default": "", "description": "Insulation thermal resistance (IEC 60287)", "units": "K·m/W"}]
    d += [{"type": "Parameter", "tex": "T_4", "data": "T4", "model": "T4", "default": "", "description": "External/soil thermal resistance (IEC 60287)", "units": "K·m/W"}]
    d += [{"type": "Parameter", "tex": "Q_1", "data": "Q1", "model": "Q1", "default": "", "description": "Conductor-node thermal capacitance", "units": "J/(m·K)"}]
    d += [{"type": "Parameter", "tex": "Q_2", "data": "Q2", "model": "Q2", "default": "", "description": "External-node thermal capacitance", "units": "J/(m·K)"}]
    d += [{"type": "Parameter", "tex": "p",   "data": "p",  "model": "p_vw", "default": "", "description": "Van Wormer capacitance-split factor", "units": "-"}]
    # Parameters — n-stage ladder (alternative to T1/T4/Q1/Q2/p; mutually exclusive)
    d += [{"type": "Parameter", "tex": "R_i", "data": "R", "model": "R1..Rn", "default": "", "description": "Cauer ladder series thermal resistances, cable outwards — selects the n-stage form", "units": "K·m/W"}]
    d += [{"type": "Parameter", "tex": "C_i", "data": "C", "model": "C1..Cn", "default": "", "description": "Cauer ladder shunt thermal capacitances, cable outwards — selects the n-stage form", "units": "J/(m·K)"}]

    # Input
    d += [{"type": "Input", "tex": "T_{soil}", "data": "t_soil", "model": "t_soil", "default": 15.0, "description": "Ambient soil temperature", "units": "°C"}]

    # Dynamic states
    d += [{"type": "Dynamic State", "tex": "T_{cond}",  "data": "", "model": "t_conductor", "default": "", "description": "Conductor temperature", "units": "°C"}]
    d += [{"type": "Dynamic State", "tex": "T_{sheath}","data": "", "model": "t_sheath",    "default": "", "description": "External/sheath-region temperature (two-node form only)", "units": "°C"}]
    d += [{"type": "Dynamic State", "tex": "T_{i}",     "data": "", "model": "t_ladder_{i}","default": "", "description": "i-th ladder node temperature, i = 2..n (n-stage form only)", "units": "°C"}]

    # Outputs
    d += [{"type": "Output", "tex": "W_c",     "data": "", "model": "W_c",     "default": "", "description": "Conductor (Joule) loss", "units": "W/m"}]
    d += [{"type": "Output", "tex": "W_{tot}", "data": "", "model": "W_total", "default": "", "description": "Total heat input", "units": "W/m"}]
    d += [{"type": "Output", "tex": "R_{ac}",  "data": "", "model": "R_ac",    "default": "", "description": "AC resistance at conductor temperature", "units": "Ω/m"}]
    d += [{"type": "Output", "tex": "I_{cond}","data": "", "model": "I_cond",  "default": "", "description": "Conductor current", "units": "A"}]
    return d


def _cable_thermal_network(d):
    r"""
    Derive the lumped two-node thermal-network coefficients from cable geometry.

    Returns a dict with ``T1, T4, Q1, Q2, p`` (IEC 60287 resistances, Van Wormer
    split capacitances). Values already present in *d* are passed through so a
    user can override any computed coefficient directly.
    """
    g = lambda k: d.get(k, _DEFAULTS[k])

    dc, t_ins, De = g('dc'), g('t_ins'), g('De')
    De_ins = dc + 2.0 * t_ins                      # diameter over insulation
    ratio = De_ins / dc

    # IEC 60287 thermal resistances [K·m/W]
    T1 = g('rho_ins') / (2.0 * math.pi) * math.log(ratio)
    u = 2.0 * g('L_depth') / De                    # depth/diameter ratio
    T4 = g('rho_soil') / (2.0 * math.pi) * math.log(u + math.sqrt(u * u - 1.0))

    # Van Wormer factor — splits insulation capacitance & dielectric loss
    # between the conductor and external nodes (IEC 60853 / Anders). Lies in
    # (0, 0.5]: →0.5 for thin insulation, →0 for thick.
    p = 1.0 / (2.0 * math.log(ratio)) - 1.0 / (ratio ** 2 - 1.0)

    # Thermal capacitances [J/(m·K)] = (cross-section area) × (volumetric cv)
    Q_cond   = math.pi / 4.0 * dc ** 2 * g('cv_cond')
    Q_ins    = math.pi / 4.0 * (De_ins ** 2 - dc ** 2) * g('cv_ins')
    Q_jacket = math.pi / 4.0 * (De ** 2 - De_ins ** 2) * g('cv_ins')
    soil_out = g('soil_shell') * De
    Q_soil   = math.pi / 4.0 * (soil_out ** 2 - De ** 2) * g('cv_soil')

    Q1 = Q_cond + p * Q_ins                        # conductor node
    Q2 = (1.0 - p) * Q_ins + Q_jacket + Q_soil     # external / soil node

    return {
        'T1': d.get('T1', T1), 'T4': d.get('T4', T4),
        'Q1': d.get('Q1', Q1), 'Q2': d.get('Q2', Q2),
        'p':  d.get('p', p),
    }


#: Keys whose presence selects the n-stage ladder form.
_LADDER_KEYS = ('R', 'C')

#: Keys that only exist in the two-node form, and so may not be combined
#: with an ``R``/``C`` ladder.
_TWO_NODE_KEYS = ('T1', 'T4', 'Q1', 'Q2', 'p')


def _resolve_ladder(dtr_data):
    """
    Pull an n-stage ladder out of the ``dtr`` data, or ``None`` for the
    two-node form.

    Returns ``(R, C)`` as lists of floats, cable outwards. Raises on anything
    ambiguous or unphysical rather than falling back quietly: an ignored
    parameter here means a cable silently running on the wrong thermal model,
    which shows up as a plausible-looking temperature rather than as an error.
    """
    R, C = dtr_data.get('R'), dtr_data.get('C')
    if R is None and C is None:
        return None
    if R is None or C is None:
        raise ValueError(
            "the n-stage cable ladder needs BOTH 'R' and 'C'; got only "
            f"{'R' if C is None else 'C'}")
    both = [k for k in _TWO_NODE_KEYS if k in dtr_data]
    if both:
        raise ValueError(
            f"a cable dtr block gives the ladder 'R'/'C' and also {both}. "
            "Those are alternative models — the ladder takes the TOTAL loss at "
            "node 1 while the two-node form splits W_c and lambda1*W_c across "
            "its two nodes — so pick one rather than have the other silently "
            "dropped.")
    try:
        R = [float(v) for v in R]
        C = [float(v) for v in C]
    except (TypeError, ValueError) as exc:
        raise ValueError("cable ladder 'R' and 'C' must be lists of "
                         f"numbers: {exc}") from exc
    if len(R) != len(C):
        raise ValueError(f"cable ladder 'R' has {len(R)} entries but 'C' has "
                         f"{len(C)}; they are stage-by-stage pairs")
    if not R:
        raise ValueError("cable ladder 'R'/'C' are empty")
    bad = [i + 1 for i, (r, c) in enumerate(zip(R, C)) if r <= 0.0 or c <= 0.0]
    if bad:
        raise ValueError(
            f"cable ladder stages {bad} have a non-positive R or C; a passive "
            "thermal ladder has strictly positive elements throughout "
            "(pydae.bps.lines.cable_fem checks this when it identifies one)")
    return R, C


def DTR_Cable_Symbolic(self, line_name, bus_j, bus_k, dtr_data):
    r"""
    Append the cable thermal model for one DTR cable line.

    Dispatches on the data: ``R``/``C`` lists build the n-stage Cauer ladder,
    anything else the IEC 60853 two-node form. Both define
    ``t_conductor_{line}``, which the network admittance in
    :func:`pydae.bps.lines.line_dtr.add_line_dtr` depends on.

    *dtr_data* is the already-resolved ``dtr`` data dict (see
    :func:`pydae.bps.lines.line_dtr._resolve_dtr_data`); ``model: "cable"`` has
    already routed us here.
    """
    ladder = _resolve_ladder(dtr_data)
    if ladder is not None:
        return _dtr_cable_ladder(self, line_name, bus_j, bus_k, dtr_data,
                                 *ladder)
    return _dtr_cable_two_node(self, line_name, bus_j, bus_k, dtr_data)


def _dtr_cable_common(self, line_name, bus_j, bus_k, dtr_data):
    """
    The pieces both forms share: the conductor current, the soil-temperature
    input and the loss parameters.

    Returns ``(i_cond, i_cond_sq, t_soil, syms)`` with *syms* holding
    ``lambda1``, ``Wd``, ``R_ac_20`` and ``alpha_R``. ``i_cond_sq`` is
    :math:`|i_{cond}|^2` formed *without* a square root and is what the loss
    terms must use; ``i_cond`` carries the root and is for reporting only.
    """
    backend = self.backend
    name = line_name

    # --- Network current (A) and conductor-current scaling ------------------
    # Inlined from the branch admittance and the bus voltages rather than read
    # from an algebraic variable: in amperes it is 3-5 orders above the
    # per-unit quantities beside it in y and wrecks cond(Gy). See
    # lines/branch_current.py. The losses need I**2, so the square root is
    # only ever formed for the reported current.
    idx_j = self.buses_list.index(bus_j)
    I_base = self.system['S_base']/(np.sqrt(3)*self.buses[idx_j]['U_kV']*1000)
    K_dtr = backend.symbols(f'K_dtr_{name}')
    i_j_k, _i_k_j = line_currents_pu(backend, bus_j, bus_k, name)
    i_cond_sq = current_magnitude_sq(*i_j_k, I_base*K_dtr)
    i_cond = current_magnitude(*i_j_k, I_base*K_dtr)

    # --- Ambient soil temperature (input) -----------------------------------
    t_soil = backend.symbols(f't_soil_{name}')

    syms = {
        'K_dtr': K_dtr,
        'lambda1': backend.symbols(f'lambda1_{name}'),
        'Wd': backend.symbols(f'Wd_{name}'),
        'R_ac_20': backend.symbols(f'R_ac_20_{name}'),
        'alpha_R': backend.symbols(f'alpha_R_{name}'),
    }
    return i_cond, i_cond_sq, t_soil, syms


def _dtr_cable_ladder(self, line_name, bus_j, bus_k, dtr_data, R, C):
    r"""
    Append an n-stage Cauer ladder for one DTR cable line.

    .. math::
        C_1\dot\theta_1 &= W_{\rm tot} - (\theta_1-\theta_2)/R_1 \\
        C_i\dot\theta_i &= (\theta_{i-1}-\theta_i)/R_{i-1}
                           - (\theta_i-\theta_{i+1})/R_i \\
        C_n\dot\theta_n &= (\theta_{n-1}-\theta_n)/R_{n-1}
                           - (\theta_n-\theta_{\rm soil})/R_n

    with :math:`W_{\rm tot} = (1+\lambda_1)W_c + W_d` injected at node 1 and
    :math:`\theta_1` the conductor. See the module docstring for why the loss
    is *not* re-split across the nodes here.
    """
    backend = self.backend
    name = line_name
    n = len(R)

    g = lambda k: dtr_data.get(k, _DEFAULTS[k])
    i_cond, i_cond_sq, t_soil, syms = _dtr_cable_common(
        self, name, bus_j, bus_k, dtr_data)

    # --- States: node 1 is the conductor, the rest are ladder nodes ---------
    # Node 1 must be called t_conductor_{name}: the temperature-dependent line
    # admittance is written against that symbol.
    theta = [backend.symbols(f't_conductor_{name}' if i == 0
                             else f't_ladder_{i + 1}_{name}')
             for i in range(n)]
    R_s = [backend.symbols(f'R{i + 1}_{name}') for i in range(n)]
    C_s = [backend.symbols(f'C{i + 1}_{name}') for i in range(n)]

    # --- Losses -------------------------------------------------------------
    R_ac = syms['R_ac_20'] * (1.0 + syms['alpha_R'] * (theta[0] - 20.0))
    W_c = i_cond_sq * R_ac                            # conductor loss [W/m]
    W_total = W_c * (1.0 + syms['lambda1']) + syms['Wd']

    # --- Heat balance -------------------------------------------------------
    # flow[i] leaves node i through R[i]; the last stage returns to the soil,
    # whose temperature is an input, and that grounding term is what makes the
    # steady state finite (sum(R) is the DC thermal resistance).
    flow = [(theta[i] - theta[i + 1]) / R_s[i] for i in range(n - 1)]
    flow.append((theta[n - 1] - t_soil) / R_s[n - 1])

    self.dae['f'] += [((W_total if i == 0 else flow[i - 1]) - flow[i]) / C_s[i]
                      for i in range(n)]
    self.dae['x'] += theta

    # --- Inputs (runtime settable: soil temperature) ------------------------
    self.dae['u_ini_dict'].update({str(t_soil): g('t_soil')})
    self.dae['u_run_dict'].update({str(t_soil): g('t_soil')})

    # --- Parameters ---------------------------------------------------------
    params = {str(s): float(v) for s, v in zip(R_s, R)}
    params.update({str(s): float(v) for s, v in zip(C_s, C)})
    params.update({
        str(syms['lambda1']): g('lambda1'), str(syms['Wd']): g('Wd'),
        str(syms['R_ac_20']): g('R_ac_20'), str(syms['alpha_R']): g('alpha_R'),
        str(syms['K_dtr']): g('K_dtr'),
    })
    self.dae['params_dict'].update(params)

    # --- Initial guesses ----------------------------------------------------
    # Shape the guess like the ladder itself: node i sits above the soil by the
    # resistance still downstream of it. A flat guess makes the Newton solve
    # start with heat flowing the wrong way through the outer stages.
    r_total = sum(R)
    rise = 20.0
    self.dae['xy_0_dict'].update({
        str(theta[i]): g('t_soil') + rise * sum(R[i:]) / r_total
        for i in range(n)
    })

    # --- Outputs ------------------------------------------------------------
    outputs = {
        f't_conductor_{name}': theta[0],
        f'W_c_{name}': W_c,                       # conductor loss [W/m]
        f'W_total_{name}': W_total,               # total heat input [W/m]
        f'R_ac_{name}': R_ac,                     # [Ω/m]
        f'I_cond_{name}': i_cond,                 # conductor current [A]
    }
    outputs.update({f't_ladder_{i + 1}_{name}': theta[i] for i in range(1, n)})
    self.dae['h_dict'].update(outputs)


def _dtr_cable_two_node(self, line_name, bus_j, bus_k, dtr_data):
    r"""
    Append the IEC 60853 two-node cable thermal model for one DTR cable line.
    """
    backend = self.backend
    name = line_name

    g = lambda k: dtr_data.get(k, _DEFAULTS[k])
    net = _cable_thermal_network(dtr_data)

    i_cond, i_cond_sq, t_soil, syms = _dtr_cable_common(
        self, name, bus_j, bus_k, dtr_data)
    lambda1, Wd = syms['lambda1'], syms['Wd']
    R_ac_20, alpha_R, K_dtr = syms['R_ac_20'], syms['alpha_R'], syms['K_dtr']

    # --- States: conductor and external/sheath temperatures -----------------
    t_cond   = backend.symbols(f't_conductor_{name}')
    t_sheath = backend.symbols(f't_sheath_{name}')

    # --- Lumped thermal-network parameters ----------------------------------
    T1 = backend.symbols(f'T1_{name}')
    T4 = backend.symbols(f'T4_{name}')
    Q1 = backend.symbols(f'Q1_{name}')
    Q2 = backend.symbols(f'Q2_{name}')
    p  = backend.symbols(f'p_vw_{name}')

    # --- Losses -------------------------------------------------------------
    R_ac = R_ac_20 * (1.0 + alpha_R * (t_cond - 20.0))   # [Ω/m]
    W_c = i_cond_sq * R_ac                              # conductor loss [W/m]

    # --- Two-node heat balance ----------------------------------------------
    q_cond_node = W_c + p * Wd
    q_ext_node  = (t_cond - t_sheath) / T1 + (1.0 - p) * Wd + lambda1 * W_c

    dt_cond   = (q_cond_node - (t_cond - t_sheath) / T1) / Q1
    dt_sheath = (q_ext_node - (t_sheath - t_soil) / T4) / Q2

    self.dae['f'] += [dt_cond, dt_sheath]
    self.dae['x'] += [t_cond, t_sheath]

    # --- Inputs (runtime settable: soil temperature) ------------------------
    self.dae['u_ini_dict'].update({str(t_soil): g('t_soil')})
    self.dae['u_run_dict'].update({str(t_soil): g('t_soil')})

    # --- Parameters (numeric defaults from geometry / materials) ------------
    self.dae['params_dict'].update({
        str(T1): net['T1'], str(T4): net['T4'],
        str(Q1): net['Q1'], str(Q2): net['Q2'], str(p): net['p'],
        str(lambda1): g('lambda1'), str(Wd): g('Wd'),
        str(R_ac_20): g('R_ac_20'), str(alpha_R): g('alpha_R'),
        str(K_dtr): g('K_dtr'),
    })

    # --- Initial guesses ----------------------------------------------------
    self.dae['xy_0_dict'].update({
        str(t_cond): g('t_soil') + 20.0,
        str(t_sheath): g('t_soil') + 10.0,
    })

    # --- Outputs ------------------------------------------------------------
    self.dae['h_dict'].update({
        f't_conductor_{name}': t_cond,
        f't_sheath_{name}': t_sheath,
        f'W_c_{name}': W_c,                       # conductor loss [W/m]
        f'W_total_{name}': W_c * (1.0 + lambda1) + Wd,  # total heat input [W/m]
        f'R_ac_{name}': R_ac,                     # [Ω/m]
        f'I_cond_{name}': i_cond,                 # conductor current [A]
    })


def test():
    """Minimal build + ini + dynamic run for a single IEC 60853 cable line."""
    from pydae.bps import BpsBuilder
    from pydae.core.builder.casadi_builder import CasadiBuilder
    from pydae.core.model.casadi_model import CasadiModel

    data = {
        "system": {"name": "cable_dtr_test", "S_base": 25e6, "K_xif": 0.01},
        "buses": [{"name": "1", "P_W": 0.0, "Q_var": 0.0, "U_kV": 20.0},
                  {"name": "2", "P_W": 10e6, "Q_var": 2e6, "U_kV": 20.0}],
        "lines": [{"bus_j": "1", "bus_k": "2", "R_km": 0.253, "X_km": 0.12,
                   "Bs_km": 4.7e-5, "km": 2.0, "monitor": True,
                   "dtr": {"model": "cable"}}],
        "sources": [{"type": "genape", "bus": "1", "S_n": 1e9, "F_n": 50.0,
                     "X_v": 0.001, "R_v": 0.0, "K_delta": 0.001, "K_alpha": 1e-6}],
    }
    grid = BpsBuilder(data, use_casadi=True)
    grid.construct('cable_dtr_test')
    model = CasadiModel(CasadiBuilder(grid.sys_dict).build())
    model.ini({}, newton_tol=1e-10)

    t_c = float(model.get_value('t_conductor_1_2'))
    t_s = float(model.get_value('t_sheath_1_2'))
    t_soil = float(model.get_value('t_soil_1_2'))
    W_c = float(model.get_value('W_c_1_2'))
    T1 = float(model.get_value('T1_1_2'))
    T4 = float(model.get_value('T4_1_2'))
    p = float(model.get_value('p_vw_1_2'))
    print(f"t_conductor = {t_c:.2f} C, t_sheath = {t_s:.2f} C, t_soil = {t_soil:.2f} C")

    # Van Wormer factor in physical range
    assert 0.0 < p <= 0.5, f"Van Wormer factor out of range: {p}"
    # Steady-state ladder: theta_c = theta_soil + W_c*(T1+T4), theta_e = +W_c*T4
    assert abs(t_c - (t_soil + W_c * (T1 + T4))) < 1e-3, "conductor SS mismatch"
    assert abs(t_s - (t_soil + W_c * T4)) < 1e-3, "sheath SS mismatch"
    assert t_c > t_s > t_soil, "temperatures not monotonic outward"

    # Dynamic: a soil-temperature step must raise (slowly) the conductor temp
    model.Dt = 10.0
    model.run(3600.0, {'t_soil_1_2': 25.0})
    model.post()
    assert float(model.get_value('t_conductor_1_2')) > t_c, "no response to soil step"
    print("cable DTR test passed")


def test_ladder():
    """Build + ini + dynamic run for an n-stage ladder cable line."""
    from pydae.bps import BpsBuilder
    from pydae.core.builder.casadi_builder import CasadiBuilder
    from pydae.core.model.casadi_model import CasadiModel

    # A five-stage ladder identified from the CIGRE MV benchmark by
    # pydae.bps.lines.cable_fem (see its ladder_dtr()).
    R = [0.06962, 0.4661, 0.40922, 0.56014, 0.0987]
    C = [530.14, 962.03, 86095.2, 937163.5, 34453437.0]
    dtr = {"model": "cable", "R": R, "C": C, "lambda1": 0.4822,
           "R_ac_20": 2.6406e-4, "alpha_R": 0.004, "Wd": 0.0, "K_dtr": 1.0,
           "t_soil": 18.94}

    data = {
        "system": {"name": "cable_ladder_test", "S_base": 25e6, "K_xif": 0.01},
        "buses": [{"name": "1", "P_W": 0.0, "Q_var": 0.0, "U_kV": 20.0},
                  {"name": "2", "P_W": 6.7e6, "Q_var": 1.7e6, "U_kV": 20.0}],
        "lines": [{"bus_j": "1", "bus_k": "2", "R_km": 0.501, "X_km": 0.12,
                   "Bs_km": 4.7e-5, "km": 2.0, "monitor": True, "dtr": dtr}],
        "sources": [{"type": "genape", "bus": "1", "S_n": 1e9, "F_n": 50.0,
                     "X_v": 0.001, "R_v": 0.0, "K_delta": 0.001,
                     "K_alpha": 1e-6}],
    }
    grid = BpsBuilder(data, use_casadi=True)
    grid.construct('cable_ladder_test')
    model = CasadiModel(CasadiBuilder(grid.sys_dict).build())
    model.ini({}, newton_tol=1e-10)

    n = len(R)
    g = lambda k: float(model.get_value(f'{k}_1_2'))
    nodes = [g('t_conductor')] + [g(f't_ladder_{i}') for i in range(2, n + 1)]
    t_soil, w_total = g('t_soil'), g('W_total')
    print("ladder nodes [degC]: " + ", ".join(f"{v:.2f}" for v in nodes)
          + f"   t_soil {t_soil:.2f}, W_total {w_total:.2f} W/m")

    # Steady state: every node sits above the soil by the resistance still
    # downstream of it, driven by the TOTAL loss. sum(R) is therefore the DC
    # conductor-to-soil thermal resistance.
    for i, theta in enumerate(nodes):
        expect = t_soil + w_total * sum(R[i:])
        assert abs(theta - expect) < 1e-6, \
            f"node {i + 1}: {theta:.6f} vs {expect:.6f} degC"
    # Monotonically cooler outward, and hotter than the soil throughout.
    assert all(a > b for a, b in zip(nodes, nodes[1:])), "not monotonic outward"
    assert nodes[-1] > t_soil

    # Losses consistent with the conductor temperature.
    assert abs(g('W_total') - g('W_c') * (1.0 + 0.4822)) < 1e-9
    assert abs(g('R_ac') - 2.6406e-4 * (1.0 + 0.004 * (nodes[0] - 20.0))) < 1e-15

    # A soil-temperature step must propagate to the conductor. The outer
    # stages have time constants of days, so 6 h moves it only a little --
    # which is the entire point of having them.
    model.Dt = 1800.0
    model.run(6 * 3600.0, {'t_soil_1_2': t_soil + 10.0})
    model.post()
    t_after = float(model.get_value('t_conductor_1_2'))
    assert nodes[0] < t_after < nodes[0] + 10.0, \
        f"conductor went from {nodes[0]:.2f} to {t_after:.2f} degC in 6 h"

    # R/C and T1/T4 are alternative models, so mixing them must raise rather
    # than silently honouring one.
    for clash in ({'T1': 0.37}, {'Q2': 1e5}, {'p': 0.4}):
        try:
            _resolve_ladder({'R': R, 'C': C, **clash})
        except ValueError:
            pass
        else:                                          # pragma: no cover
            raise AssertionError(f"mixing R/C with {clash} was accepted")
    # ... and so must a malformed ladder.
    for bad in ({'R': R}, {'C': C}, {'R': R, 'C': C[:-1]},
                {'R': [0.1, -0.2], 'C': [1.0, 2.0]}, {'R': [], 'C': []}):
        try:
            _resolve_ladder(bad)
        except ValueError:
            pass
        else:                                          # pragma: no cover
            raise AssertionError(f"malformed ladder accepted: {bad}")
    assert _resolve_ladder({'T1': 0.37, 'T4': 1.3}) is None, \
        "a two-node block must not be read as a ladder"

    print("cable ladder DTR test passed")


if __name__ == "__main__":
    test()
    test_ladder()
