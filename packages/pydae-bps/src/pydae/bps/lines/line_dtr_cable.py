# -*- coding: utf-8 -*-
"""
Dynamic Thermal Rating for **underground cables** — IEC 60853 two-node model.

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
    Single source of truth for the underground-cable (IEC 60853 two-node) DTR
    parameters, inputs, states and outputs. Consumed by the docs autosummary
    generator.

    Geometry/material values are supplied per line through the ``dtr`` field
    (inline dict, ``dtr_codes`` reference, or reference with overrides) whose
    ``model: "cable"`` routes here; the ``data`` column is the corresponding key.
    The lumped network coefficients T1/T4/Q1/Q2/p are derived from the geometry
    (see :func:`_cable_thermal_network`) but may also be supplied directly.
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

    # Input
    d += [{"type": "Input", "tex": "T_{soil}", "data": "t_soil", "model": "t_soil", "default": 15.0, "description": "Ambient soil temperature", "units": "°C"}]

    # Dynamic states
    d += [{"type": "Dynamic State", "tex": "T_{cond}",  "data": "", "model": "t_conductor", "default": "", "description": "Conductor temperature", "units": "°C"}]
    d += [{"type": "Dynamic State", "tex": "T_{sheath}","data": "", "model": "t_sheath",    "default": "", "description": "External/sheath-region temperature", "units": "°C"}]

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


def DTR_Cable_Symbolic(self, line_name, bus_j, bus_k, dtr_data):
    r"""
    Append the IEC 60853 two-node cable thermal model for one DTR cable line.

    *dtr_data* is the already-resolved ``dtr`` data dict (see
    :func:`pydae.bps.lines.line_dtr._resolve_dtr_data`); ``model: "cable"`` has
    already routed us here.
    """
    backend = self.backend
    name = line_name

    g = lambda k: dtr_data.get(k, _DEFAULTS[k])
    net = _cable_thermal_network(dtr_data)

    # --- States: conductor and external/sheath temperatures -----------------
    t_cond   = backend.symbols(f't_conductor_{name}')
    t_sheath = backend.symbols(f't_sheath_{name}')

    # --- Network current (A) and conductor-current scaling ------------------
    I_j_k, _I_k_j = backend.symbols(f"I_{bus_j}_{bus_k}, I_{bus_k}_{bus_j}")
    K_dtr = backend.symbols(f'K_dtr_{name}')
    i_cond = I_j_k * K_dtr

    # --- Ambient soil temperature (input) -----------------------------------
    t_soil = backend.symbols(f't_soil_{name}')

    # --- Lumped thermal-network parameters ----------------------------------
    T1 = backend.symbols(f'T1_{name}')
    T4 = backend.symbols(f'T4_{name}')
    Q1 = backend.symbols(f'Q1_{name}')
    Q2 = backend.symbols(f'Q2_{name}')
    p  = backend.symbols(f'p_vw_{name}')
    lambda1 = backend.symbols(f'lambda1_{name}')
    Wd = backend.symbols(f'Wd_{name}')
    R_ac_20 = backend.symbols(f'R_ac_20_{name}')
    alpha_R = backend.symbols(f'alpha_R_{name}')

    # --- Losses -------------------------------------------------------------
    R_ac = R_ac_20 * (1.0 + alpha_R * (t_cond - 20.0))   # [Ω/m]
    W_c = i_cond ** 2 * R_ac                              # conductor loss [W/m]

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


if __name__ == "__main__":
    test()
