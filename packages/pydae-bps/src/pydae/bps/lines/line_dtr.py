# -*- coding: utf-8 -*-
r"""
Dynamic Thermal Rating (DTR) lines for pydae-bps.

A DTR line augments the ordinary :math:`\pi`-section with a *thermal* model of
the conductor, so its temperature becomes a state of the DAE. Two thermal
models are available, dispatched on the resolved ``dtr`` data:

- **overhead** (default) — IEEE-738 bare-conductor heat balance in air
  (:func:`DTR_Symbolic`, this module). Single conductor node cooled by forced
  + natural convection and radiation, heated by Joule and solar gain.
- **cable** — IEC 60853 two-node buried-cable ladder
  (:func:`pydae.bps.lines.line_dtr_cable.DTR_Cable_Symbolic`), selected with
  ``dtr: {model: "cable", ...}``.

**Closed-loop coupling.** The series admittance ``g_{line}``/``b_{line}`` is
*not* a constant parameter: it is an algebraic variable evaluated from the
temperature-dependent resistance

.. math::  R(T) = R_{20}\,\bigl(1 + \alpha_R (T - 20)\bigr),

so the conductor temperature feeds back into the network resistance, losses and
voltage drop (see :func:`add_line_dtr`). The same ``R_20`` (from the line's
``R_km``) and single coefficient ``alpha_R`` drive both the network admittance
and the conductor heat balance — one resistance per line.

**Data interface.** Cable/ambient parameters are resolved per line by
:func:`_resolve_dtr_data` from the ``dtr`` field, which may be:

- ``true``                       — built-in defaults;
- an inline ``{...}`` dict       — those values;
- ``"code_name"``                — copy a named entry from the top-level
                                   ``dtr_codes`` section;
- ``{code: "code_name", ...}``   — that named code with per-line overrides.

This lets a cable type be defined once in ``dtr_codes`` and reused by many
lines. Results are summarised by :mod:`pydae.bps.utils.dtr_report`.

Originally created 2022-08-10 (@jmmauricio).
"""

import numpy as np

from pydae.bps.lines.branch_current import (line_currents_pu,
                                            current_magnitude_sq)


def descriptions():
    """
    Single source of truth for the overhead (IEEE-738) DTR parameters, inputs,
    states and outputs. Consumed by the docs autosummary generator.

    Cable/ambient values are supplied per line through the ``dtr`` field (an
    inline dict, a ``dtr_codes`` reference, or a reference with overrides); the
    ``data`` column below is the corresponding key.
    """
    d = []

    # Parameters (cable geometry / material; per-line suffix _{bus_j}_{bus_k})
    d += [{"type": "Parameter", "tex": "D_{cond}",  "data": "D_conductor", "model": "D_conductor", "default": 0.0277, "description": "Conductor overall diameter", "units": "m"}]
    d += [{"type": "Parameter", "tex": "D_{core}",  "data": "D_core",      "model": "D_core",      "default": 0.00308, "description": "Steel-core diameter (0 if homogeneous)", "units": "m"}]
    d += [{"type": "Parameter", "tex": "\\epsilon", "data": "Emissivity",  "model": "Emissivity",  "default": 0.8,   "description": "Surface emissivity (radiation)", "units": "-"}]
    d += [{"type": "Parameter", "tex": "\\alpha_s", "data": "Absorptivity","model": "Absorptivity","default": 0.8,   "description": "Solar absorptivity", "units": "-"}]
    d += [{"type": "Parameter", "tex": "H_{elev}",  "data": "Elevation",   "model": "Elevation",   "default": 0.0,   "description": "Line elevation above sea level", "units": "m"}]
    d += [{"type": "Parameter", "tex": "m",         "data": "Mass_linear", "model": "Mass_linear", "default": 1.62,  "description": "Conductor mass per unit length", "units": "kg/m"}]
    d += [{"type": "Parameter", "tex": "C_p",       "data": "Cp",          "model": "Cp",          "default": 900.0, "description": "Conductor specific heat", "units": "J/(kg·K)"}]
    d += [{"type": "Parameter", "tex": "K_{dtr}",   "data": "K_dtr",       "model": "K_dtr",       "default": 0.0,   "description": "Network-current to conductor-current scaling", "units": "-"}]
    d += [{"type": "Parameter", "tex": "R_{ac,20}", "data": "R_ac_20",     "model": "R_ac_20",     "default": 5.0e-4, "description": "AC resistance at 20 °C (defaults to R_km/1000)", "units": "Ω/m"}]
    d += [{"type": "Parameter", "tex": "\\alpha_R", "data": "alpha_R",     "model": "alpha_R",     "default": 0.004, "description": "Resistance temperature coefficient", "units": "1/K"}]

    # Inputs (weather; runtime-settable)
    d += [{"type": "Input", "tex": "T_{amb}",  "data": "t_ambient",  "model": "t_ambient",  "default": 25.0,   "description": "Ambient air temperature", "units": "°C"}]
    d += [{"type": "Input", "tex": "v_{wind}", "data": "v_wind",     "model": "v_wind",     "default": 1.0,    "description": "Wind speed", "units": "m/s"}]
    d += [{"type": "Input", "tex": "\\phi",    "data": "angle_wind", "model": "angle_wind", "default": 90.0,   "description": "Wind attack angle to the line", "units": "deg"}]
    d += [{"type": "Input", "tex": "Q_s",      "data": "irradiance", "model": "irradiance", "default": 1000.0, "description": "Solar irradiance", "units": "W/m²"}]

    # Dynamic state
    d += [{"type": "Dynamic State", "tex": "T_{cond}", "data": "", "model": "t_conductor", "default": "", "description": "Conductor temperature", "units": "°C"}]

    # Algebraic states (intermediate thermal quantities + closed-loop admittance)
    d += [{"type": "Algebraic State", "tex": "\\rho_{air}", "data": "", "model": "dens",     "default": "", "description": "Air density", "units": "kg/m³"}]
    d += [{"type": "Algebraic State", "tex": "Re",          "data": "", "model": "reynolds", "default": "", "description": "Reynolds number", "units": "-"}]
    d += [{"type": "Algebraic State", "tex": "Ra",          "data": "", "model": "rayleigh", "default": "", "description": "Rayleigh number", "units": "-"}]
    d += [{"type": "Algebraic State", "tex": "Nu",          "data": "", "model": "nu",       "default": "", "description": "Nusselt number", "units": "-"}]
    d += [{"type": "Algebraic State", "tex": "q_s",         "data": "", "model": "p_s",      "default": "", "description": "Solar heat gain", "units": "W/m"}]
    d += [{"type": "Algebraic State", "tex": "q_c",         "data": "", "model": "p_c",      "default": "", "description": "Convective heat loss", "units": "W/m"}]
    d += [{"type": "Algebraic State", "tex": "g",           "data": "", "model": "g",        "default": "", "description": "Series conductance G(T) (temperature-coupled)", "units": "pu"}]
    d += [{"type": "Algebraic State", "tex": "b",           "data": "", "model": "b",        "default": "", "description": "Series susceptance B(T) (temperature-coupled)", "units": "pu"}]

    # Outputs
    d += [{"type": "Output", "tex": "R_{ac}", "data": "", "model": "R_ac", "default": "", "description": "AC resistance at conductor temperature", "units": "Ω/m"}]
    d += [{"type": "Output", "tex": "q_j",    "data": "", "model": "p_j",  "default": "", "description": "Joule heat gain", "units": "W/m"}]
    return d


def _resolve_dtr_data(grid, line):
    r"""
    Resolve the per-line DTR cable/ambient data dict for one ``lines[]`` entry.

    The line's ``dtr`` field may be:

    - ``True``                       — use the built-in defaults (empty dict).
    - ``{...}``  (inline)            — those keys are the data.
    - ``"code_name"`` (string)       — copy a named entry from the top-level
                                       ``dtr_codes`` section.
    - ``{code: "code_name", ...}``   — start from that named code, then apply
                                       the remaining keys as per-line overrides
                                       (so several lines can share one code and
                                       only spell out their differences).

    Named codes live in a top-level ``dtr_codes`` list, each a dict with a
    ``name`` plus any of the DTR parameters::

        dtr_codes: [
          {name: "ohl_A1", D_conductor: 0.0102, Cp: 900.0, K_dtr: 1.0, ...}
        ]
        lines: [
          {bus_j: "B12", bus_k: "B13", ..., dtr: "ohl_A1"},
          {bus_j: "B13", bus_k: "B14", ..., dtr: {code: "ohl_A1", v_wind: 2.0}}
        ]

    Returns a plain dict of DTR parameters (never includes the ``name``/``code``
    bookkeeping keys).
    """
    raw = line.get('dtr')
    if raw is None or raw is True or raw is False:
        return {}

    codes = {c['name']: c for c in grid.data.get('dtr_codes', [])}

    def _code(code_name):
        if code_name not in codes:
            raise KeyError(
                f"line {line.get('bus_j')}-{line.get('bus_k')} references dtr code "
                f"'{code_name}' which is not defined in the 'dtr_codes' section "
                f"(available: {sorted(codes)})")
        return {k: v for k, v in codes[code_name].items() if k != 'name'}

    if isinstance(raw, str):
        return _code(raw)

    if isinstance(raw, dict):
        code_name = raw.get('code') or raw.get('dtr_code')
        base = _code(code_name) if code_name else {}
        overrides = {k: v for k, v in raw.items() if k not in ('code', 'dtr_code')}
        return {**base, **overrides}

    return {}


def DTR_Symbolic(self, line_name, bus_j, bus_k, dtr_data):
    """
    Symbolic implementation of the IEEE 738 Dynamic Thermal Rating model
    (overhead conductor in air). Designed for integration with PyDAE or other
    symbolic solvers.

    *dtr_data* is the already-resolved cable/ambient data dict (see
    :func:`_resolve_dtr_data`). When a key is omitted the generic ACSR defaults
    below are used, preserving backward compatibility.
    """
    backend = self.backend
    name = line_name

    # --- 1. Define State Variables and Inputs (Symbols) ---
    # Algebraic State (Unknown to solve for) - Lowercase
    self.t_conductor = backend.symbols(f't_conductor_{name}')

    # Inputs (Control or Disturbance) - Lowercase
    self.K_dtr = backend.symbols(f'K_dtr_{name}')

    # Network current, inlined from the branch admittance and the bus voltages
    # instead of carried as an algebraic variable. In amperes it is 3-5 orders
    # above the per-unit quantities beside it in y and wrecks cond(Gy); see
    # lines/branch_current.py. Nothing is lost by inlining: the current is a
    # function of the bus voltages and the branch primitive, and g_/b_ resolve
    # to this line's own (temperature-dependent) algebraic variables, so the
    # current still reflects R(T).
    #
    # The heat balance only ever needs the SQUARE of the current, so the
    # magnitude's square root -- and its 0/0 derivative at exactly zero flow --
    # never has to be formed at all.
    idx_j = self.buses_list.index(bus_j)
    I_base = self.system['S_base']/(np.sqrt(3)*self.buses[idx_j]['U_kV']*1000)
    i_j_k, _i_k_j = line_currents_pu(backend, bus_j, bus_k, name)
    self.i_line_sq = current_magnitude_sq(*i_j_k, I_base*self.K_dtr)
    self.t_ambient = backend.symbols(f't_ambient_{name}')
    self.v_wind = backend.symbols(f'v_wind_{name}')
    self.angle_wind = backend.symbols(f'angle_wind_{name}')
    self.irradiance = backend.symbols(f'irradiance_{name}')

    # Physical Parameters (Fixed constants) - Initial Capital
    self.D_conductor = backend.symbols(f'D_conductor_{name}')
    self.D_core = backend.symbols(f'D_core_{name}')
    self.Emissivity = backend.symbols(f'Emissivity_{name}')
    self.Absorptivity = backend.symbols(f'Absorptivity_{name}')
    self.Elevation = backend.symbols(f'Elevation_{name}')
    self.Mass_linear = backend.symbols(f'Mass_linear_{name}') # [kg/m]
    self.Cp = backend.symbols(f'Cp_{name}')                   # [J/(kg*C)]

    # Constants - Initial Capital
    self.Sigma = 5.670373e-8  # Stefan-Boltzmann constant
    self.Gravity = 9.81             # Gravity

    # pydae
    # self.f_list = []
    # self.x_list = []
    # self.dae['g'] = []
    # self.dae['y_ini'] = []
    # self.u_dict = {}


    # Initialize Default Values
    # Inputs
    # self.dae['u_ini_dict'].update({str(self.i_line): 0.0})
    # self.dae['u_run_dict'].update({str(self.i_line): 0.0})


    # Ambient conditions (inputs) — defaults overridable via the dtr dict
    t_ambient_0 = dtr_data.get('t_ambient', 25.0)
    v_wind_0    = dtr_data.get('v_wind', 1.0)
    angle_wind_0 = dtr_data.get('angle_wind', 90.0)
    irradiance_0 = dtr_data.get('irradiance', 1000.0)

    self.dae['u_ini_dict'].update({str(self.t_ambient): t_ambient_0})
    self.dae['u_ini_dict'].update({str(self.v_wind): v_wind_0})
    self.dae['u_ini_dict'].update({str(self.angle_wind): angle_wind_0})
    self.dae['u_ini_dict'].update({str(self.irradiance): irradiance_0})

    self.dae['u_run_dict'].update({str(self.t_ambient): t_ambient_0})
    self.dae['u_run_dict'].update({str(self.v_wind): v_wind_0})
    self.dae['u_run_dict'].update({str(self.angle_wind): angle_wind_0})
    self.dae['u_run_dict'].update({str(self.irradiance): irradiance_0})

    # Cable parameters — generic ACSR defaults overridable via the dtr dict
    self.dae['params_dict'].update({str(self.D_conductor): dtr_data.get('D_conductor', 0.0277)})
    self.dae['params_dict'].update({str(self.D_core): dtr_data.get('D_core', 0.00308)})
    self.dae['params_dict'].update({str(self.Emissivity): dtr_data.get('Emissivity', 0.8)})
    self.dae['params_dict'].update({str(self.Absorptivity): dtr_data.get('Absorptivity', 0.8)})
    self.dae['params_dict'].update({str(self.Elevation): dtr_data.get('Elevation', 0.0)})
    # Default Mass (Approx for ACSR 20mm) and Cp (Aluminum/Steel mix)
    self.dae['params_dict'].update({str(self.Mass_linear): dtr_data.get('Mass_linear', 1.62)}) # kg/m
    self.dae['params_dict'].update({str(self.Cp): dtr_data.get('Cp', 900.0)})                   # J/kg-C

    # K_dtr scales the network line-current variable to physical conductor
    # current; default 0.0 means NO Joule heating — set it for a real study.
    self.dae['params_dict'].update({str(self.K_dtr): dtr_data.get('K_dtr', 0.0)})

    """
    Constructs the symbolic heat balance equations.
    Returns:
        g_list: List of algebraic equations (g(y,u) = 0)
        y_list: List of algebraic variables
    """

    y_list_considered = ['dens','reynolds','rayleigh','nu','p_s','p_c'] #  t_conductor is always considered in y or in x, it doesn't work
    #y_list_considered = [] #  t_conductor is always considered in y, it works
    #y_list_considered = ['p_s','p_c', 'p_j'] #  t_conductor is always considered in y, it works
    # y_list_considered = ['rayleigh','p_s','p_c', 'p_j'] #  t_conductor is always considered in y, it works
    # y_list_considered = ['reynolds','rayleigh','p_s','p_c', 'p_j'] #  t_conductor is always considered in y, it works
    # y_list_considered = ['reynolds','rayleigh','nu','p_s','p_c', 'p_j','R_ac'] #  t_conductor is always considered in y, it works
    # y_list_considered = ['dens','reynolds','rayleigh','nu','p_s','p_c', 'p_j','R_ac'] #  t_conductor is always considered in y, it doesn't work
    #y_list_considered = ['reynolds','rayleigh'] #  t_conductor is always considered in y or in x, it doesn't work

    # --- A. Film Temperature ---
    t_film = 0.5 * (self.t_ambient + self.t_conductor)

    # --- B. Air Properties ---
    # Thermal conductivity (lmb)
    lmb = 2.368e-2 + 7.23e-5*t_film - 2.763e-8*t_film**2

    # Air Density (dens)
    dens_num = 1.293 - 1.525e-4*self.Elevation + 6.379e-9*self.Elevation**2
    self.dens = dens_num / (1 + 0.00367*t_film)

    # Intermediate variable for PyDAE (lowercase)

    if 'dens' in y_list_considered:
        dens = backend.symbols(f'dens_{line_name}')
        self.dae['g'] += [self.dens - dens]
        self.dae['y_ini'] += [dens]
        self.dae['y_run'] += [dens]
    else:
        dens = self.dens


    # Dynamic Viscosity (mu_f)
    mu_f = (17.239 + 4.635e-2*t_film - 2.03e-5*t_film**2) * 1e-6

    # Kinematic Viscosity (uf)
    uf = mu_f / dens

    # --- C. Convection (p_c) Calculation ---
    # Reynolds Number
    self.reynolds = self.v_wind * self.D_conductor / uf
    if 'reynolds' in y_list_considered:
        reynolds = backend.symbols(f'reynolds_{line_name}')
        self.dae['g'] += [self.reynolds - reynolds]
        self.dae['y_ini'] += [reynolds]
        self.dae['y_run'] += [reynolds]
    else:
        reynolds = self.reynolds

    # Roughness factor
    Rs = self.D_core / (2 * (self.D_conductor - self.D_core))

    # Forced-convection Nusselt number  Nu_90 = B · Re^n  (wind perpendicular to
    # the line). B and n are the IEEE-738 coefficient/exponent tables, selected
    # by the flow regime (Reynolds threshold 2650) and the conductor surface
    # roughness Rs: rough high-Re, smooth high-Re, or low-Re laminar.
    cond_high_smooth = backend.And(reynolds > 2650, Rs <= 0.05)
    cond_high_rough  = reynolds > 2650

    B = backend.Piecewise(
        (0.178, cond_high_smooth),
        (0.048, cond_high_rough),
        (0.641, True)
    )

    n = backend.Piecewise(
        (0.633, cond_high_smooth),
        (0.800, cond_high_rough),
        (0.471, True)
    )

    nu_90 = B * reynolds**n

    # Wind-direction factor: convection weakens as the wind turns parallel to
    # the conductor. delta(phi) scales Nu_90 down toward the grazing-flow value
    # (IEEE-738 angle correction; seno = |sin(attack angle)|).
    phi_rad = self.angle_wind/180*np.pi
    seno = backend.Abs(backend.sin(phi_rad))
    eff_angle = backend.Mod(self.angle_wind, 180)

    delta = backend.Piecewise(
        (0.42 + 0.68 * seno**1.08, eff_angle <= 24),
        (0.42 + 0.58 * seno**0.9, True)
    )

    nu_forz = nu_90 * delta   # forced-convection Nusselt at the actual wind angle

    # Natural (buoyant) convection, used when wind is weak. Grashof number Gr
    # measures buoyancy vs viscosity; Pr_num is the Prandtl number (air property
    # polynomial in film temperature). Rayleigh = Gr·Pr drives the free-convection
    # Nusselt below.
    Gr = (self.D_conductor**3 * (self.t_conductor - self.t_ambient) * self.Gravity) / ((t_film + 273.15) * uf**2)
    Pr_num = (1.9327e-10*t_film**4 - 7.9999e-7*t_film**3 + 1.1407e-3*t_film**2 - 0.4489*t_film + 1057.5) * mu_f / lmb

    self.rayleigh = Gr * Pr_num
    if 'rayleigh' in y_list_considered:
        rayleigh = backend.symbols(f'rayleigh_{line_name}')
        self.dae['g'] += [self.rayleigh - rayleigh]
        self.dae['y_ini'] += [rayleigh]
        self.dae['y_run'] += [rayleigh]
    else:
        rayleigh = self.rayleigh

    # Free-convection Nusselt  Nu_nat = A · Ra^m, with the IEEE-738 coefficient
    # A and exponent m tabulated by Rayleigh-number decade.
    A_nat = backend.Piecewise(
        (1.02,  backend.And(rayleigh > 0.1, rayleigh < 1e2)),
        (0.85,  backend.And(rayleigh >= 1e2, rayleigh < 1e4)),
        (0.48,  backend.And(rayleigh >= 1e4, rayleigh < 1e7)),
        (0.125, backend.And(rayleigh >= 1e7, rayleigh < 1e12)),
        (0.0, True)
    )

    m_nat = backend.Piecewise(
        (0.148, backend.And(rayleigh > 0.1, rayleigh < 1e2)),
        (0.188, backend.And(rayleigh >= 1e2, rayleigh < 1e4)),
        (0.25,  backend.And(rayleigh >= 1e4, rayleigh < 1e7)),
        (0.333, backend.And(rayleigh >= 1e7, rayleigh < 1e12)),
        (0.0, True)
    )

    nu_nat = A_nat * rayleigh**m_nat

    # Effective Nusselt = whichever of forced (wind) or natural (buoyant)
    # convection is the stronger cooling mechanism at this operating point.
    self.nu = backend.Max(nu_nat, nu_forz)
    if 'nu' in y_list_considered:
        nu = backend.symbols(f'nu_{line_name}')
        self.dae['g'] += [self.nu - nu]
        self.dae['y_ini'] += [nu]
        self.dae['y_run'] += [nu]
    else:
        nu = self.nu

    # Heat-flux terms per unit length [W/m].
    # Convective cooling: q_c = pi · lambda_air · (T_cond − T_amb) · Nu.
    self.p_c = np.pi * lmb * (self.t_conductor - self.t_ambient) * nu
    if 'p_c' in y_list_considered:
        p_c = backend.symbols(f'p_c_{line_name}')
        self.dae['g'] += [self.p_c - p_c]
        self.dae['y_ini'] += [p_c]
        self.dae['y_run'] += [p_c]
    else:
        p_c = self.p_c

    # Radiative cooling (Stefan-Boltzmann): q_r = pi · D · sigma · eps · (T_c^4 − T_a^4).
    self.p_r = np.pi * self.D_conductor * self.Sigma * self.Emissivity * ((self.t_conductor + 273.15)**4 - (self.t_ambient + 273.15)**4)
    if 'p_r' in y_list_considered:
        p_r = backend.symbols(f'p_r_{line_name}')
        self.dae['g'] += [self.p_r - p_r]
        self.dae['y_ini'] += [p_r]
        self.dae['y_run'] += [p_r]
    else:
        p_r = self.p_r

    # Solar heating: q_s = alpha_solar · Q_solar · D  (absorbed irradiance).
    self.p_s = self.Absorptivity * self.irradiance * self.D_conductor
    if 'p_s' in y_list_considered:
        p_s = backend.symbols(f'p_s_{line_name}')
        self.dae['g'] += [self.p_s - p_s]
        self.dae['y_ini'] += [p_s]
        self.dae['y_run'] += [p_s]
    else:
        p_s = self.p_s

    # Joule Heating. R_ac(T) shares the line's resistance (R_ac_20 [Ω/m]) and a
    # single temperature coefficient (alpha_R) with the network admittance, so
    # both rise together as the conductor heats — see add_line_dtr (closed loop).
    R_ac_20 = backend.symbols(f'R_ac_20_{name}')
    alpha_R = backend.symbols(f'alpha_R_{name}')
    self.dae['params_dict'][str(R_ac_20)] = dtr_data.get('R_ac_20', 5.0e-4)
    self.dae['params_dict'][str(alpha_R)] = dtr_data.get('alpha_R', 0.004)
    self.R_ac = R_ac_20 * (1.0 + alpha_R * (self.t_conductor - 20.0))
    if 'R_ac' in y_list_considered:
        R_ac = backend.symbols(f'R_ac_{line_name}')
        self.dae['g'] += [self.R_ac - R_ac]
        self.dae['y_ini'] += [R_ac]
        self.dae['y_run'] += [R_ac]
    else:
        R_ac = self.R_ac

    # Joule heating: q_j = i_cond^2 · R_ac(T), with i_cond = I_line·K_dtr.
    # i_line_sq is |i_cond|^2 built without a square root (branch_current.py).
    self.p_j = self.i_line_sq * self.R_ac
    if 'p_j' in y_list_considered:
        p_j = backend.symbols(f'p_j_{line_name}')
        self.dae['g'] += [self.p_j - p_j]
        self.dae['y_ini'] += [p_j]
        self.dae['y_run'] += [p_j]
    else:
        p_j = self.p_j

    # --- G. Transient heat balance ------------------------------------------
    # m·Cp·dT/dt = q_in − q_out = (q_j + q_s) − (q_c + q_r). The conductor
    # temperature T is a true dynamic state; at steady state the bracket is 0,
    # giving the IEEE-738 static rating. m·Cp = Cp · Mass_linear  [J/(m·K)].
    self.balance = self.p_j + self.p_s - self.p_c - self.p_r
    dt_conductor = 1/(self.Cp*self.Mass_linear)*self.balance

    self.dae['f'] += [     dt_conductor]
    self.dae['x'] += [ self.t_conductor]


    self.dae['xy_0_dict'].update({f"p_line_pu_{line_name}":1})
    self.dae['xy_0_dict'].update({f'dens_{line_name}':1.16})
    self.dae['xy_0_dict'].update({f'reynolds_{line_name}' :  1732.85})
    self.dae['xy_0_dict'].update({f'rayleigh_{line_name}' :  20558.69})
    self.dae['xy_0_dict'].update({f'nu_{line_name}' :  21.49})
    self.dae['xy_0_dict'].update({f'p_c_{line_name}' :  17.69})
    self.dae['xy_0_dict'].update({f'p_s_{line_name}' :  22.16})
    self.dae['xy_0_dict'].update({f'R_ac_{line_name}' :  0.08})
    self.dae['xy_0_dict'].update({f'p_j_{line_name}' :  0.1})
    self.dae['xy_0_dict'].update({f't_conductor_{line_name}' : 35.14})


    self.dae['h_dict'].update({f'dens_{line_name}':dens})
    self.dae['h_dict'].update({f'reynolds_{line_name}' :  reynolds})
    self.dae['h_dict'].update({f'rayleigh_{line_name}' :  rayleigh})
    self.dae['h_dict'].update({f'nu_{line_name}' :  nu})
    self.dae['h_dict'].update({f'p_c_{line_name}' : p_c})
    self.dae['h_dict'].update({f'p_s_{line_name}' : p_s})
    self.dae['h_dict'].update({f'R_ac_{line_name}' : R_ac})
    self.dae['h_dict'].update({f'p_j_{line_name}' :  p_j})
    self.dae['h_dict'].update({f't_conductor_{line_name}' : self.t_conductor})


    return self.dae['g'], self.dae['y_ini'], self.dae['y_run']
    

def _line_impedance(grid, line):
    r"""
    Return ``(R_pu_20, X_pu, R_ac_20)`` for one ``lines[]`` entry.

    - ``R_pu_20`` / ``X_pu`` : series resistance / reactance in system pu, at the
      20 °C reference temperature.
    - ``R_ac_20`` : conductor AC resistance per metre [Ω/m] at 20 °C, for the
      thermal heat balance — derived from the same line resistance so the
      heating and the network share a single resistance (``None`` when no length
      ``km`` is available to convert to per-metre).
    """
    sys = grid.system
    S_base = sys['S_base']
    bus_idx = grid.buses_list.index(line['bus_j'])
    U_base = grid.buses[bus_idx]['U_kV'] * 1000.0
    Z_base = U_base ** 2 / S_base

    if 'X_pu' in line:
        S_line = 1e6 * line['S_mva']
        R_pu = line['R_pu'] * S_base / S_line
        X_pu = line['X_pu'] * S_base / S_line
        R_ac_20 = R_pu * Z_base / (line['km'] * 1000.0) if 'km' in line else None
        return R_pu, X_pu, R_ac_20
    if 'X' in line:
        R_pu = line['R'] / Z_base
        X_pu = line['X'] / Z_base
        R_ac_20 = line['R'] / (line['km'] * 1000.0) if 'km' in line else None
        return R_pu, X_pu, R_ac_20
    if 'X_km' in line:
        R_pu = line['R_km'] * line['km'] / Z_base
        X_pu = line['X_km'] * line['km'] / Z_base
        return R_pu, X_pu, line['R_km'] / 1000.0
    raise ValueError(
        f"DTR line {line.get('bus_j')}-{line.get('bus_k')}: no impedance key "
        f"(X_pu / X / X_km) found")


def add_line_dtr(self, line):

    backend = self.backend
    sys = self.system

    bus_j = line['bus_j']
    bus_k = line['bus_k']

    idx_j = self.buses_list.index(bus_j)
    idx_k = self.buses_list.index(bus_k)

    self.A[self.it, idx_j] = 1
    self.A[self.it, idx_k] = -1
    self.A[self.it + 1, idx_j] = 1
    self.A[self.it + 2, idx_k] = 1

    line_name = f"{bus_j}_{bus_k}"

    # Series impedance at 20 °C (pu) and conductor resistance per metre [Ω/m].
    R_pu_20, X_pu, R_ac_20 = _line_impedance(self, line)

    # Resolve the cable/ambient data once. Inject the line-derived conductor
    # resistance and a single temperature coefficient so the heat balance and
    # the network admittance share one data-driven resistance R(T) =
    # R_20·(1 + alpha_R·(T − 20)). Either may still be overridden in dtr_data.
    dtr_data = _resolve_dtr_data(self, line)
    if R_ac_20 is not None:
        dtr_data.setdefault('R_ac_20', R_ac_20)
    dtr_data.setdefault('alpha_R', 0.004)

    # Dispatch to the thermal model — defines t_conductor_{name} and registers
    # the alpha_R_{name} / R_ac_20_{name} parameters used below.
    # model: "overhead" (IEEE-738, default) or "cable" (IEC 60853 2-node).
    model = str(dtr_data.get('model', 'overhead')).lower()
    if model in ('cable', 'underground', 'iec60853'):
        from pydae.bps.lines.line_dtr_cable import DTR_Cable_Symbolic
        DTR_Cable_Symbolic(self, line_name, bus_j, bus_k, dtr_data)
    else:
        DTR_Symbolic(self, line_name, bus_j, bus_k, dtr_data)

    if 'dtr' not in line:
        line.update({'dtr': False})

    # Primitive admittance entries. g_{line}/b_{line} are now ALGEBRAIC
    # VARIABLES (closed-loop DTR): they depend on the conductor temperature so
    # the network resistance, losses and voltage drop rise as the line heats.
    g_jk = backend.symbols(f"g_{line_name}")
    b_jk = backend.symbols(f"b_{line_name}")
    bs_jk = backend.symbols(f"bs_{line_name}")
    self.G_primitive[self.it, self.it] = g_jk
    self.B_primitive[self.it, self.it] = b_jk
    self.B_primitive[self.it + 1, self.it + 1] = bs_jk / 2
    self.B_primitive[self.it + 2, self.it + 2] = bs_jk / 2

    # Temperature-dependent series admittance: R(T) = R_pu_20·(1+alpha_R·(T−20)).
    t_cond = backend.symbols(f't_conductor_{line_name}')
    alpha_R = backend.symbols(f'alpha_R_{line_name}')
    R_pu_T = R_pu_20 * (1.0 + alpha_R * (t_cond - 20.0))
    denom = R_pu_T ** 2 + X_pu ** 2
    self.dae['g'] += [R_pu_T / denom - g_jk, -X_pu / denom - b_jk]
    self.dae['y_ini'] += [g_jk, b_jk]
    self.dae['y_run'] += [g_jk, b_jk]
    denom_20 = R_pu_20 ** 2 + X_pu ** 2
    self.dae['xy_0_dict'].update({
        str(g_jk): R_pu_20 / denom_20,
        str(b_jk): -X_pu / denom_20,
    })

    # Shunt charging susceptance (temperature-independent parameter).
    self.dae['params_dict'].update({f'bs_{line_name}': 0.0})
    if 'Bs_pu' in line:
        S_line = 1e6 * line['S_mva']
        self.dae['params_dict'][f'bs_{line_name}'] = line['Bs_pu'] * S_line / sys['S_base']
    if 'Bs_km' in line:
        bus_idx = self.buses_list.index(line['bus_j'])
        U_base = self.buses[bus_idx]['U_kV'] * 1000
        Y_base = sys['S_base'] / U_base ** 2
        self.dae['params_dict'][f'bs_{line_name}'] = line['Bs_km'] * line['km'] / Y_base

    self.it += 3


def change_line(model,bus_j,bus_k, *args,**kwagrs):
    line = kwagrs
    S_base = model.get_value('S_base')
    
    line_name = f"{bus_j}_{bus_k}"
    if 'X_pu' in line:
        if 'S_mva' in line: S_line = 1e6*line['S_mva']
        R = line['R_pu']*S_base/S_line  # in pu of the model base
        X = line['X_pu']*S_base/S_line  # in pu of the model base
    if 'X' in line:
        U_base = model.get_value(f'U_{bus_j}_n') 
        Z_base = U_base**2/S_base
        R = line['R']/Z_base  # in pu of the model base
        X = line['X']/Z_base  # in pu of the model base
    if 'X_km' in line:
        U_base = model.get_value(f'U_{bus_j}_n')
        Z_base = U_base**2/S_base
        R = line['R_km']*line['km']/Z_base  # in pu of the model base
        X = line['X_km']*line['km']/Z_base  # in pu of the model base
    if 'Bs_km' in line:
        U_base = model.get_value(f'U_{bus_j}_n')
        Z_base = U_base**2/S_base
        Y_base = 1.0/Z_base
        Bs = line['Bs_km']*line['km']/Y_base  # in pu of the model base
        bs = Bs
        model.set_value(f'bs_{line_name}',bs)

    G =  R/(R**2+X**2)
    B = -X/(R**2+X**2)
    model.set_value(f"g_{line_name}",G)
    model.set_value(f"b_{line_name}",B)

def get_line_current(model,bus_j,bus_k, units='A'):
    V_j_m,theta_j = model.get_mvalue([f'V_{bus_j}',f'theta_{bus_j}'])
    V_k_m,theta_k = model.get_mvalue([f'V_{bus_k}',f'theta_{bus_k}'])
    name = f'b_{bus_j}_{bus_k}'
    if name in model.params_list:
        B_j_k,G_j_k = model.get_mvalue([f'b_{bus_j}_{bus_k}',f'g_{bus_j}_{bus_k}'])
        Bs_j_k = model.get_value(f'bs_{bus_j}_{bus_k}')
    elif f'b_{bus_k}_{bus_j}' in model.params_list:
        B_j_k,G_j_k = model.get_mvalue([f'b_{bus_k}_{bus_j}',f'g_{bus_k}_{bus_j}'])
        Bs_j_k = model.get_value(f'bs_{bus_k}_{bus_j}')

    
    Y_j_k = G_j_k + 1j*B_j_k 
    V_j = V_j_m*np.exp(1j*theta_j)
    V_k = V_k_m*np.exp(1j*theta_k)
    I_j_k_pu = (V_j - V_k)*Y_j_k + 1j*V_j*Bs_j_k/2

    if units == 'A':
        S_b = model.get_value('S_base')
        U_b = model.get_value(f'U_{bus_j}_n')
        I_b = S_b/(np.sqrt(3)*U_b)
        I_j_k = I_j_k_pu*I_b

    return I_j_k


def test_line_pu_build():
    
    from pydae.bps import BpsBuilder
    from pydae.build_v2 import build_mkl,build_numba

    data = {
        "system":{"name":"temp","S_base":100e6, "K_p_agc":0.01,"K_i_agc":0.01, "K_xif":0.01},       
        "buses":[{"name":"1", "P_W":0.0,"Q_var":0.0,"U_kV":20.0},
                 {"name":"2", "P_W":0.0,"Q_var":0.0,"U_kV":20.0}],
        "lines":[{"bus_j":"1", "bus_k":"2", "X_pu":0.15,"R_pu":0.0,"Bs_pu":0.5, "S_mva":900.0, "monitor":True, "dtr":True}],
        "sources":[{"bus":"1","type":"vsource","V_mag_pu":1.0,"V_ang_rad":0.0,"K_delta":0.1}]
        }

    grid = BpsBuilder(data)
    grid.uz_jacs = False
    grid.verbose = True
    grid.construct(f'temp')
    build_numba(grid.sys_dict)

def test_line_pu_ini():
    
    import temp

    model = temp.model()
    model.ini({'P_2':0e6, "K_xif":0.01},'xy_0.json')
    print(f"V_1 = {model.get_value('V_1'):2.2f}, V_2 = {model.get_value('V_2'):2.2f}")

    model.report_y()

def test_line_km_build():
    
    from pydae.bps import BpsBuilder







    grid = BpsBuilder(data)
    grid.uz_jacs = False
    grid.verbose = True
    grid.construct(f'temp')
    build_numba(grid.sys_dict)

def test_line_km_ini():


    
    from pydae.bps import BpsBuilder
    from pydae.core.builder.casadi_builder import CasadiBuilder
    from pydae.core.model.casadi_model import CasadiModel

    R_km = 0.0268  # (Ω/km)
    X_km = 0.2766  # (Ω/km)
    B_km = 4.59e-6 # (℧-1/Km)

    N_parallel = 44.0

    Lenght_km = 22000.0/N_parallel
    
    data = {
        "system":{"name":"temp","S_base":100e6, "K_p_agc":0.01,"K_i_agc":0.01, "K_xif":0.01},       
        "buses":[{"name":"1", "P_W":0.0,"Q_var":0.0,"U_kV":400.0},
                 {"name":"2", "P_W":5000e6,"Q_var":0.0,"U_kV":400.0}],
        "lines":[{"bus_j":"1", "bus_k":"2", 'X_km':X_km/44, 'R_km':R_km/44, 'Bs_km':B_km*44, "km":Lenght_km, "monitor":True, "dtr":True}],
        "sources":[{"bus":"1","type":"vsource","V_mag_pu":1.0,"V_ang_rad":0.0,"K_delta":0.1}]
        }
    grid = BpsBuilder(data, use_casadi=True)
    grid.construct('line_dtr_test')
    bld = CasadiBuilder(grid.sys_dict).build()

    model = CasadiModel(bld)

    model.ini({'P_2':5000e6,'Q_2':0, "K_xif":0.01},'xy_0.json')
    print(f"V_1 = {model.get_value('V_1'):2.2f}, V_2 = {model.get_value('V_2'):2.2f}")

    print("report_y:")
    model.report_y()
    print("-----------------------------------------------------------------------")
    print("report_z:")
    model.report_z()

    U_1 = 400e3
    U_2 = 400e3 
    Bs = 4.59e-6*22000
    Q = U_1**2*Bs/2 + U_2**2*Bs/2
    print(f"Q = {Q/1e6:2.2f} Mvar")


if __name__ == "__main__":

    #test_line_pu_build()
    #test_line_pu_ini()

    #test_line_km_build()
    test_line_km_ini()


