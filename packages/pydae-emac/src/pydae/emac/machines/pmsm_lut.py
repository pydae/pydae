# -*- coding: utf-8 -*-
r"""
Synchronous machine from FEA look-up tables, mean-value dq model (PMSM_LUT).

Any synchronous machine an FEA can map -- PM, interior PM, reluctance,
PM-assisted reluctance -- with its saturation and cross-saturation, in the
rotor's dq frame.  The tables are the FEA's, inverted to the direction a
flux-state model needs and averaged over the rotor angle
(:func:`pydae.emac.luts.build_flux_tables`, :func:`~pydae.emac.luts.mean_tables`):

$$i_d = I_d(\lambda_d, \lambda_q), \qquad i_q = I_q(\lambda_d, \lambda_q),
  \qquad T_e = \mathcal{T}(\lambda_d, \lambda_q)$$

each a bicubic B-spline on a regular flux grid.  The stator equations and
the rotor (:mod:`pydae.emac.machines.mechanics`) are those of ``pmsm_dq``:

$$\frac{d\lambda_d}{dt} = v_d - R_s i_d + \omega_e \lambda_q, \qquad
  \frac{d\lambda_q}{dt} = v_q - R_s i_q - \omega_e \lambda_d$$

**What "mean-value" leaves out.**  The torque is the FEA's own (Maxwell
stress), averaged over the angle; the spatial ripple, and everything the
inverter adds, is the switched drive's job (``pydae.emac.drive``), which is
built from the same tables.  Without the angle the model has a true
equilibrium, so ``ini()``, eigenvalues and controller design all apply.

**Frame.**  The model works in whatever dq frame the table is in: for a PM
map d is usually on the magnet, for a reluctance map on the high-permeance
axis.  Means do not depend on where the table puts the d axis against
phase a, so ``theta_offset_e`` does not arise here.

**CasADi only** (``EmacBuilder(..., use_casadi=True)``, the default): the
B-spline is a CasADi interpolant.

**Data.**  Either ``lut_file`` (a ``FluxMap_dqt.npz``, see
:mod:`pydae.emac.luts.flux_tables`) with optional ``n_lam`` (flux grid
points per axis, default 40), or -- from Python -- ``tables``, the dict
:func:`~pydae.emac.luts.build_flux_tables` returns.  ``p`` comes from the
table.
"""

from pydae.emac.machines import mechanics
from pydae.emac.utils.docs import descriptions_tables


def descriptions():
    """Single source of truth for parameters, inputs, states and outputs."""
    d = []
    d += [{"type": "Parameter", "tex": "R_s", "data": "R_s", "model": "R_s", "default": 0.1, "description": "Stator phase resistance", "units": "Ohm"}]
    d += [{"type": "Parameter", "tex": "p", "data": "", "model": "p", "default": "", "description": "Pole pairs (from the table)", "units": "-"}]
    d += [{"type": "Input", "tex": "v_d", "data": "v_d", "model": "v_d", "default": 0.0, "description": "d-axis stator voltage (until a controller drives it)", "units": "V"}]
    d += [{"type": "Input", "tex": "v_q", "data": "v_q", "model": "v_q", "default": 0.0, "description": "q-axis stator voltage (until a controller drives it)", "units": "V"}]
    d += [{"type": "Dynamic State", "tex": r"\lambda_d", "data": "", "model": "lam_d", "default": "", "description": "d-axis flux linkage", "units": "Wb"}]
    d += [{"type": "Dynamic State", "tex": r"\lambda_q", "data": "", "model": "lam_q", "default": "", "description": "q-axis flux linkage", "units": "Wb"}]
    d += [{"type": "Algebraic State", "tex": "i_d", "data": "", "model": "i_d", "default": "", "description": "d-axis current (peak), from the table", "units": "A"}]
    d += [{"type": "Algebraic State", "tex": "i_q", "data": "", "model": "i_q", "default": "", "description": "q-axis current (peak), from the table", "units": "A"}]
    d += mechanics.descriptions()
    d += [{"type": "Output", "tex": "T_e", "data": "", "model": "T_e", "default": "", "description": "Electromagnetic torque, theta-mean, from the table", "units": "N.m"}]
    d += [{"type": "Output", "tex": "P_m", "data": "", "model": "P_m", "default": "", "description": "Mechanical power T_e omega_m", "units": "W"}]
    d += [{"type": "Output", "tex": "P_s", "data": "", "model": "P_s", "default": "", "description": "Stator electrical power 1.5 (v_d i_d + v_q i_q)", "units": "W"}]
    return d


def load_tables(data_dict):
    """The theta-mean tables of a machine item: from ``tables`` (Python) or
    ``lut_file`` (+ ``n_lam``)."""
    from pydae.emac.luts.flux_tables import build_flux_tables, load_fea_lut, mean_tables

    if "tables" in data_dict:
        tables = data_dict["tables"]
    elif "lut_file" in data_dict:
        tables = build_flux_tables(load_fea_lut(data_dict["lut_file"]),
                                   n_lam=int(data_dict.get("n_lam", 40)))
    else:
        raise KeyError("pmsm_lut needs 'lut_file' or 'tables'")
    return tables if "n_basis" not in tables else mean_tables(tables)


def pmsm_lut(grid, name, data_dict):
    """Append a look-up-table dq machine named ``name`` to ``grid.dae``."""
    if not grid.use_casadi:
        raise NotImplementedError("pmsm_lut is CasADi-only (a B-spline interpolant): "
                                  "use EmacBuilder(..., use_casadi=True)")
    import casadi as ca

    bk = grid.backend
    dae = grid.dae
    tab = load_tables(data_dict)
    spline = ca.interpolant(f"lut_{name}", "bspline", [tab["grid_d"], tab["grid_q"]],
                            tab["data"], {"degree": [3, 3]})

    R_s, p = bk.symbols(f"R_s_{name}"), bk.symbols(f"p_{name}")
    lam_d, lam_q = bk.symbols(f"lam_d_{name}"), bk.symbols(f"lam_q_{name}")
    i_d, i_q = bk.symbols(f"i_d_{name}"), bk.symbols(f"i_q_{name}")
    v_d, v_q = bk.symbols(f"v_d_{name}"), bk.symbols(f"v_q_{name}")

    I_d, I_q, T_e = ca.vertsplit(spline(ca.vertcat(lam_d, lam_q)))
    omega_m = mechanics.add_mechanics(dae, name, bk, data_dict, T_e)
    omega_e = p * omega_m

    dae["f_list"] += [v_d - R_s * i_d + omega_e * lam_q,
                      v_q - R_s * i_q - omega_e * lam_d]
    dae["x_list"] += [lam_d, lam_q]
    dae["g_list"] += [i_d - I_d, i_q - I_q]
    dae["y_ini_list"] += [i_d, i_q]
    dae["y_run_list"] += [i_d, i_q]
    for k, s in (("v_d", v_d), ("v_q", v_q)):
        dae["u_ini_dict"][f"{s}"] = float(data_dict.get(k, 0.0))
        dae["u_run_dict"][f"{s}"] = float(data_dict.get(k, 0.0))
    dae["params_dict"].update({f"{R_s}": float(data_dict.get("R_s", 0.1)),
                               f"{p}": float(tab["p"])})
    dae["xy_0_dict"].update({f"{lam_d}": float(tab["lam_0"][0]), f"{lam_q}": float(tab["lam_0"][1]),
                             f"{i_d}": 0.0, f"{i_q}": 0.0})
    dae["h_dict"].update({f"T_e_{name}": T_e,
                          f"P_s_{name}": 1.5 * (v_d * i_d + v_q * i_q)})
    grid.machines[name] = dict(type="pmsm_lut", omega_e=omega_e, data=data_dict,
                               tables=tab, R_s=float(data_dict.get("R_s", 0.1)))


__doc__ += descriptions_tables(descriptions())
