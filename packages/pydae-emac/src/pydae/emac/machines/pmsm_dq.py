# -*- coding: utf-8 -*-
r"""
Linear permanent-magnet synchronous machine in the rotor's dq frame (PMSM_DQ).

Constant inductances and a magnet flux on the d axis.  Covers the surface
PM machine ($L_d = L_q$), the interior PM machine ($L_d < L_q$) and, with
$\Psi_{pm} = 0$, the synchronous reluctance machine.  Amplitude-invariant dq
(peak values), the same convention as the FEA tables and the switched drive.

The fluxes are the states, as in the look-up-table machine
(``pmsm_lut``), so the two are interchangeable in a model: only the
flux-current relation changes.

**Stator, rotor frame**

$$\frac{d\lambda_d}{dt} = v_d - R_s i_d + \omega_e \lambda_q, \qquad
  \frac{d\lambda_q}{dt} = v_q - R_s i_q - \omega_e \lambda_d,
  \qquad \omega_e = p\,\omega_m$$

**Flux linkage (algebraic)**

$$0 = \lambda_d - (L_d i_d + \Psi_{pm}), \qquad 0 = \lambda_q - L_q i_q$$

**Torque**

$$T_e = \tfrac{3}{2} p \left(\lambda_d i_q - \lambda_q i_d\right)$$

The rotor is :mod:`pydae.emac.machines.mechanics` (``mech``: ``"imposed"``
or ``"free"``).  The stator voltages $v_d$, $v_q$ are inputs until a
controller (``ctrls``) drives them, which promotes them to algebraic
variables.  All symbols carry the suffix ``_{name}``.
"""

from pydae.emac.machines import mechanics
from pydae.emac.utils.docs import descriptions_tables


def descriptions():
    """Single source of truth for parameters, inputs, states and outputs."""
    d = []
    d += [{"type": "Parameter", "tex": "R_s", "data": "R_s", "model": "R_s", "default": 0.1, "description": "Stator phase resistance", "units": "Ohm"}]
    d += [{"type": "Parameter", "tex": "L_d", "data": "L_d", "model": "L_d", "default": 1e-3, "description": "d-axis inductance", "units": "H"}]
    d += [{"type": "Parameter", "tex": "L_q", "data": "L_q", "model": "L_q", "default": 1e-3, "description": "q-axis inductance", "units": "H"}]
    d += [{"type": "Parameter", "tex": r"\Psi_{pm}", "data": "Psi_pm", "model": "Psi_pm", "default": 0.1, "description": "Magnet flux linkage (peak, d axis)", "units": "Wb"}]
    d += [{"type": "Parameter", "tex": "p", "data": "p", "model": "p", "default": 2, "description": "Pole pairs", "units": "-"}]
    d += [{"type": "Input", "tex": "v_d", "data": "v_d", "model": "v_d", "default": 0.0, "description": "d-axis stator voltage (until a controller drives it)", "units": "V"}]
    d += [{"type": "Input", "tex": "v_q", "data": "v_q", "model": "v_q", "default": 0.0, "description": "q-axis stator voltage (until a controller drives it)", "units": "V"}]
    d += [{"type": "Dynamic State", "tex": r"\lambda_d", "data": "", "model": "lam_d", "default": "", "description": "d-axis flux linkage", "units": "Wb"}]
    d += [{"type": "Dynamic State", "tex": r"\lambda_q", "data": "", "model": "lam_q", "default": "", "description": "q-axis flux linkage", "units": "Wb"}]
    d += [{"type": "Algebraic State", "tex": "i_d", "data": "", "model": "i_d", "default": "", "description": "d-axis current (peak)", "units": "A"}]
    d += [{"type": "Algebraic State", "tex": "i_q", "data": "", "model": "i_q", "default": "", "description": "q-axis current (peak)", "units": "A"}]
    d += mechanics.descriptions()
    d += [{"type": "Output", "tex": "T_e", "data": "", "model": "T_e", "default": "", "description": "Electromagnetic torque", "units": "N.m"}]
    d += [{"type": "Output", "tex": "P_m", "data": "", "model": "P_m", "default": "", "description": "Mechanical power T_e omega_m", "units": "W"}]
    d += [{"type": "Output", "tex": "P_s", "data": "", "model": "P_s", "default": "", "description": "Stator electrical power 1.5 (v_d i_d + v_q i_q)", "units": "W"}]
    return d


def pmsm_dq(grid, name, data_dict):
    """Append a linear dq PMSM named ``name`` to ``grid.dae``."""
    bk = grid.backend
    dae = grid.dae
    default = {item["data"]: item["default"] for item in descriptions() if item["data"]}
    val = {k: float(data_dict.get(k, default[k])) for k in ("R_s", "L_d", "L_q", "Psi_pm", "p")}

    R_s, L_d, L_q, Psi_pm, p = (bk.symbols(f"{k}_{name}") for k in ("R_s", "L_d", "L_q", "Psi_pm", "p"))
    lam_d, lam_q = bk.symbols(f"lam_d_{name}"), bk.symbols(f"lam_q_{name}")
    i_d, i_q = bk.symbols(f"i_d_{name}"), bk.symbols(f"i_q_{name}")
    v_d, v_q = bk.symbols(f"v_d_{name}"), bk.symbols(f"v_q_{name}")

    T_e = 1.5 * p * (lam_d * i_q - lam_q * i_d)
    omega_m = mechanics.add_mechanics(dae, name, bk, data_dict, T_e)
    omega_e = p * omega_m

    dae["f_list"] += [v_d - R_s * i_d + omega_e * lam_q,
                      v_q - R_s * i_q - omega_e * lam_d]
    dae["x_list"] += [lam_d, lam_q]
    dae["g_list"] += [lam_d - (L_d * i_d + Psi_pm),
                      lam_q - L_q * i_q]
    dae["y_ini_list"] += [i_d, i_q]
    dae["y_run_list"] += [i_d, i_q]
    dae["u_ini_dict"].update({f"{v_d}": float(data_dict.get("v_d", 0.0)),
                              f"{v_q}": float(data_dict.get("v_q", 0.0))})
    dae["u_run_dict"].update({f"{v_d}": float(data_dict.get("v_d", 0.0)),
                              f"{v_q}": float(data_dict.get("v_q", 0.0))})
    dae["params_dict"].update({f"{k}_{name}": v for k, v in val.items()})
    dae["xy_0_dict"].update({f"{lam_d}": val["Psi_pm"], f"{lam_q}": 0.0,
                             f"{i_d}": 0.0, f"{i_q}": 0.0})
    dae["h_dict"].update({f"T_e_{name}": T_e,
                          f"P_s_{name}": 1.5 * (v_d * i_d + v_q * i_q)})
    grid.machines[name] = dict(type="pmsm_dq", omega_e=omega_e, data=data_dict,
                               L_dq=(val["L_d"], val["L_q"]), R_s=val["R_s"])


__doc__ += descriptions_tables(descriptions())


def test():
    """Build the sibling ``pmsm_dq.hjson`` drive (machine + FOC current loop),
    initialise it at its references and check the closed-form steady state."""
    import os

    from pydae.emac import EmacBuilder

    drive = EmacBuilder(os.path.join(os.path.dirname(__file__), "pmsm_dq.hjson"))
    model = drive.build("pmsm_dq_test")
    model.ini({}, drive.sys_dict["xy_0_dict"])
    m = drive.data["machines"][0]
    c = drive.data["ctrls"][0]
    i_d, i_q = c["i_d_ref"], c["i_q_ref"]
    w_e = m["p"] * m["omega_m"]
    assert abs(model.get_value("i_d_1") - i_d) < 1e-8
    assert abs(model.get_value("i_q_1") - i_q) < 1e-8
    assert abs(model.get_value("v_d_1") - (m["R_s"] * i_d - w_e * m["L_q"] * i_q)) < 1e-6
    assert abs(model.get_value("v_q_1") - (m["R_s"] * i_q + w_e * (m["L_d"] * i_d + m["Psi_pm"]))) < 1e-6
    return model
