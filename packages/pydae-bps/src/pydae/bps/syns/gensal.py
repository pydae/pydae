# -*- coding: utf-8 -*-
r"""
GENSAL — salient-pole 5th-order synchronous machine (IEEE 1110 Model 2.1 /
PSS/E ``GENSAL``).

Salient-pole counterpart of :mod:`pydae.bps.syns.genrou`. Used for hydro
generators. Five states (no q-axis transient EMF — there is no field
winding on the q-axis of a salient-pole rotor):

- $\delta, \omega$ — rotor angle, electrical speed
- $e_q'$ — d-axis transient EMF
- $e_q''$, $e_d''$ — subtransient EMFs

Same convention as ``genrou``: $X_d''$, $X_q''$ are **terminal-referred**
(IEEE 115-2019 Eq. (88)). No ``X_l`` field. Saturation is applied on the
d-axis only — the salient q-axis has no field winding and is dominated
by reluctance, so $S_q = 0$ is the standard PSS/E choice.

**Auxiliar equations**

$$v_d = V \sin(\delta - \theta), \quad v_q = V \cos(\delta - \theta)$$
$$\tau_e = (v_d + R_a i_d)\,i_d + (v_q + R_a i_q)\,i_q$$
$$\psi_{AT} = \sqrt{e_q'^{\,2} + \epsilon}, \quad
  S(\psi_{AT}) = \frac{B_{sat}\,\max(\psi_{AT} - A_{sat},\,0)^2}{\psi_{AT}}$$
$$S_d = S(\psi_{AT}), \quad S_q = 0$$
$$\omega_s = \omega_{coi}$$

**Dynamic equations**

$$\frac{d\delta}{dt} = \Omega_b (\omega - \omega_s) - K_\delta \delta$$
$$\frac{d\omega}{dt} = \frac{1}{2H}\bigl(\tau_m - \tau_e - D(\omega - \omega_s)\bigr)$$
$$T_{d0}' \frac{de_q'}{dt} = -e_q' - (X_d - X_d')\,i_d - S_d\,e_q' + v_f$$
$$T_{d0}'' \frac{de_q''}{dt} = -e_q'' + e_q' - (X_d' - X_d'')\,i_d$$
$$T_{q0}'' \frac{de_d''}{dt} = -e_d'' + (X_q - X_q'')\,i_q$$

The q-axis subtransient is driven directly from $i_q$ (single damper
winding; no transient stage), which is the structural change vs
``genrou``. The d-axis sub-stack is unchanged.

**Algebraic equations** (terminal-referred subtransient, identical to
``genrou``):

$$0 = v_q + R_a i_q - e_q'' + X_d''\,i_d$$
$$0 = v_d + R_a i_d - e_d'' - X_q''\,i_q$$
$$0 = i_d v_d + i_q v_q - p_g$$
$$0 = i_d v_q - i_q v_d - q_g$$

**Choosing between gensal and genrou.** Pick ``gensal`` for hydro units
(salient poles, $X_q < X_d$, no q-axis field) and ``genrou`` for turbo
units (round rotor, $X_q \approx X_d$, q-axis field present). PSS/E
GENSAL data tables drop straight in: ``X_d, X_q, X1d, X2d, X2q, T1d0,
T2d0, T2q0, S_10, S_12, H, D, R_a``.
"""

import numpy as np
import io


def descriptions():
    """Single source of truth for parameters, inputs, states, outputs."""
    descriptions_list = []

    # Parameters
    descriptions_list += [{"type": "Parameter", "tex": "S_n",         "data": "S_n",     "model": "S_n",     "default": 100e6, "description": "Nominal power", "units": "VA"}]
    descriptions_list += [{"type": "Parameter", "tex": "F_n",         "data": "F_n",     "model": "F_n",     "default": 50.0,  "description": "Nominal frequency", "units": "Hz"}]
    descriptions_list += [{"type": "Parameter", "tex": "H",           "data": "H",       "model": "H",       "default": 4.0,   "description": "Inertia constant", "units": "s"}]
    descriptions_list += [{"type": "Parameter", "tex": "D",           "data": "D",       "model": "D",       "default": 0.0,   "description": "Damping coefficient", "units": "-"}]
    descriptions_list += [{"type": "Parameter", "tex": "X_d",         "data": "X_d",     "model": "X_d",     "default": 1.10,  "description": "d-axis synchronous reactance", "units": "pu-m"}]
    descriptions_list += [{"type": "Parameter", "tex": "X_q",         "data": "X_q",     "model": "X_q",     "default": 0.70,  "description": "q-axis synchronous reactance (salient: X_q < X_d)", "units": "pu-m"}]
    descriptions_list += [{"type": "Parameter", "tex": "X'_d",        "data": "X1d",     "model": "X1d",     "default": 0.25,  "description": "d-axis transient reactance", "units": "pu-m"}]
    descriptions_list += [{"type": "Parameter", "tex": "X''_d",       "data": "X2d",     "model": "X2d",     "default": 0.20,  "description": "d-axis subtransient reactance (terminal-referred)", "units": "pu-m"}]
    descriptions_list += [{"type": "Parameter", "tex": "X''_q",       "data": "X2q",     "model": "X2q",     "default": 0.20,  "description": "q-axis subtransient reactance (terminal-referred)", "units": "pu-m"}]
    descriptions_list += [{"type": "Parameter", "tex": "T'_{d0}",     "data": "T1d0",    "model": "T1d0",    "default": 8.0,   "description": "d-axis open-circuit transient time constant", "units": "s"}]
    descriptions_list += [{"type": "Parameter", "tex": "T''_{d0}",    "data": "T2d0",    "model": "T2d0",    "default": 0.03,  "description": "d-axis open-circuit subtransient time constant", "units": "s"}]
    descriptions_list += [{"type": "Parameter", "tex": "T''_{q0}",    "data": "T2q0",    "model": "T2q0",    "default": 0.05,  "description": "q-axis open-circuit subtransient time constant", "units": "s"}]
    descriptions_list += [{"type": "Parameter", "tex": "R_a",         "data": "R_a",     "model": "R_a",     "default": 0.0,   "description": "Armature resistance", "units": "pu-m"}]
    descriptions_list += [{"type": "Parameter", "tex": "S_{1.0}",     "data": "S_10",    "model": "S_10",    "default": 0.0,   "description": "Saturation factor at E=1.0", "units": "-"}]
    descriptions_list += [{"type": "Parameter", "tex": "S_{1.2}",     "data": "S_12",    "model": "S_12",    "default": 0.0,   "description": "Saturation factor at E=1.2", "units": "-"}]
    descriptions_list += [{"type": "Parameter", "tex": "K_{\\delta}", "data": "K_delta", "model": "K_delta", "default": 0.0,   "description": "Reference-machine constant", "units": "-"}]
    descriptions_list += [{"type": "Parameter", "tex": "K_{sec}",     "data": "K_sec",   "model": "K_sec",   "default": 0.0,   "description": "Secondary-frequency control participation", "units": "-"}]

    # Inputs
    descriptions_list += [{"type": "Input", "tex": "p_m", "data": "p_m", "model": "p_m", "default": 0.5, "description": "Mechanical power", "units": "pu-m"}]
    descriptions_list += [{"type": "Input", "tex": "v_f", "data": "v_f", "model": "v_f", "default": 1.0, "description": "Field voltage", "units": "pu-m"}]

    # Dynamic States — note: no e1d (no q-axis transient EMF in a salient machine)
    descriptions_list += [{"type": "Dynamic State", "tex": "\\delta", "data": "", "model": "delta", "default": "", "description": "Rotor angle", "units": "rad"}]
    descriptions_list += [{"type": "Dynamic State", "tex": "\\omega", "data": "", "model": "omega", "default": "", "description": "Rotor speed", "units": "pu"}]
    descriptions_list += [{"type": "Dynamic State", "tex": "e'_q",    "data": "", "model": "e1q",   "default": "", "description": "q-axis transient EMF (d-axis flux)", "units": "pu-m"}]
    descriptions_list += [{"type": "Dynamic State", "tex": "e''_q",   "data": "", "model": "e2q",   "default": "", "description": "q-axis subtransient EMF", "units": "pu-m"}]
    descriptions_list += [{"type": "Dynamic State", "tex": "e''_d",   "data": "", "model": "e2d",   "default": "", "description": "d-axis subtransient EMF", "units": "pu-m"}]

    # Algebraic States
    descriptions_list += [{"type": "Algebraic State", "tex": "i_d", "data": "", "model": "i_d", "default": "", "description": "d-axis current", "units": "pu-m"}]
    descriptions_list += [{"type": "Algebraic State", "tex": "i_q", "data": "", "model": "i_q", "default": "", "description": "q-axis current", "units": "pu-m"}]
    descriptions_list += [{"type": "Algebraic State", "tex": "p_g", "data": "", "model": "p_g", "default": "", "description": "Active power injection (machine base)", "units": "pu-m"}]
    descriptions_list += [{"type": "Algebraic State", "tex": "q_g", "data": "", "model": "q_g", "default": "", "description": "Reactive power injection (machine base)", "units": "pu-m"}]

    # Outputs
    descriptions_list += [{"type": "Output", "tex": "p_e",    "data": "", "model": "p_e",   "default": "", "description": "Electrical power", "units": "pu-m"}]
    descriptions_list += [{"type": "Output", "tex": "v_f",    "data": "", "model": "v_f",   "default": "", "description": "Field voltage (echo)", "units": "pu-m"}]
    descriptions_list += [{"type": "Output", "tex": "v_d",    "data": "", "model": "v_d",   "default": "", "description": "d-axis terminal voltage", "units": "pu-m"}]
    descriptions_list += [{"type": "Output", "tex": "v_q",    "data": "", "model": "v_q",   "default": "", "description": "q-axis terminal voltage", "units": "pu-m"}]
    descriptions_list += [{"type": "Output", "tex": "S_{at}", "data": "", "model": "S_at",  "default": "", "description": "Saturation factor S(psi_AT)", "units": "-"}]

    return descriptions_list


def _saturation_constants(S_10, S_12):
    """Two-point saturation fit — same helper as in genrou.py."""
    if S_10 > 0.0 and S_12 > 0.0:
        R_val = np.sqrt(1.2 * S_12 / S_10)
        A_sat_val = (1.2 - R_val) / (1.0 - R_val)
        B_sat_val = S_10 / (1.0 - A_sat_val) ** 2
    else:
        A_sat_val = 0.8
        B_sat_val = 0.0
    return A_sat_val, B_sat_val


def gensal(grid, name, bus_name, data_dict):
    backend = grid.backend
    sin = backend.sin
    cos = backend.cos

    meta = descriptions()
    default_map = {item['data']: item['default'] for item in meta if item.get('data')}

    # Inputs
    V = backend.symbols(f"V_{bus_name}")
    theta = backend.symbols(f"theta_{bus_name}")
    p_m = backend.symbols(f"p_m_{name}")
    v_f = backend.symbols(f"v_f_{name}")
    omega_coi = backend.symbols("omega_coi")

    # Dynamic states — 5 instead of 6 (no e1d)
    delta = backend.symbols(f"delta_{name}")
    omega = backend.symbols(f"omega_{name}")
    e1q = backend.symbols(f"e1q_{name}")
    e2q = backend.symbols(f"e2q_{name}")
    e2d = backend.symbols(f"e2d_{name}")

    # Algebraic states
    i_d = backend.symbols(f"i_d_{name}")
    i_q = backend.symbols(f"i_q_{name}")
    p_g = backend.symbols(f"p_g_{name}")
    q_g = backend.symbols(f"q_g_{name}")

    # Parameters
    S_n = backend.symbols(f"S_n_{name}")
    Omega_b = backend.symbols(f"Omega_b_{name}")
    H = backend.symbols(f"H_{name}")
    T1d0 = backend.symbols(f"T1d0_{name}")
    T2d0 = backend.symbols(f"T2d0_{name}")
    T2q0 = backend.symbols(f"T2q0_{name}")
    X_d = backend.symbols(f"X_d_{name}")
    X_q = backend.symbols(f"X_q_{name}")
    X1d = backend.symbols(f"X1d_{name}")
    X2d = backend.symbols(f"X2d_{name}")
    X2q = backend.symbols(f"X2q_{name}")
    D = backend.symbols(f"D_{name}")
    R_a = backend.symbols(f"R_a_{name}")
    K_delta = backend.symbols(f"K_delta_{name}")

    S_10 = data_dict.get('S_10', default_map.get('S_10', 0.0))
    S_12 = data_dict.get('S_12', default_map.get('S_12', 0.0))
    A_sat_val, B_sat_val = _saturation_constants(S_10, S_12)
    A_sat = backend.symbols(f"A_sat_{name}")
    B_sat = backend.symbols(f"B_sat_{name}")

    params_list = ['S_n', 'H', 'T1d0', 'T2d0', 'T2q0',
                   'X_d', 'X_q', 'X1d', 'X2d', 'X2q',
                   'D', 'R_a', 'K_delta', 'K_sec']

    # Auxiliar
    v_d = V * sin(delta - theta)
    v_q = V * cos(delta - theta)
    tau_e = (v_d + R_a * i_d) * i_d + (v_q + R_a * i_q) * i_q
    omega_s = omega_coi

    # Saturation — d-axis only (PSS/E GENSAL convention)
    EPS = 1e-12
    psi_AT = backend.sqrt(e1q ** 2 + EPS)
    S_at = B_sat * backend.max(psi_AT - A_sat, 0) ** 2 / psi_AT
    S_d = S_at

    # Dynamic equations — 5 states; q-axis subtransient driven directly from i_q
    ddelta = Omega_b * (omega - omega_s) - K_delta * delta
    domega = 1 / (2 * H) * (p_m - tau_e - D * (omega - omega_s))
    de1q = (-e1q - (X_d - X1d) * i_d - S_d * e1q + v_f) / T1d0
    de2q = (-e2q + e1q - (X1d - X2d) * i_d) / T2d0
    de2d = (-e2d + (X_q - X2q) * i_q) / T2q0

    # Algebraic equations — same stator form as genrou
    g_i_d = v_q + R_a * i_q - e2q + X2d * i_d
    g_i_q = v_d + R_a * i_d - e2d - X2q * i_q
    g_p_g = i_d * v_d + i_q * v_q - p_g
    g_q_g = i_d * v_q - i_q * v_d - q_g

    f_syn = [ddelta, domega, de1q, de2q, de2d]
    x_syn = [delta, omega, e1q, e2q, e2d]
    g_syn = [g_i_d, g_i_q, g_p_g, g_q_g]
    y_syn = [i_d, i_q, p_g, q_g]

    grid.H_total += H
    grid.omega_coi_numerator += omega * H * S_n
    grid.omega_coi_denominator += H * S_n

    grid.dae['f'] += f_syn
    grid.dae['x'] += x_syn
    grid.dae['g'] += g_syn
    grid.dae['y_ini'] += y_syn
    grid.dae['y_run'] += y_syn

    val_v_f = data_dict.get('v_f', default_map.get('v_f', 1.0))
    grid.dae['u_ini_dict'].update({f'{v_f}': val_v_f})
    grid.dae['u_run_dict'].update({f'{v_f}': val_v_f})

    val_p_m = data_dict.get('p_m', default_map.get('p_m', 0.5))
    grid.dae['u_ini_dict'].update({f'{p_m}': val_p_m})
    grid.dae['u_run_dict'].update({f'{p_m}': val_p_m})

    grid.dae['xy_0_dict'].update({str(omega): 1.0})
    grid.dae['xy_0_dict'].update({str(e1q): 1.0})
    grid.dae['xy_0_dict'].update({str(e2q): 1.0})
    grid.dae['xy_0_dict'].update({str(e2d): 0.0})
    grid.dae['xy_0_dict'].update({str(i_d): 0.5})
    grid.dae['xy_0_dict'].update({str(i_q): 0.5})

    grid.dae['h_dict'].update({f"p_e_{name}": tau_e})
    grid.dae['h_dict'].update({f"v_f_{name}": v_f})
    grid.dae['h_dict'].update({f"v_d_{name}": v_d})
    grid.dae['h_dict'].update({f"v_q_{name}": v_q})
    grid.dae['h_dict'].update({f"S_at_{name}": S_at})

    F_n_val = data_dict.get('F_n', default_map.get('F_n', 50.0))
    grid.dae['params_dict'].update({f"Omega_b_{name}": 2 * np.pi * F_n_val})

    grid.dae['params_dict'].update({f"A_sat_{name}": A_sat_val})
    grid.dae['params_dict'].update({f"B_sat_{name}": B_sat_val})

    for item in params_list:
        val = data_dict.get(item, default_map.get(item, 0.0))
        grid.dae['params_dict'].update({f"{item}_{name}": val})

    p_W = p_g * S_n
    q_var = q_g * S_n

    return p_W, q_var


# =============================================================================
# Sphinx Documentation Auto-Generator
# =============================================================================
def dict_list_to_aligned_markdown_table(data: list[dict]) -> str:
    if not data or not isinstance(data, list):
        return ""
    dict_data = [item for item in data if isinstance(item, dict)]
    if not dict_data:
        return ""

    all_headers = []
    header_set = set()
    for row_dict in dict_data:
        for key in row_dict.keys():
            if key not in header_set:
                header_set.add(key)
                all_headers.append(key)

    max_widths = {header: len(header) for header in all_headers}
    for row_dict in dict_data:
        for header in all_headers:
            max_widths[header] = max(max_widths[header], len(str(row_dict.get(header, ""))))

    output = io.StringIO()
    padded_headers = [header.ljust(max_widths[header]) for header in all_headers]
    output.write("| " + " | ".join(padded_headers) + " |\n")
    separators = ['-' * max_widths[header] for header in all_headers]
    output.write("| " + " | ".join(separators) + " |\n")

    for row_dict in dict_data:
        padded_values = [str(row_dict.get(header, "")).ljust(max_widths[header]) for header in all_headers]
        output.write("| " + " | ".join(padded_values) + " |\n")

    return output.getvalue().strip()


def generate_sphinx_tables():
    docs = ""
    categories = ["Parameter", "Input", "Dynamic State", "Algebraic State", "Output"]
    full_list = descriptions()
    for cat in categories:
        cat_list = [{k: v for k, v in item.items() if k != 'type'}
                    for item in full_list if item.get('type') == cat]
        if cat_list:
            docs += f"\n### {cat}s\n\n"
            docs += dict_list_to_aligned_markdown_table(cat_list)
            docs += "\n"
    return docs


__doc__ += generate_sphinx_tables()
