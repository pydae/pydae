# -*- coding: utf-8 -*-
r"""
GENCLS — classical 2nd-order synchronous machine (PSS/E ``GENCLS``).

The simplest useful machine model: just rotor swing dynamics with a
**constant internal voltage** $E$ behind a **single transient
reactance** $X'$. No field winding, no AVR, no damper windings, no
saturation, no saliency.

Used as a "back-of-the-envelope" machine for first-swing transient
stability studies, for the *infinite-machine equivalent* of distant
generation, and for benchmarks where rotor flux dynamics are not
relevant. Two states: $\delta$, $\omega$. The internal voltage $E$
(stored under the name ``e1q``) is an external input — typically
pinned to the magnitude obtained from a power-flow solution and held
fixed throughout the simulation.

Choose ``gencls`` when you want only the swing equation. Choose
``gensal`` for hydro and ``genrou`` for thermal/turbo where rotor flux
dynamics matter.

**Auxiliar equations**

$$v_d = V \sin(\delta - \theta), \quad v_q = V \cos(\delta - \theta)$$
$$\tau_e = (v_d + R_a i_d)\,i_d + (v_q + R_a i_q)\,i_q$$
$$\omega_s = \omega_{coi}$$

**Dynamic equations**

$$\frac{d\delta}{dt} = \Omega_b (\omega - \omega_s) - K_\delta \delta$$
$$\frac{d\omega}{dt} = \frac{1}{2H}\bigl(\tau_m - \tau_e - D(\omega - \omega_s)\bigr)$$

**Algebraic equations** — constant internal voltage $E = e_q^{(1)}$
behind a single transient reactance $X'$ (no saliency, $X_d' = X_q' = X'$):

$$0 = v_q + R_a i_q - E + X'\,i_d$$
$$0 = v_d + R_a i_d - X'\,i_q$$
$$0 = i_d v_d + i_q v_q - p_g$$
$$0 = i_d v_q - i_q v_d - q_g$$

**Compared with ``milano2ord``** — `milano2ord` lets you set distinct
$X'_d$ and $X'_q$ (i.e. it can model salient-pole transient saliency
even at the classical-model level). `gencls` is the strict PSS/E
`GENCLS` form with a single $X'$. When $X_q' = X_d'$, `milano2ord`
collapses onto `gencls` exactly.
"""

import numpy as np
import io


def descriptions():
    """Single source of truth for parameters, inputs, states, outputs."""
    descriptions_list = []

    # Parameters
    descriptions_list += [{"type": "Parameter", "tex": "S_n",         "data": "S_n",     "model": "S_n",     "default": 100e6, "description": "Nominal power", "units": "VA"}]
    descriptions_list += [{"type": "Parameter", "tex": "F_n",         "data": "F_n",     "model": "F_n",     "default": 50.0,  "description": "Nominal frequency", "units": "Hz"}]
    descriptions_list += [{"type": "Parameter", "tex": "H",           "data": "H",       "model": "H",       "default": 5.0,   "description": "Inertia constant", "units": "s"}]
    descriptions_list += [{"type": "Parameter", "tex": "D",           "data": "D",       "model": "D",       "default": 0.0,   "description": "Damping coefficient", "units": "-"}]
    descriptions_list += [{"type": "Parameter", "tex": "X'",          "data": "X1d",     "model": "X1d",     "default": 0.30,  "description": "Single transient reactance (X'_d = X'_q = X')", "units": "pu-m"}]
    descriptions_list += [{"type": "Parameter", "tex": "R_a",         "data": "R_a",     "model": "R_a",     "default": 0.0,   "description": "Armature resistance", "units": "pu-m"}]
    descriptions_list += [{"type": "Parameter", "tex": "K_{\\delta}", "data": "K_delta", "model": "K_delta", "default": 0.0,   "description": "Reference-machine constant", "units": "-"}]
    descriptions_list += [{"type": "Parameter", "tex": "K_{sec}",     "data": "K_sec",   "model": "K_sec",   "default": 0.0,   "description": "Secondary-frequency control participation", "units": "-"}]

    # Inputs
    descriptions_list += [{"type": "Input", "tex": "p_m",   "data": "p_m", "model": "p_m", "default": 0.5, "description": "Mechanical power", "units": "pu-m"}]
    descriptions_list += [{"type": "Input", "tex": "E",     "data": "e1q", "model": "e1q", "default": 1.0, "description": "Internal voltage magnitude (constant)", "units": "pu-m"}]

    # Dynamic States
    descriptions_list += [{"type": "Dynamic State", "tex": "\\delta", "data": "", "model": "delta", "default": "", "description": "Rotor angle", "units": "rad"}]
    descriptions_list += [{"type": "Dynamic State", "tex": "\\omega", "data": "", "model": "omega", "default": "", "description": "Rotor speed", "units": "pu"}]

    # Algebraic States
    descriptions_list += [{"type": "Algebraic State", "tex": "i_d", "data": "", "model": "i_d", "default": "", "description": "d-axis current", "units": "pu-m"}]
    descriptions_list += [{"type": "Algebraic State", "tex": "i_q", "data": "", "model": "i_q", "default": "", "description": "q-axis current", "units": "pu-m"}]
    descriptions_list += [{"type": "Algebraic State", "tex": "p_g", "data": "", "model": "p_g", "default": "", "description": "Active power injection (machine base)", "units": "pu-m"}]
    descriptions_list += [{"type": "Algebraic State", "tex": "q_g", "data": "", "model": "q_g", "default": "", "description": "Reactive power injection (machine base)", "units": "pu-m"}]

    # Outputs
    descriptions_list += [{"type": "Output", "tex": "p_e", "data": "", "model": "p_e",  "default": "", "description": "Electrical power", "units": "pu-m"}]
    descriptions_list += [{"type": "Output", "tex": "v_d", "data": "", "model": "v_d",  "default": "", "description": "d-axis terminal voltage", "units": "pu-m"}]
    descriptions_list += [{"type": "Output", "tex": "v_q", "data": "", "model": "v_q",  "default": "", "description": "q-axis terminal voltage", "units": "pu-m"}]

    return descriptions_list


def gencls(grid, name, bus_name, data_dict):
    backend = grid.backend
    sin = backend.sin
    cos = backend.cos

    meta = descriptions()
    default_map = {item['data']: item['default'] for item in meta if item.get('data')}

    # Inputs
    V = backend.symbols(f"V_{bus_name}")
    theta = backend.symbols(f"theta_{bus_name}")
    p_m = backend.symbols(f"p_m_{name}")
    e1q = backend.symbols(f"e1q_{name}")
    omega_coi = backend.symbols("omega_coi")

    # Dynamic states (2)
    delta = backend.symbols(f"delta_{name}")
    omega = backend.symbols(f"omega_{name}")

    # Algebraic states
    i_d = backend.symbols(f"i_d_{name}")
    i_q = backend.symbols(f"i_q_{name}")
    p_g = backend.symbols(f"p_g_{name}")
    q_g = backend.symbols(f"q_g_{name}")

    # Parameters
    S_n = backend.symbols(f"S_n_{name}")
    Omega_b = backend.symbols(f"Omega_b_{name}")
    H = backend.symbols(f"H_{name}")
    X1d = backend.symbols(f"X1d_{name}")   # single transient reactance
    D = backend.symbols(f"D_{name}")
    R_a = backend.symbols(f"R_a_{name}")
    K_delta = backend.symbols(f"K_delta_{name}")

    params_list = ['S_n', 'H', 'X1d', 'D', 'R_a', 'K_delta', 'K_sec']

    # Auxiliar
    v_d = V * sin(delta - theta)
    v_q = V * cos(delta - theta)
    tau_e = (v_d + R_a * i_d) * i_d + (v_q + R_a * i_q) * i_q
    omega_s = omega_coi

    # Dynamic equations — just the swing
    ddelta = Omega_b * (omega - omega_s) - K_delta * delta
    domega = 1 / (2 * H) * (p_m - tau_e - D * (omega - omega_s))

    # Algebraic equations — single transient reactance X', constant E
    g_i_d = v_q + R_a * i_q - e1q + X1d * i_d
    g_i_q = v_d + R_a * i_d - X1d * i_q
    g_p_g = i_d * v_d + i_q * v_q - p_g
    g_q_g = i_d * v_q - i_q * v_d - q_g

    f_syn = [ddelta, domega]
    x_syn = [delta, omega]
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

    val_e1q = data_dict.get('e1q', default_map.get('e1q', 1.0))
    grid.dae['u_ini_dict'].update({f'{e1q}': val_e1q})
    grid.dae['u_run_dict'].update({f'{e1q}': val_e1q})

    val_p_m = data_dict.get('p_m', default_map.get('p_m', 0.5))
    grid.dae['u_ini_dict'].update({f'{p_m}': val_p_m})
    grid.dae['u_run_dict'].update({f'{p_m}': val_p_m})

    grid.dae['xy_0_dict'].update({str(omega): 1.0})
    grid.dae['xy_0_dict'].update({str(i_d): 0.5})
    grid.dae['xy_0_dict'].update({str(i_q): 0.5})

    grid.dae['h_dict'].update({f"p_e_{name}": tau_e})
    grid.dae['h_dict'].update({f"v_d_{name}": v_d})
    grid.dae['h_dict'].update({f"v_q_{name}": v_q})

    F_n_val = data_dict.get('F_n', default_map.get('F_n', 50.0))
    grid.dae['params_dict'].update({f"Omega_b_{name}": 2 * np.pi * F_n_val})

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
