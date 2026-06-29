# -*- coding: utf-8 -*-
"""
Post-ini() reporter for Dynamic Thermal Rating (DTR) lines in pydae-bps.

A DTR line (``lines[]`` entry with a truthy ``dtr`` flag, built by
:func:`pydae.bps.lines.line_dtr.add_line_dtr`) carries an IEEE-738 conductor
heat-balance model. After ``ini()`` (or ``run()`` + ``post()``) this module
renders the thermal operating point as GitHub-flavoured markdown tables.

    report_dtr(model, data)   — conductor temperature, current and heat balance

*data* may be a BpsBuilder instance, a parsed dict, or a path to a
``.hjson`` / ``.json`` file (same contract as
:mod:`pydae.bps.utils.reporter`).

Conductor heat balance (steady state, per unit length, W/m):

    p_j + p_s  =  p_c + p_r          (Joule + solar = convection + radiation)

``p_j``, ``p_s``, ``p_c`` are read from the model ``h_dict``; ``p_r`` (radiation)
is reconstructed here from the Stefan-Boltzmann law because it is not stored.
"""

import math

from pydae.bps.utils.reporter import _load_data, _get, _md_table


# Stefan-Boltzmann constant (matches line_dtr.DTR_Symbolic)
_SIGMA = 5.670373e-8


def _dtr_lines(data_dict, model_filter=None):
    """
    Yield (line_name, bus_j, bus_k) for every DTR line in the data dict.

    *model_filter* (``"overhead"`` or ``"cable"``) restricts to that thermal
    model; ``None`` yields all DTR lines. The model is resolved through the same
    ``dtr_codes`` mechanism the builder uses.
    """
    from pydae.bps.lines.line_dtr import _resolve_dtr_data
    shim = type('G', (), {'data': data_dict})()
    for line in data_dict.get('lines', []):
        if not line.get('dtr'):
            continue
        model = str(_resolve_dtr_data(shim, line).get('model', 'overhead')).lower()
        model = 'cable' if model in ('cable', 'underground', 'iec60853') else 'overhead'
        if model_filter is not None and model != model_filter:
            continue
        bus_j = str(line['bus_j'])
        bus_k = str(line['bus_k'])
        yield f"{bus_j}_{bus_k}", bus_j, bus_k


def _p_radiation(model, name, t_cond, t_amb):
    """Radiated power per metre [W/m] from the conductor (Stefan-Boltzmann)."""
    D = _get(model, f'D_conductor_{name}')
    eps = _get(model, f'Emissivity_{name}')
    if any(math.isnan(x) for x in (D, eps, t_cond, t_amb)):
        return float('nan')
    return math.pi * D * _SIGMA * eps * ((t_cond + 273.15) ** 4 - (t_amb + 273.15) ** 4)


def _r_rise_pct(model, name, t_cond):
    """
    Closed-loop sensitivity: the per-cent increase of the line resistance at the
    operating temperature relative to its 20 °C value.

    ``R(T)/R_20 - 1 = alpha_R * (T - 20)`` — this is exactly the extra series
    resistance (and hence the extra ohmic loss / voltage drop) the network sees
    because the conductor is hot. Returns NaN if the model has no ``alpha_R``
    (e.g. an older open-loop build).
    """
    alpha_R = _get(model, f'alpha_R_{name}')
    if math.isnan(alpha_R) or math.isnan(t_cond):
        return float('nan')
    return 100.0 * alpha_R * (t_cond - 20.0)


def report_dtr(model, data, print_table=True):
    """
    Markdown tables of the DTR conductor thermal state after ini().

    Returns the concatenated markdown string (empty string if the system has no
    DTR lines). Two tables are produced:

    1. **Thermal state** — conductor current, temperature, ambient and rise.
    2. **Heat balance** — Joule / solar / convective / radiative terms [W/m]
       and the residual (≈ 0 at a converged steady state).
    3. **Convection** — wind, dimensionless numbers and AC resistance.

    Covers **overhead** (IEEE-738) DTR lines only; buried-cable (IEC 60853) DTR
    lines are reported by :func:`report_dtr_cable`. When both kinds exist this
    function also appends the cable section.

    Column notes
    ------------
    - ``I_cond`` is the physical conductor current ``I_line * K_dtr`` (A); with
      ``K_dtr = 1`` this equals the per-phase line current.
    - ``R_ac`` is converted from the model's Ω/m to Ω/km.
    """
    data_dict = _load_data(data)
    names = list(_dtr_lines(data_dict, model_filter='overhead'))
    if not names:
        # No overhead lines — defer to the cable report (may itself be empty).
        return report_dtr_cable(model, data, print_table=print_table)

    state_rows, balance_rows, conv_rows = [], [], []
    for name, bus_j, bus_k in names:
        label = f"{bus_j}→{bus_k}"

        t_cond = _get(model, f't_conductor_{name}')
        t_amb  = _get(model, f't_ambient_{name}')
        i_line = _get(model, f'I_{bus_j}_{bus_k}')
        k_dtr  = _get(model, f'K_dtr_{name}')
        i_cond = i_line * k_dtr if not (math.isnan(i_line) or math.isnan(k_dtr)) else float('nan')
        d_t    = t_cond - t_amb if not (math.isnan(t_cond) or math.isnan(t_amb)) else float('nan')

        p_j = _get(model, f'p_j_{name}')
        p_s = _get(model, f'p_s_{name}')
        p_c = _get(model, f'p_c_{name}')
        p_r = _p_radiation(model, name, t_cond, t_amb)
        residual = (p_j + p_s - p_c - p_r
                    if not any(math.isnan(x) for x in (p_j, p_s, p_c, p_r))
                    else float('nan'))

        r_ac = _get(model, f'R_ac_{name}')
        r_ac_km = r_ac * 1e3 if not math.isnan(r_ac) else float('nan')
        dr_pct = _r_rise_pct(model, name, t_cond)   # closed-loop resistance rise

        state_rows.append([label, i_cond, t_cond, t_amb, d_t, r_ac_km, dr_pct])
        balance_rows.append([label, p_j, p_s, p_c, p_r, residual])
        conv_rows.append([label,
                          _get(model, f'v_wind_{name}'),
                          _get(model, f'angle_wind_{name}'),
                          _get(model, f'reynolds_{name}'),
                          _get(model, f'rayleigh_{name}'),
                          _get(model, f'nu_{name}')])

    # ΔR(%) = closed-loop resistance rise at the operating temperature; the
    # extra ohmic loss / voltage drop the network carries because the line is hot.
    t1 = _md_table(
        ['Line', 'I_cond (A)', 'T_cond (°C)', 'T_amb (°C)', 'ΔT (°C)', "R_ac (Ω/km)", 'ΔR (%)'],
        state_rows,
        ['{}', '{:.1f}', '{:.2f}', '{:.1f}', '{:.2f}', '{:.4f}', '{:.2f}'],
    )
    t2 = _md_table(
        ['Line', 'P_joule (W/m)', 'P_solar (W/m)', 'P_conv (W/m)', 'P_rad (W/m)', 'Residual (W/m)'],
        balance_rows,
        ['{}', '{:.3f}', '{:.3f}', '{:.3f}', '{:.3f}', '{:.2e}'],
    )
    t3 = _md_table(
        ['Line', 'v_wind (m/s)', 'angle (°)', 'Reynolds', 'Rayleigh', 'Nusselt'],
        conv_rows,
        ['{}', '{:.2f}', '{:.0f}', '{:.1f}', '{:.1f}', '{:.3f}'],
    )

    out = (f'### DTR (overhead) — thermal state\n\n{t1}\n\n'
           f'### DTR (overhead) — heat balance (steady state: P_joule + P_solar = P_conv + P_rad)\n\n{t2}\n\n'
           f'### DTR (overhead) — convection\n\n{t3}')
    if print_table:
        print('\n' + out)

    # Append the cable section if the system also has buried-cable DTR lines.
    cable_out = report_dtr_cable(model, data, print_table=print_table)
    if cable_out:
        out = out + '\n\n' + cable_out
    return out


def report_dtr_cable(model, data, print_table=True):
    """
    Markdown table of the IEC 60853 two-node **cable** DTR thermal state.

    Columns: Line | I_cond (A) | T_cond (°C) | T_sheath (°C) | T_soil (°C) |
    ΔT (°C) | W_c (W/m) | T1+T4 (K·m/W).

    Returns the markdown string (empty if there are no cable DTR lines).
    The steady-state ladder is ``T_cond = T_soil + W_c·(T1+T4)`` and
    ``T_sheath = T_soil + W_c·T4``.
    """
    data_dict = _load_data(data)
    names = list(_dtr_lines(data_dict, model_filter='cable'))
    if not names:
        return ''

    rows = []
    for name, bus_j, bus_k in names:
        label = f"{bus_j}→{bus_k}"
        t_c = _get(model, f't_conductor_{name}')
        t_s = _get(model, f't_sheath_{name}')
        t_soil = _get(model, f't_soil_{name}')
        d_t = t_c - t_soil if not (math.isnan(t_c) or math.isnan(t_soil)) else float('nan')
        i_c = _get(model, f'I_cond_{name}')
        w_c = _get(model, f'W_c_{name}')
        T1 = _get(model, f'T1_{name}')
        T4 = _get(model, f'T4_{name}')
        T_th = T1 + T4 if not (math.isnan(T1) or math.isnan(T4)) else float('nan')
        dr_pct = _r_rise_pct(model, name, t_c)   # closed-loop resistance rise
        rows.append([label, i_c, t_c, t_s, t_soil, d_t, w_c, T_th, dr_pct])

    table = _md_table(
        ['Line', 'I_cond (A)', 'T_cond (°C)', 'T_sheath (°C)', 'T_soil (°C)',
         'ΔT (°C)', 'W_c (W/m)', 'T1+T4 (K·m/W)', 'ΔR (%)'],
        rows,
        ['{}', '{:.1f}', '{:.2f}', '{:.2f}', '{:.1f}', '{:.2f}', '{:.3f}', '{:.4f}', '{:.2f}'],
    )
    out = f'### DTR (cable, IEC 60853) — thermal state\n\n{table}'
    if print_table:
        print('\n' + out)
    return out
