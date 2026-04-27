# -*- coding: utf-8 -*-
"""
Post-ini() reporter for pydae-bps systems.

Three functions produce GitHub-flavoured markdown tables printed to stdout
and returned as strings:

    report_buses(model, data)   — bus voltages, angles, load injections
    report_gens(model, data)    — generator P, Q, v_f, p_m, p_g
    report_lines(model, data)   — line / transformer power flows

*data* may be any of:
    - a BpsBuilder instance  (uses .data attribute)
    - a parsed dict
    - a path string to a .hjson or .json file
"""

import math
import json
import os


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_data(data):
    if hasattr(data, 'data'):           # BpsBuilder instance
        return data.data
    if isinstance(data, dict):
        return data
    if isinstance(data, str):
        ext = os.path.splitext(data)[1].lower()
        if ext in ('.hjson', '.json'):
            try:
                import hjson
                with open(data, encoding='utf-8') as f:
                    return hjson.loads(f.read())
            except ImportError:
                with open(data, encoding='utf-8') as f:
                    return json.load(f)
    raise TypeError(f"Cannot load data from {type(data)}")


def _get(model, name, default=float('nan')):
    """Return model.get_value(name) or *default* if the variable does not exist."""
    try:
        return model.get_value(name)
    except Exception:
        return default


def _md_table(headers, rows, fmts=None):
    """
    Render *rows* as a GitHub-flavoured markdown table.

    headers : list[str]
    rows    : list[list]   — each inner list matches headers
    fmts    : list[str]    — optional printf-style format per column
    """
    if fmts is None:
        fmts = ['{}'] * len(headers)
    # format all cells
    def _is_missing(v):
        if v is None:
            return True
        try:
            return math.isnan(float(v))
        except (TypeError, ValueError):
            return False  # non-numeric strings render as-is

    str_rows = []
    for row in rows:
        str_rows.append([fmt.format(v) if not _is_missing(v) else '—'
                         for fmt, v in zip(fmts, row)])
    # column widths
    widths = [len(h) for h in headers]
    for row in str_rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    # render
    sep = '| ' + ' | '.join('-' * w for w in widths) + ' |'
    hdr = '| ' + ' | '.join(h.ljust(w) for h, w in zip(headers, widths)) + ' |'
    lines = [hdr, sep]
    for row in str_rows:
        lines.append('| ' + ' | '.join(c.ljust(w) for c, w in zip(row, widths)) + ' |')
    return '\n'.join(lines)


def _line_g_b_bs(line, buses_dict, S_base):
    """
    Return (G, B, Bs_half) in system pu for a lines[] or transformers[] entry.
    Bs_half = shunt susceptance at each end (from line charging).
    Returns (None, None, None) if the entry cannot be parsed.
    """
    try:
        if 'X_pu' in line:
            S_line = 1e6 * line.get('S_mva', S_base / 1e6)
            R = line['R_pu'] * S_base / S_line
            X = line['X_pu'] * S_base / S_line
        elif 'X_km' in line:
            bj = line['bus_j']
            U_base = buses_dict.get(bj, {}).get('U_kV', 1.0) * 1e3
            Z_base = U_base ** 2 / S_base
            Y_base = 1.0 / Z_base
            km = line['km']
            R = line['R_km'] * km / Z_base
            X = line['X_km'] * km / Z_base
            Bs_total = line.get('Bs_km', 0.0) * km / Y_base
            denom = R ** 2 + X ** 2
            G = R / denom
            B = -X / denom
            return G, B, Bs_total / 2
        else:
            return None, None, None

        denom = R ** 2 + X ** 2
        G = R / denom
        B = -X / denom
        # shunt susceptance (total, halved at each end)
        S_line_for_bs = 1e6 * line.get('S_mva', S_base / 1e6)
        Bs_pu_val = line.get('Bs_pu', line.get('B_pu', 0.0)) * S_base / S_line_for_bs
        return G, B, Bs_pu_val / 2
    except Exception:
        return None, None, None


def _flow_from_to(V_j, th_j, V_k, th_k, G, B, Bs_half, S_base_MW):
    """Return (P_jk_MW, Q_jk_Mvar) — complex power leaving bus j toward bus k."""
    d = th_j - th_k
    P = V_j**2 * G - V_j * V_k * (G * math.cos(d) + B * math.sin(d))
    Q = -V_j**2 * (B + Bs_half) + V_j * V_k * (B * math.cos(d) - G * math.sin(d))
    return P * S_base_MW, Q * S_base_MW


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def report_buses(model, data, print_table=True):
    """
    Markdown table of bus voltages, angles and load injections after ini().

    Columns: Bus | U_kV | V (pu) | θ (°) | P_load (MW) | Q_load (Mvar)

    Load is taken from the *data* dict:
      - ``buses[].P_W / Q_var``  (bus-level injection, negative = load)
      - ``loads[].p_mw / q_mvar`` (ZIP load entries)
    Both contributions are summed; generation is shown in ``report_gens``.
    """
    data_dict = _load_data(data)
    S_base = data_dict.get('system', {}).get('S_base', 100e6)

    # build load lookup: bus_name → (P_load_MW, Q_load_Mvar)
    load_map = {}

    # from buses P_W / Q_var  (negative = consuming)
    for bus in data_dict.get('buses', []):
        name = str(bus['name'])
        p_w  = bus.get('P_W', 0.0)
        q_w  = bus.get('Q_var', 0.0)
        # in pydae convention P_W is the external injection; negative = load
        p_load = -p_w / 1e6   # → MW consumed by load (positive if load)
        q_load = -q_w / 1e6
        if p_load != 0.0 or q_load != 0.0:
            load_map[name] = load_map.get(name, (0.0, 0.0))
            load_map[name] = (load_map[name][0] + p_load,
                              load_map[name][1] + q_load)

    # from loads[] section (p_mw positive = consuming)
    for ld in data_dict.get('loads', []):
        name = str(ld['bus'])
        p_load = ld.get('p_mw', 0.0)
        q_load = ld.get('q_mvar', 0.0)
        load_map[name] = load_map.get(name, (0.0, 0.0))
        load_map[name] = (load_map[name][0] + p_load,
                          load_map[name][1] + q_load)

    rows = []
    for bus in data_dict.get('buses', []):
        name = str(bus['name'])
        u_kv = bus.get('U_kV', float('nan'))
        bus_type = bus.get('type', 'pq')

        v    = _get(model, f'V_{name}')
        th   = _get(model, f'theta_{name}')
        th_d = math.degrees(th) if not math.isnan(th) else float('nan')

        # for slack buses V and theta are inputs (fixed); try to get them anyway
        p_l, q_l = load_map.get(name, (0.0, 0.0))
        rows.append([name, u_kv, bus_type, v, th_d, p_l, q_l])

    headers = ['Bus', 'U_kV', 'Type', 'V (pu)', 'θ (°)', 'P_load (MW)', 'Q_load (Mvar)']
    fmts    = ['{}', '{:.0f}', '{}', '{:.4f}', '{:.3f}', '{:.1f}', '{:.1f}']
    table   = _md_table(headers, rows, fmts)

    if print_table:
        print('\n### Buses\n')
        print(table)
    return table


def report_gens(model, data, print_table=True):
    """
    Markdown table of generator operating points after ini().

    Columns: Gen | Bus | S_n (MVA) | P_g (MW) | Q_g (Mvar) | p_m (pu) | v_f (pu) | V (pu)

    - P_g, Q_g  — active / reactive grid injection (from algebraic states p_g, q_g)
    - p_m       — mechanical power in machine pu (input or algebraic depending on gov)
    - v_f       — field voltage in machine pu (input or algebraic depending on AVR)
    """
    data_dict = _load_data(data)
    S_base    = data_dict.get('system', {}).get('S_base', 100e6)

    rows = []
    for syn in data_dict.get('syns', []):
        bus  = str(syn['bus'])
        name = syn.get('name', bus)
        s_n  = syn.get('S_n', S_base)
        s_mva = s_n / 1e6

        p_g_pu = _get(model, f'p_g_{name}')
        q_g_pu = _get(model, f'q_g_{name}')
        p_m_pu = _get(model, f'p_m_{name}')
        v_f_pu = _get(model, f'v_f_{name}')
        v_bus  = _get(model, f'V_{bus}')

        p_mw   = p_g_pu * s_mva if not math.isnan(p_g_pu) else float('nan')
        q_mvar = q_g_pu * s_mva if not math.isnan(q_g_pu) else float('nan')

        rows.append([name, bus, s_mva, p_mw, q_mvar, p_m_pu, v_f_pu, v_bus])

    headers = ['Gen', 'Bus', 'S_n (MVA)', 'P_g (MW)', 'Q_g (Mvar)',
               'p_m (pu)', 'v_f (pu)', 'V (pu)']
    fmts    = ['{}', '{}', '{:.1f}', '{:.1f}', '{:.1f}',
               '{:.4f}', '{:.4f}', '{:.4f}']
    table   = _md_table(headers, rows, fmts)

    if print_table:
        print('\n### Generators\n')
        print(table)
    return table


def report_lines(model, data, print_table=True):
    """
    Markdown table of line and transformer power flows after ini().

    Columns: From | To | P_jk (MW) | Q_jk (Mvar) | P_kj (MW) | Q_kj (Mvar) | Loss (MW)

    Flows are computed from bus voltages and the line admittance parameters in
    *data*.  Both the ``lines[]`` and ``transformers[]`` sections are processed.
    Entries whose admittance cannot be parsed (missing R/X) are skipped.
    """
    data_dict = _load_data(data)
    S_base    = data_dict.get('system', {}).get('S_base', 100e6)
    S_base_MW = S_base / 1e6

    # fast bus dict lookup: name → {U_kV: ...}
    buses_dict = {str(b['name']): b for b in data_dict.get('buses', [])}

    rows = []
    sections = (
        [(ln, 'line') for ln in data_dict.get('lines', [])] +
        [(tr, 'trafo') for tr in data_dict.get('transformers', [])]
    )

    seen = {}   # count duplicate parallel circuits: (j,k) → count
    for entry, kind in sections:
        j = str(entry['bus_j'])
        k = str(entry['bus_k'])
        key = (j, k)
        idx = seen.get(key, 0) + 1
        seen[key] = idx
        label = f'{j}–{k}' if idx == 1 else f'{j}–{k} #{idx}'

        G, B, Bs_h = _line_g_b_bs(entry, buses_dict, S_base)
        if G is None:
            continue

        V_j  = _get(model, f'V_{j}')
        th_j = _get(model, f'theta_{j}')
        V_k  = _get(model, f'V_{k}')
        th_k = _get(model, f'theta_{k}')

        if any(math.isnan(x) for x in (V_j, th_j, V_k, th_k)):
            rows.append([label, kind, float('nan'), float('nan'),
                         float('nan'), float('nan'), float('nan')])
            continue

        P_jk, Q_jk = _flow_from_to(V_j, th_j, V_k, th_k, G, B, Bs_h, S_base_MW)
        P_kj, Q_kj = _flow_from_to(V_k, th_k, V_j, th_j, G, B, Bs_h, S_base_MW)
        loss = P_jk + P_kj   # algebraic loss (positive = resistive)

        rows.append([label, kind, P_jk, Q_jk, P_kj, Q_kj, loss])

    headers = ['Branch', 'Type', 'P_jk (MW)', 'Q_jk (Mvar)',
               'P_kj (MW)', 'Q_kj (Mvar)', 'Loss (MW)']
    fmts    = ['{}', '{}', '{:.2f}', '{:.2f}', '{:.2f}', '{:.2f}', '{:.2f}']
    table   = _md_table(headers, rows, fmts)

    if print_table:
        print('\n### Lines & Transformers\n')
        print(table)
    return table


def report_all(model, data):
    """Print buses, generators, and line flows in sequence."""
    report_buses(model, data)
    report_gens(model, data)
    report_lines(model, data)
