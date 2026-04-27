# -*- coding: utf-8 -*-
"""
Post-ini() validator for pydae-bps systems.

Compares model results against reference values stored in the ``results:``
section of the system HJSON (see :mod:`reporter` for the plain reporter).

Three public functions:

    validate_buses(model, data)   — bus V and θ vs results.buses
    validate_gens(model, data)    — generator P, Q vs results.syns
    validate_all(model, data)     — both in sequence, returns overall pass/fail

*data* may be any of:
    - a BpsBuilder instance  (uses .data attribute)
    - a parsed dict
    - a path string to a .hjson or .json file

Expected ``results:`` schema in HJSON::

    results: {
      buses: [
        {name: "1", V_pu: 1.040, theta_rad: 0.420},
        ...
      ],
      syns: [
        {bus: "1", P_MW: 1000.0, Q_Mvar: 150.0},
        ...
      ]
    }

Tolerances (all keyword arguments with defaults):

    v_atol      absolute tolerance on V (pu),           default 0.01
    th_atol     absolute tolerance on θ (deg),           default 3.0
    p_rtol      relative tolerance on P (fraction 0–1),  default 0.05
    q_rtol      relative tolerance on Q (fraction 0–1),  default 0.10
"""

import math
import json
import os
from pydae.bps.utils.reporter import _load_data, _get, _md_table


# ─────────────────────────────────────────────────────────────────────────────
# Formatting helpers
# ─────────────────────────────────────────────────────────────────────────────

def _isnan(v):
    """Robust NaN / missing check: handles None, numpy scalars, and Python floats."""
    if v is None:
        return True
    try:
        return math.isnan(float(v))
    except (TypeError, ValueError):
        return True


def _to_float(v):
    """Convert *v* to a Python float, returning nan on failure."""
    if v is None:
        return float('nan')
    try:
        return float(v)
    except (TypeError, ValueError):
        return float('nan')


_PASS = '✓'
_FAIL = '✗'


def _pct(model_val, ref_val):
    """Relative error in percent. Returns nan if ref is zero."""
    if abs(ref_val) < 1e-12:
        return float('nan')
    return (model_val - ref_val) / abs(ref_val) * 100.0


def _abs_err(model_val, ref_val):
    return model_val - ref_val


def _status(ok):
    return _PASS if ok else _FAIL


# ─────────────────────────────────────────────────────────────────────────────
# Public functions
# ─────────────────────────────────────────────────────────────────────────────

def validate_buses(model, data,
                   v_atol=0.01,
                   th_atol=3.0,
                   print_table=True):
    """
    Compare bus voltages and angles against ``results.buses`` in *data*.

    Parameters
    ----------
    v_atol  : float
        Absolute tolerance for voltage magnitude (pu).  Default 0.01.
    th_atol : float
        Absolute tolerance for voltage angle (degrees).  Default 3.0°.

    Returns
    -------
    tuple (table_str, all_passed)
    """
    data_dict  = _load_data(data)
    results    = data_dict.get('results', {})
    ref_buses  = {str(b['name']): b for b in results.get('buses', [])}

    if not ref_buses:
        msg = '_No reference bus data found in results section._'
        if print_table:
            print('\n### Bus Validation\n\n' + msg)
        return msg, True

    headers = ['Bus', 'V_ref (pu)', 'V_model (pu)', 'ΔV (pu)', 'V',
                       'θ_ref (°)',  'θ_model (°)',  'Δθ (°)', 'θ', 'Overall']
    rows      = []
    all_ok    = True

    for name, ref in sorted(ref_buses.items(), key=lambda x: x[0]):
        v_ref  = _to_float(ref.get('V_pu'))
        # Accept theta_deg (takes priority) or theta_rad; both → degrees for comparison
        if 'theta_deg' in ref:
            th_ref_d = _to_float(ref['theta_deg'])
        elif 'theta_rad' in ref:
            th_ref_d = math.degrees(_to_float(ref['theta_rad']))
        else:
            th_ref_d = float('nan')

        v_mod  = _to_float(_get(model, f'V_{name}'))
        th_mod = _to_float(_get(model, f'theta_{name}'))
        th_mod_d = math.degrees(th_mod) if not _isnan(th_mod) else float('nan')

        dv    = _abs_err(v_mod, v_ref)       if not _isnan(v_mod)   else float('nan')
        dth   = _abs_err(th_mod_d, th_ref_d) if not _isnan(th_mod_d) else float('nan')

        v_ok  = (abs(dv)  <= v_atol)  if not _isnan(dv)  else False
        th_ok = (abs(dth) <= th_atol) if not _isnan(dth) else False
        row_ok = v_ok and th_ok
        all_ok = all_ok and row_ok

        rows.append([
            name,
            v_ref,     v_mod,   dv,    _status(v_ok),
            th_ref_d,  th_mod_d, dth,  _status(th_ok),
            _status(row_ok),
        ])

    fmts = ['{}',
            '{:.4f}', '{:.4f}', '{:+.4f}', '{}',
            '{:.3f}', '{:.3f}', '{:+.3f}', '{}',
            '{}']
    table = _md_table(headers, rows, fmts)

    summary = f'\n> Tolerances: V ≤ {v_atol} pu | θ ≤ {th_atol}°  ' \
              f'— Overall: **{"PASS" if all_ok else "FAIL"}**'

    if print_table:
        print('\n### Bus Validation\n')
        print(table)
        print(summary)

    return table + '\n' + summary, all_ok


def validate_gens(model, data,
                  p_rtol=0.05,
                  q_rtol=0.10,
                  print_table=True):
    """
    Compare generator active and reactive powers against ``results.syns``.

    The reference entry may use either ``bus`` or ``name`` to identify the
    generator (``name`` takes precedence when both are present).

    Parameters
    ----------
    p_rtol : float
        Relative tolerance for P (fraction, 0–1).  Default 0.05 (5 %).
    q_rtol : float
        Relative tolerance for Q (fraction, 0–1).  Default 0.10 (10 %).

    Returns
    -------
    tuple (table_str, all_passed)
    """
    data_dict = _load_data(data)
    S_base    = data_dict.get('system', {}).get('S_base', 100e6)
    results   = data_dict.get('results', {})
    ref_list  = results.get('syns', [])

    if not ref_list:
        msg = '_No reference generator data found in results section._'
        if print_table:
            print('\n### Generator Validation\n\n' + msg)
        return msg, True

    # Build lookup: name → S_n for all syns
    sn_map = {}
    for syn in data_dict.get('syns', []):
        key = syn.get('name', str(syn['bus']))
        sn_map[key] = syn.get('S_n', S_base)

    headers = ['Gen', 'P_ref (MW)', 'P_model (MW)', 'ΔP (%)', 'P',
                      'Q_ref (Mvar)', 'Q_model (Mvar)', 'ΔQ (%)', 'Q', 'Overall']
    rows   = []
    all_ok = True

    for ref in ref_list:
        # resolve generator name
        gen_name = ref.get('name', ref.get('bus'))
        gen_name = str(gen_name)
        bus_name = str(ref.get('bus', gen_name))

        p_ref = ref.get('P_MW',   float('nan'))
        q_ref = ref.get('Q_Mvar', float('nan'))

        s_n   = sn_map.get(gen_name, sn_map.get(bus_name, S_base))
        s_mva = s_n / 1e6

        p_g_pu = _to_float(_get(model, f'p_g_{gen_name}'))
        q_g_pu = _to_float(_get(model, f'q_g_{gen_name}'))
        p_mod  = p_g_pu * s_mva if not _isnan(p_g_pu) else float('nan')
        q_mod  = q_g_pu * s_mva if not _isnan(q_g_pu) else float('nan')

        dp_pct = _pct(p_mod, p_ref) if not _isnan(p_mod) else float('nan')
        dq_pct = _pct(q_mod, q_ref) if not _isnan(q_mod) else float('nan')

        p_ok   = (abs(dp_pct) <= p_rtol * 100) if not _isnan(dp_pct) else False
        q_ok   = (abs(dq_pct) <= q_rtol * 100) if not _isnan(dq_pct) else False
        row_ok = p_ok and q_ok
        all_ok = all_ok and row_ok

        rows.append([
            gen_name,
            p_ref,  p_mod,  dp_pct, _status(p_ok),
            q_ref,  q_mod,  dq_pct, _status(q_ok),
            _status(row_ok),
        ])

    fmts = ['{}',
            '{:.1f}', '{:.1f}', '{:+.2f}', '{}',
            '{:.1f}', '{:.1f}', '{:+.2f}', '{}',
            '{}']
    table   = _md_table(headers, rows, fmts)
    summary = f'\n> Tolerances: P ≤ {p_rtol*100:.0f}% | Q ≤ {q_rtol*100:.0f}%  ' \
              f'— Overall: **{"PASS" if all_ok else "FAIL"}**'

    if print_table:
        print('\n### Generator Validation\n')
        print(table)
        print(summary)

    return table + '\n' + summary, all_ok


def validate_all(model, data,
                 v_atol=0.01, th_atol=3.0,
                 p_rtol=0.05, q_rtol=0.10,
                 print_table=True):
    """
    Run bus and generator validation in sequence.

    Returns ``True`` if both pass within the specified tolerances.
    """
    _, buses_ok = validate_buses(model, data,
                                 v_atol=v_atol, th_atol=th_atol,
                                 print_table=print_table)
    _, gens_ok  = validate_gens(model, data,
                                p_rtol=p_rtol, q_rtol=q_rtol,
                                print_table=print_table)
    overall = buses_ok and gens_ok
    if print_table:
        print(f'\n**Validation overall: {"PASS ✓" if overall else "FAIL ✗"}**\n')
    return overall
