# -*- coding: utf-8 -*-
r"""
Created on Sun June 15 13:59:30 2026

@author: jmmauricio

Deadband block (DEADBAND).

A purely algebraic block that suppresses small inputs and passes
larger ones with slope 1:

$$z = \max(u - db_{high},\, 0) + \min(u - db_{low},\, 0)$$

Inside the deadband ($db_{low} \le u \le db_{high}$) both terms
vanish and $z = 0$. Outside the deadband the output rises with
unit slope from zero (no jump at the edges), so a continuous
controller does not see a discontinuity when the input first
exits the deadband.

The deadband may be asymmetric -- $db_{low} \le 0 \le db_{high}$
is the only constraint. Default is a symmetric $\pm 0.01$ band,
matching the typical governor / AGC nominal-frequency deadband
on per-unit speed.
"""

import io


def descriptions():
    """
    Single source of truth for model parameters, inputs, states, and outputs.
    """
    descriptions_list = []

    descriptions_list += [{"type": "Parameter", "tex": "db_{low}",  "data": "db_low",  "model": "db_low",  "default": -0.01, "description": "Deadband lower edge (must be <= 0)", "units": "-"}]
    descriptions_list += [{"type": "Parameter", "tex": "db_{high}", "data": "db_high", "model": "db_high", "default":  0.01, "description": "Deadband upper edge (must be >= 0)", "units": "-"}]

    descriptions_list += [{"type": "Input", "tex": "u", "data": "u", "model": "u", "default": 0.0, "description": "Deadband input signal", "units": "-"}]

    descriptions_list += [{"type": "Algebraic State", "tex": "z", "data": "", "model": "z", "default": "", "description": "Deadband output (zero inside, slope 1 outside, exposed in h_dict)", "units": "-"}]

    return descriptions_list


def deadband(dae, name, backend, data_dict=None, input=None, output=None):
    """Append a deadband block to `dae`.

    See :mod:`pydae.control.continuos._common` for the input/output
    aliasing semantics. A ``list[str]`` input is combined as
    ``u = input[0] - sum(input[1:])`` (the conventional error-signal
    convention) before being fed to the deadband.
    """
    from pydae.control.continuos._common import combine_inputs

    data_dict = data_dict or {}

    meta = descriptions()
    default_map = {item['data']: item['default'] for item in meta if item.get('data')}
    val_db_low  = data_dict.get('db_low',  default_map.get('db_low',  -0.01))
    val_db_high = data_dict.get('db_high', default_map.get('db_high',  0.01))
    val_u       = data_dict.get('u',       default_map.get('u',        0.0))

    if val_db_low > 0:
        raise ValueError("deadband: db_low must be <= 0 (the lower edge of "
                         "the deadband sits at or below zero)")
    if val_db_high < 0:
        raise ValueError("deadband: db_high must be >= 0 (the upper edge of "
                         "the deadband sits at or above zero)")

    if input is None:
        u = backend.symbols(f"u_deadband_{name}")
        dae['u_ini_dict'][f"{u}"] = val_u
        dae['u_run_dict'][f"{u}"] = val_u
        u_expr = u
    else:
        u_expr = combine_inputs(input, dae, backend, val_u)

    if output is None:
        z = backend.symbols(f"z_deadband_{name}")
        output_aliased = False
    else:
        z = backend.symbols(output)
        output_aliased = True

    db_low  = backend.symbols(f"db_low_deadband_{name}")
    db_high = backend.symbols(f"db_high_deadband_{name}")

    # z = max(u - db_high, 0) + min(u - db_low, 0)
    above = backend.max(u_expr - db_high, 0.0)
    below = backend.min(u_expr - db_low,  0.0)
    g_z = z - (above + below)

    if output_aliased:
        dae['u_ini_dict'].pop(str(z), None)
        dae['u_run_dict'].pop(str(z), None)

    # Init hint via the same closed-form expression.
    val_z = max(val_u - val_db_high, 0.0) + min(val_u - val_db_low, 0.0)
    dae['xy_0_dict'][f"{z}"] = val_z

    dae['g_list'] += [g_z]
    dae['y_ini_list'] += [z]
    dae['y_run_list'] += [z]
    dae['h_dict'].update({f"{z}": z})
    dae['params_dict'].update({f"{db_low}": val_db_low, f"{db_high}": val_db_high})

    return None

# =============================================================================
# Sphinx Documentation Auto-Generator
# =============================================================================
def dict_list_to_aligned_markdown_table(data: list[dict]) -> str:
    """Generates the Markdown table for the Sphinx docstring."""
    if not data or not isinstance(data, list): return ""
    dict_data = [item for item in data if isinstance(item, dict)]
    if not dict_data: return ""

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
    """Groups descriptions by 'type' and generates separate tables."""
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

# =============================================================================
# Testing Block
# =============================================================================
def _build_deadband_model(name='1', db_low=-0.01, db_high=0.01, u_0=0.0):
    """Standalone test model with a parallel lpf1 for the dummy state."""
    from pydae.core.builder.casadi_builder import MathBackend, CasadiBuilder
    from pydae.core.model.casadi_model import CasadiModel
    from pydae.control.continuos.lpf1 import lpf1

    dae = {'name': f'test_deadband_{name}',
           'f_list': [], 'x_list': [],
           'g_list': [], 'y_ini_list': [], 'y_run_list': [],
           'u_ini_dict': {}, 'u_run_dict': {},
           'params_dict': {}, 'h_dict': {}, 'xy_0_dict': {}}
    backend = MathBackend(use_casadi=True)
    lpf1(dae, '_dummy', backend, {'T': 1.0, 'u': 0.0})
    deadband(dae, name, backend,
             {'db_low': db_low, 'db_high': db_high, 'u': u_0})
    model = CasadiModel(CasadiBuilder(dae).build())
    model.ini({}, dae['xy_0_dict'])
    return model


def test_zero_inside_deadband():
    """For u in [db_low, db_high] the output is exactly zero."""
    model = _build_deadband_model(name='1', db_low=-0.05, db_high=0.05, u_0=0.02)
    assert abs(model.get_value('z_deadband_1')) < 1e-12


def test_slope_one_above_db_high():
    """z = u - db_high for u > db_high."""
    name = '1'
    db_high = 0.05
    u_step = 0.20
    model = _build_deadband_model(name=name, db_low=-0.05, db_high=db_high, u_0=0.0)
    model.run(1.0, {f'u_deadband_{name}': u_step})
    model.post()
    assert abs(model.get_value(f'z_deadband_{name}') - (u_step - db_high)) < 1e-9


def test_slope_one_below_db_low():
    """z = u - db_low for u < db_low."""
    name = '1'
    db_low = -0.05
    u_step = -0.20
    model = _build_deadband_model(name=name, db_low=db_low, db_high=0.05, u_0=0.0)
    model.run(1.0, {f'u_deadband_{name}': u_step})
    model.post()
    assert abs(model.get_value(f'z_deadband_{name}') - (u_step - db_low)) < 1e-9


def test_asymmetric_band():
    """db_low and db_high need not be opposites."""
    model = _build_deadband_model(name='1', db_low=-0.5, db_high=0.1, u_0=0.3)
    assert abs(model.get_value('z_deadband_1') - (0.3 - 0.1)) < 1e-9


def test_validation_rejects_inverted_band():
    """db_low > 0 or db_high < 0 raises."""
    from pydae.core.builder.casadi_builder import MathBackend
    for kwargs in ({'db_low': 0.1, 'db_high': 0.5}, {'db_low': -0.1, 'db_high': -0.05}):
        dae = {'h_dict': {}, 'g_list': [], 'y_ini_list': [], 'y_run_list': [],
               'u_ini_dict': {}, 'u_run_dict': {}, 'params_dict': {},
               'xy_0_dict': {}, 'x_list': [], 'f_list': []}
        try:
            deadband(dae, 't', MathBackend(use_casadi=True), kwargs)
        except ValueError:
            continue
        raise AssertionError(f"should have rejected: {kwargs}")


if __name__ == '__main__':
    model = _build_deadband_model(name='1', u_0=0.5)
    print(f"z = {model.get_value('z_deadband_1'):.4f}")
