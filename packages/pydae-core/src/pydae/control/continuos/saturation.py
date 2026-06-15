# -*- coding: utf-8 -*-
r"""
Created on Sun June 15 13:59:00 2026

@author: jmmauricio

Saturation (clip) block (SATURATION).

A purely algebraic clamp

$$z = \mathrm{clip}\!\left(u,\; z_{min},\; z_{max}\right)$$

with one input, one output, no dynamic state. Both limits may be
asymmetric (use a very large magnitude on the unconstrained side
for a one-sided saturation).

Equivalent to a single call to ``backend.hard_limits(u, z_min, z_max)``
inline, but the block form lets the user wire it declaratively into a
``ControlBuilder`` HJSON description and chain its output through
``output=`` aliasing.
"""

import io


def descriptions():
    """
    Single source of truth for model parameters, inputs, states, and outputs.
    """
    descriptions_list = []

    descriptions_list += [{"type": "Parameter", "tex": "z_{min}", "data": "z_min", "model": "z_min", "default": -1.0, "description": "Lower output limit", "units": "-"}]
    descriptions_list += [{"type": "Parameter", "tex": "z_{max}", "data": "z_max", "model": "z_max", "default":  1.0, "description": "Upper output limit", "units": "-"}]

    descriptions_list += [{"type": "Input", "tex": "u", "data": "u", "model": "u", "default": 0.0, "description": "Saturation input signal", "units": "-"}]

    descriptions_list += [{"type": "Algebraic State", "tex": "z", "data": "", "model": "z", "default": "", "description": "Saturated output, clamped to [z_min, z_max] (exposed in h_dict)", "units": "-"}]

    return descriptions_list


def saturation(dae, name, backend, data_dict=None, input=None, output=None):
    """Append a saturation block to `dae`.

    See :mod:`pydae.control.continuos._common` for the input/output
    aliasing semantics. A ``list[str]`` input is combined as
    ``u = input[0] - sum(input[1:])`` (the conventional error-signal
    convention) before being clipped.
    """
    from pydae.control.continuos._common import combine_inputs

    data_dict = data_dict or {}

    meta = descriptions()
    default_map = {item['data']: item['default'] for item in meta if item.get('data')}
    val_z_min = data_dict.get('z_min', default_map.get('z_min', -1.0))
    val_z_max = data_dict.get('z_max', default_map.get('z_max',  1.0))
    val_u     = data_dict.get('u',     default_map.get('u',      0.0))

    if input is None:
        u = backend.symbols(f"u_saturation_{name}")
        dae['u_ini_dict'][f"{u}"] = val_u
        dae['u_run_dict'][f"{u}"] = val_u
        u_expr = u
    else:
        u_expr = combine_inputs(input, dae, backend, val_u)

    if output is None:
        z = backend.symbols(f"z_saturation_{name}")
        output_aliased = False
    else:
        z = backend.symbols(output)
        output_aliased = True

    z_min = backend.symbols(f"z_min_saturation_{name}")
    z_max = backend.symbols(f"z_max_saturation_{name}")

    g_z = z - backend.hard_limits(u_expr, z_min, z_max)

    if output_aliased:
        dae['u_ini_dict'].pop(str(z), None)
        dae['u_run_dict'].pop(str(z), None)

    val_z = min(max(val_u, val_z_min), val_z_max)
    dae['xy_0_dict'][f"{z}"] = val_z

    dae['g_list'] += [g_z]
    dae['y_ini_list'] += [z]
    dae['y_run_list'] += [z]
    dae['h_dict'].update({f"{z}": z})
    dae['params_dict'].update({f"{z_min}": val_z_min, f"{z_max}": val_z_max})

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
def _build_saturation_model(name='1', z_min=-1.0, z_max=1.0, u_0=0.0):
    """Standalone test model. saturation is purely algebraic, so we
    add a tiny dummy lpf1 in parallel to give the CasADi builder a
    differential equation to chew on."""
    from pydae.core.builder.casadi_builder import MathBackend, CasadiBuilder
    from pydae.core.model.casadi_model import CasadiModel
    from pydae.control.continuos.lpf1 import lpf1

    dae = {'name': f'test_saturation_{name}',
           'f_list': [], 'x_list': [],
           'g_list': [], 'y_ini_list': [], 'y_run_list': [],
           'u_ini_dict': {}, 'u_run_dict': {},
           'params_dict': {}, 'h_dict': {}, 'xy_0_dict': {}}
    backend = MathBackend(use_casadi=True)
    lpf1(dae, '_dummy', backend, {'T': 1.0, 'u': 0.0})
    saturation(dae, name, backend, {'z_min': z_min, 'z_max': z_max, 'u': u_0})

    model = CasadiModel(CasadiBuilder(dae).build())
    model.ini({}, dae['xy_0_dict'])
    return model


def test_passes_through_inside_limits():
    """For u in [z_min, z_max] the output equals the input."""
    model = _build_saturation_model(name='1', z_min=-1.0, z_max=1.0, u_0=0.3)
    assert abs(model.get_value('z_saturation_1') - 0.3) < 1e-9


def test_clamps_above_z_max():
    model = _build_saturation_model(name='1', z_min=-1.0, z_max=1.0, u_0=2.5)
    assert abs(model.get_value('z_saturation_1') - 1.0) < 1e-9


def test_clamps_below_z_min():
    model = _build_saturation_model(name='1', z_min=-1.0, z_max=1.0, u_0=-2.5)
    assert abs(model.get_value('z_saturation_1') - (-1.0)) < 1e-9


if __name__ == '__main__':
    model = _build_saturation_model(name='1', u_0=2.0)
    print(f"z = {model.get_value('z_saturation_1'):.4f}")
