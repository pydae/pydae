# -*- coding: utf-8 -*-
r"""
Created on Sun June 15 14:01:00 2026

@author: jmmauricio

Product block (PRODUCT).

A purely algebraic two-input multiplier with an optional gain:

$$z = K\,u_1\,u_2$$

The natural building block for nonlinear couplings in
power-systems models: instantaneous power $P = V\cdot I$,
reactive power $Q = V\cdot I\,\sin\delta$, droop products
$K_{droop} \cdot \Delta\omega \cdot P_{rated}$, etc.

Two-input by design (use ``K`` to scale); chain product blocks for
$n$-way multiplications. ``K`` defaults to ``1``.
"""

import io


def descriptions():
    """
    Single source of truth for model parameters, inputs, states, and outputs.
    """
    descriptions_list = []

    descriptions_list += [{"type": "Parameter", "tex": "K", "data": "K", "model": "K", "default": 1.0, "description": "Output scaling gain", "units": "-"}]

    descriptions_list += [{"type": "Input", "tex": "u_1", "data": "u_1", "model": "u_1", "default": 0.0, "description": "Multiplicand 1", "units": "-"}]
    descriptions_list += [{"type": "Input", "tex": "u_2", "data": "u_2", "model": "u_2", "default": 0.0, "description": "Multiplicand 2", "units": "-"}]

    descriptions_list += [{"type": "Algebraic State", "tex": "z", "data": "", "model": "z", "default": "", "description": "z = K*u_1*u_2 (exposed in h_dict)", "units": "-"}]

    return descriptions_list


def product(dae, name, backend, data_dict=None, input=None, output=None):
    """Append a two-input product block to `dae`.

    Parameters
    ----------
    input : list[str] or None
        ``None`` (default) -- create two fresh global inputs
        ``u_1_product_{name}`` and ``u_2_product_{name}``.
        ``list[str]`` -- must contain exactly two names. Each is
        resolved via :func:`pydae.control.continuos._common.resolve_input`
        (h_dict expression, existing-symbol alias, or new global input).
        A single string is rejected because the block has two inputs by
        definition.
    output : str or None
        See :mod:`pydae.control.continuos._common`. Aliasing a name
        that was a global input promotes it to an algebraic variable.
    """
    from pydae.control.continuos._common import resolve_input

    data_dict = data_dict or {}

    meta = descriptions()
    default_map = {item['data']: item['default'] for item in meta if item.get('data')}
    val_K   = data_dict.get('K',   default_map.get('K',   1.0))
    val_u_1 = data_dict.get('u_1', default_map.get('u_1', 0.0))
    val_u_2 = data_dict.get('u_2', default_map.get('u_2', 0.0))

    if input is None:
        u_1 = backend.symbols(f"u_1_product_{name}")
        u_2 = backend.symbols(f"u_2_product_{name}")
        dae['u_ini_dict'][f"{u_1}"] = val_u_1
        dae['u_run_dict'][f"{u_1}"] = val_u_1
        dae['u_ini_dict'][f"{u_2}"] = val_u_2
        dae['u_run_dict'][f"{u_2}"] = val_u_2
        u_1_expr, u_2_expr = u_1, u_2
    else:
        if isinstance(input, str):
            raise ValueError(
                "product expects `input` to be a list of two names "
                "(use a one-element list with K = 0 for a constant block)."
            )
        names = list(input)
        if len(names) != 2:
            raise ValueError(
                f"product expects 2 input names, got {len(names)}: {names!r}"
            )
        u_1_expr = resolve_input(names[0], dae, backend, val_u_1)
        u_2_expr = resolve_input(names[1], dae, backend, val_u_2)

    if output is None:
        z = backend.symbols(f"z_product_{name}")
        output_aliased = False
    else:
        z = backend.symbols(output)
        output_aliased = True

    K = backend.symbols(f"K_product_{name}")

    g_z = z - K * u_1_expr * u_2_expr

    if output_aliased:
        dae['u_ini_dict'].pop(str(z), None)
        dae['u_run_dict'].pop(str(z), None)

    val_z = val_K * val_u_1 * val_u_2
    dae['xy_0_dict'][f"{z}"] = val_z

    dae['g_list'] += [g_z]
    dae['y_ini_list'] += [z]
    dae['y_run_list'] += [z]
    dae['h_dict'].update({f"{z}": z})
    dae['params_dict'].update({f"{K}": val_K})

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
def _build_product_model(name='1', K=1.0, u_1_0=0.0, u_2_0=0.0):
    """Standalone test model with a parallel dummy lpf1 (product is
    purely algebraic and the CasADi builder needs at least one state)."""
    from pydae.core.builder.casadi_builder import MathBackend, CasadiBuilder
    from pydae.core.model.casadi_model import CasadiModel
    from pydae.control.continuos.lpf1 import lpf1

    dae = {'name': f'test_product_{name}',
           'f_list': [], 'x_list': [],
           'g_list': [], 'y_ini_list': [], 'y_run_list': [],
           'u_ini_dict': {}, 'u_run_dict': {},
           'params_dict': {}, 'h_dict': {}, 'xy_0_dict': {}}
    backend = MathBackend(use_casadi=True)
    lpf1(dae, '_dummy', backend, {'T': 1.0, 'u': 0.0})
    product(dae, name, backend, {'K': K, 'u_1': u_1_0, 'u_2': u_2_0})
    model = CasadiModel(CasadiBuilder(dae).build())
    model.ini({}, dae['xy_0_dict'])
    return model


def test_steady_state_product_with_gain():
    model = _build_product_model(name='1', K=3.0, u_1_0=0.4, u_2_0=0.5)
    assert abs(model.get_value('z_product_1') - 3.0 * 0.4 * 0.5) < 1e-9


def test_step_change_propagates_algebraically():
    """A step in either input gives an immediate, lag-free product."""
    name, K = '1', 2.0
    model = _build_product_model(name=name, K=K, u_1_0=0.0, u_2_0=0.0)
    model.run(1.0, {f'u_1_product_{name}': 0.5, f'u_2_product_{name}': 0.6})
    model.post()
    assert abs(model.get_value(f'z_product_{name}') - K * 0.5 * 0.6) < 1e-9


def test_input_aliasing_via_h_dict():
    """Multiply two h_dict expressions (e.g., V * I)."""
    from pydae.core.builder.casadi_builder import MathBackend, CasadiBuilder
    from pydae.core.model.casadi_model import CasadiModel
    from pydae.control.continuos.lpf1 import lpf1

    dae = {'name': 'test_product_alias',
           'f_list': [], 'x_list': [],
           'g_list': [], 'y_ini_list': [], 'y_run_list': [],
           'u_ini_dict': {}, 'u_run_dict': {},
           'params_dict': {}, 'h_dict': {}, 'xy_0_dict': {}}
    backend = MathBackend(use_casadi=True)
    lpf1(dae, 'V', backend, {'T': 0.05, 'u': 1.0})
    lpf1(dae, 'I', backend, {'T': 0.05, 'u': 2.0})
    product(dae, 'P', backend,
            {'K': 1.0, 'u_1': 1.0, 'u_2': 2.0},
            input=['z_lpf1_V', 'z_lpf1_I'])
    model = CasadiModel(CasadiBuilder(dae).build())
    model.ini({}, dae['xy_0_dict'])
    # z = 1.0 * 1.0 * 2.0 = 2.0
    assert abs(model.get_value('z_product_P') - 2.0) < 1e-9


def test_validation_rejects_bad_input():
    from pydae.core.builder.casadi_builder import MathBackend
    for kwargs in (
        dict(input_arg='only_one'),
        dict(input_arg=['a']),
        dict(input_arg=['a', 'b', 'c']),
    ):
        ia = kwargs.pop('input_arg')
        dae = {'h_dict': {}, 'g_list': [], 'y_ini_list': [], 'y_run_list': [],
               'u_ini_dict': {}, 'u_run_dict': {}, 'params_dict': {},
               'xy_0_dict': {}, 'x_list': [], 'f_list': []}
        try:
            product(dae, 't', MathBackend(use_casadi=True), {}, input=ia)
        except ValueError:
            continue
        raise AssertionError(f"should have rejected input={ia!r}")


if __name__ == '__main__':
    model = _build_product_model(name='1', K=2.0, u_1_0=0.4, u_2_0=0.5)
    print(f"z = {model.get_value('z_product_1'):.4f}  (expected {2*0.4*0.5})")
