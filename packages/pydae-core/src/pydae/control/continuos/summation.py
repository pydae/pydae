# -*- coding: utf-8 -*-
r"""
Created on Sun June 15 13:20:00 2026

@author: jmmauricio

Two-input weighted summation (SUMMATION).

A purely algebraic block (no dynamic state): produces a per-input
weighted sum of two signals, exposed as a single output.

**Algebraic equation**

$$0 = z - \left(K_1\, u_1 + K_2\, u_2\right)$$

Both gains default to ``+1`` (a plain additive sum). Setting either
gain to ``-1`` turns the block into a subtractor (``z = u_1 - u_2`` or
``z = u_2 - u_1``); intermediate values mix the inputs in arbitrary
proportions. For more than two inputs, chain summation blocks.

The block is commonly used to assemble error / reference signals
(e.g. ``u_ctrl = K_1 * y_meas - K_2 * y_ref``) without having to wire
the combination by hand in the surrounding builder, and pairs
naturally with ``input=`` aliasing -- both inputs can name existing
``h_dict`` expressions, dynamic states, or other algebraic variables
in the surrounding ``dae``.
"""

import io


def descriptions():
    """
    Single source of truth for model parameters, inputs, states, and outputs.
    """
    descriptions_list = []

    # Parameters
    descriptions_list += [{"type": "Parameter", "tex": "K_1", "data": "K_1", "model": "K_1", "default": 1.0, "description": "Gain on input 1", "units": "-"}]
    descriptions_list += [{"type": "Parameter", "tex": "K_2", "data": "K_2", "model": "K_2", "default": 1.0, "description": "Gain on input 2", "units": "-"}]

    # Inputs
    descriptions_list += [{"type": "Input", "tex": "u_1", "data": "u_1", "model": "u_1", "default": 0.0, "description": "Input signal 1", "units": "-"}]
    descriptions_list += [{"type": "Input", "tex": "u_2", "data": "u_2", "model": "u_2", "default": 0.0, "description": "Input signal 2", "units": "-"}]

    # Algebraic States
    descriptions_list += [{"type": "Algebraic State", "tex": "z", "data": "", "model": "z", "default": "", "description": "Weighted sum z = K_1*u_1 + K_2*u_2, exposed in h_dict", "units": "-"}]

    return descriptions_list


def summation(dae, name, backend, data_dict=None, input=None, output=None):
    """Append a two-input weighted summation block to `dae`.

    Adds two global inputs ``u_1_summation_{name}`` /
    ``u_2_summation_{name}`` (when ``input`` is None), per-input gains
    ``K_1_summation_{name}`` / ``K_2_summation_{name}``, and an
    algebraic output ``z_summation_{name}`` exposed through `h_dict`.

    Parameters
    ----------
    input : list[str] or None
        ``None`` (default) -- create two fresh global inputs.
        ``list[str]`` -- must contain exactly two names. Each is
        resolved via :func:`pydae.control.continuos._common.resolve_input`
        (h_dict expression, existing-symbol alias, or new global input).
        A single string is rejected because the block has two inputs by
        definition; for a single-input scaling block, use a one-element
        list (``input=['x']`` with ``K_2 = 0``).
    output : str or None
        ``None`` (default) -- expose the output as a fresh
        ``z_summation_{name}`` symbol.
        ``str`` -- alias an existing symbol; if that name was a global
        input it is removed from ``u_ini_dict`` / ``u_run_dict`` and
        promoted to an algebraic variable driven by this block.
    """
    from pydae.control.continuos._common import resolve_input

    data_dict = data_dict or {}

    # 1. Defaults (from descriptions())
    meta = descriptions()
    default_map = {item['data']: item['default'] for item in meta if item.get('data')}

    # 2. Data-dict overrides (fall back to descriptions() defaults)
    val_K_1 = data_dict.get('K_1', default_map.get('K_1', 1.0))
    val_K_2 = data_dict.get('K_2', default_map.get('K_2', 1.0))
    val_u_1 = data_dict.get('u_1', default_map.get('u_1', 0.0))
    val_u_2 = data_dict.get('u_2', default_map.get('u_2', 0.0))

    # 3. Inputs: two fresh symbols or two aliased / resolved names.
    if input is None:
        u_1 = backend.symbols(f"u_1_summation_{name}")
        u_2 = backend.symbols(f"u_2_summation_{name}")
        dae['u_ini_dict'][f"{u_1}"] = val_u_1
        dae['u_run_dict'][f"{u_1}"] = val_u_1
        dae['u_ini_dict'][f"{u_2}"] = val_u_2
        dae['u_run_dict'][f"{u_2}"] = val_u_2
        u_1_expr = u_1
        u_2_expr = u_2
    else:
        if isinstance(input, str):
            raise ValueError(
                "summation expects `input` to be a list of two names "
                "(use a one-element list with K_2=0 for a single-input scaling)."
            )
        names = list(input)
        if len(names) != 2:
            raise ValueError(
                f"summation expects 2 input names, got {len(names)}: {names!r}"
            )
        u_1_expr = resolve_input(names[0], dae, backend, val_u_1)
        u_2_expr = resolve_input(names[1], dae, backend, val_u_2)

    # 4. Algebraic state (output): fresh or aliased.
    if output is None:
        z = backend.symbols(f"z_summation_{name}")
        output_aliased = False
    else:
        z = backend.symbols(output)
        output_aliased = True

    # 5. Parameters
    K_1 = backend.symbols(f"K_1_summation_{name}")
    K_2 = backend.symbols(f"K_2_summation_{name}")

    # 6. Algebraic equation: z = K_1 * u_1 + K_2 * u_2
    g_z = z - (K_1 * u_1_expr + K_2 * u_2_expr)

    # 7. Promote an aliased output (was a global input) to algebraic state.
    if output_aliased:
        dae['u_ini_dict'].pop(str(z), None)
        dae['u_run_dict'].pop(str(z), None)

    # 8. Initialization hint: z at the value implied by the data-dict u_i.
    val_z = val_K_1 * val_u_1 + val_K_2 * val_u_2
    dae['xy_0_dict'][f"{z}"] = val_z

    # 9. Assembly
    dae['g_list'] += [g_z]
    dae['y_ini_list'] += [z]
    dae['y_run_list'] += [z]
    dae['h_dict'].update({f"{z}": z})
    dae['params_dict'].update({f"{K_1}": val_K_1, f"{K_2}": val_K_2})

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
        # Filter by category and remove the 'type' key so it doesn't render as a column
        cat_list = [{k: v for k, v in item.items() if k != 'type'}
                    for item in full_list if item.get('type') == cat]

        if cat_list:
            docs += f"\n### {cat}s\n\n"
            docs += dict_list_to_aligned_markdown_table(cat_list)
            docs += "\n"

    return docs

# Inject the categorized tables into the module's docstring for Sphinx
__doc__ += generate_sphinx_tables()

# =============================================================================
# Testing Block
# =============================================================================
def _build_summation_model(name='1', K_1=1.0, K_2=1.0, u_1_0=0.0, u_2_0=0.0):
    """Assemble a standalone CasadiModel containing the summation block.

    ``summation`` is purely algebraic (no differential equations); the
    CasADi builder needs at least one ``f`` equation to introspect
    symbols, so a tiny dummy ``lpf1`` with a separate, untouched input
    is added in parallel just to give the dae a state. It does not
    interact with the summation output.
    """
    from pydae.core.builder.casadi_builder import MathBackend, CasadiBuilder
    from pydae.core.model.casadi_model import CasadiModel
    from pydae.control.continuos.lpf1 import lpf1

    dae = {'name': f'test_summation_{name}',
           'f_list': [], 'x_list': [],
           'g_list': [], 'y_ini_list': [], 'y_run_list': [],
           'u_ini_dict': {}, 'u_run_dict': {},
           'params_dict': {},
           'h_dict': {},
           'xy_0_dict': {}}

    backend = MathBackend(use_casadi=True)
    lpf1(dae, '_dummy', backend, {'T': 1.0, 'u': 0.0})  # provides one state
    summation(dae, name, backend, {'K_1': K_1, 'K_2': K_2, 'u_1': u_1_0, 'u_2': u_2_0})

    model = CasadiModel(CasadiBuilder(dae).build())
    model.ini({}, dae['xy_0_dict'])

    return model


def test_steady_state_weighted_sum():
    """z = K_1 * u_1 + K_2 * u_2 at the initial operating point."""
    model = _build_summation_model(name='1', K_1=2.0, K_2=-1.5, u_1_0=0.4, u_2_0=0.2)

    expected = 2.0 * 0.4 + (-1.5) * 0.2
    assert abs(model.get_value('z_summation_1') - expected) < 1e-9


def test_input_change_propagates():
    """Changing an input via model.run() should change the output by
    exactly the corresponding K_i * delta_u_i (algebraic, no lag)."""
    name = '1'
    K_1, K_2 = 1.5, -0.5
    model = _build_summation_model(name=name, K_1=K_1, K_2=K_2, u_1_0=0.0, u_2_0=0.0)

    model.run(1.0, {f'u_1_summation_{name}': 2.0, f'u_2_summation_{name}': 1.0})
    model.post()

    expected = K_1 * 2.0 + K_2 * 1.0
    assert abs(model.get_value(f'z_summation_{name}') - expected) < 1e-9


def test_input_aliasing_via_h_dict_expression():
    """`input=` list whose entries refer to h_dict expressions wires the
    block's u_i to those expressions directly, without introducing new
    free symbols. Build a tiny lpf1 + summation chain in one dae:
        lpf1 with u = u_meas_1 produces h_dict['z_lpf1_meas']
        summation(input=['z_lpf1_meas', 'r'], K_1=1, K_2=-1)
            -> z = (filtered u_meas) - r
    """
    from pydae.core.builder.casadi_builder import MathBackend, CasadiBuilder
    from pydae.core.model.casadi_model import CasadiModel
    from pydae.control.continuos.lpf1 import lpf1

    dae = {'name': 'test_summation_alias',
           'f_list': [], 'x_list': [],
           'g_list': [], 'y_ini_list': [], 'y_run_list': [],
           'u_ini_dict': {}, 'u_run_dict': {},
           'params_dict': {}, 'h_dict': {}, 'xy_0_dict': {}}

    backend = MathBackend(use_casadi=True)
    lpf1(dae, 'meas', backend, {'T': 0.05, 'u': 0.7})
    summation(dae, 'err', backend,
              {'K_1': 1.0, 'K_2': -1.0, 'u_1': 0.7, 'u_2': 0.3},
              input=['z_lpf1_meas', 'r'])

    # 'r' was unknown, so the resolver created it as a global input.
    assert 'r' in dae['u_run_dict']

    model = CasadiModel(CasadiBuilder(dae).build())
    model.ini({}, dae['xy_0_dict'])

    # Steady state: z_lpf1_meas = u (= 0.7), r = 0.3 (created with default),
    # so z = 0.7 - 0.3 = 0.4.
    assert abs(model.get_value('z_lpf1_meas') - 0.7) < 1e-9
    assert abs(model.get_value('r') - 0.3) < 1e-9
    assert abs(model.get_value('z_summation_err') - 0.4) < 1e-9


if __name__ == '__main__':
    name, K_1, K_2 = '1', 2.0, -1.5
    model = _build_summation_model(name=name, K_1=K_1, K_2=K_2, u_1_0=0.4, u_2_0=0.2)

    model.report_u()
    model.report_z()
    model.report_params()

    print(f"z = {model.get_value(f'z_summation_{name}'):.6f}  "
          f"(expected {K_1*0.4 + K_2*0.2:.6f})")
