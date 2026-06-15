# -*- coding: utf-8 -*-
r"""
Created on Thu August 10 23:52:55 2026

@author: jmmauricio

Washout filter (WASHOUT).

A single-state high-pass filter block that removes the DC component
of its input while passing transients unchanged. Commonly used as the
first stage of a Power System Stabilizer (PSS), where it must block
the steady-state speed/frequency deviation and only react to dynamic
changes.

**Transfer function**

$$\frac{Z(s)}{U(s)} = \frac{sT_{wo}}{1 + sT_{wo}}$$

**Dynamic equation**

$$\frac{dx}{dt} = \frac{1}{T_{wo}}\left(u - x\right)$$

**Algebraic equation**

$$0 = z - \left(u - x\right)$$

The dynamic state $x$ tracks the (low-pass filtered) input $u$ with
time constant $T_{wo}$. The algebraic variable $z = u - x$ is the
washout output exposed through `h_dict`: at steady state $x = u$, so
$z = 0$ regardless of the operating point; for fast transients
$x \approx \text{const}$, so $z \approx u$ (unity high-frequency gain).
"""

import io


def descriptions():
    """
    Single source of truth for model parameters, inputs, states, and outputs.
    """
    descriptions_list = []

    # Parameters
    descriptions_list += [{"type": "Parameter", "tex": "T_{wo}", "data": "T_wo", "model": "T_wo", "default": 10.0, "description": "Washout time constant", "units": "s"}]

    # Inputs
    descriptions_list += [{"type": "Input", "tex": "u", "data": "u", "model": "u", "default": 0.0, "description": "Washout input signal", "units": "-"}]

    # Dynamic States
    descriptions_list += [{"type": "Dynamic State", "tex": "x", "data": "", "model": "x", "default": "", "description": "Low-pass tracking state of u", "units": "-"}]

    # Algebraic States
    descriptions_list += [{"type": "Algebraic State", "tex": "z", "data": "", "model": "z", "default": "", "description": "Washout output (z = u - x, exposed in h_dict)", "units": "-"}]

    return descriptions_list


def washout(dae, name, backend, data_dict=None, input=None, output=None):
    """Append a washout (high-pass) filter block to `dae`.

    Adds one dynamic state ``x_washout_{name}`` (a low-pass tracker of
    the input), driven by ``u_washout_{name}`` with time constant
    ``T_wo_washout_{name}``, plus an algebraic output
    ``z_washout_{name} = u - x`` that is exposed through `h_dict`.

    See :mod:`pydae.control.continuos._common` for the input/output
    aliasing semantics.
    """
    from pydae.control.continuos._common import combine_inputs

    data_dict = data_dict or {}

    # 1. Defaults (from descriptions())
    meta = descriptions()
    default_map = {item['data']: item['default'] for item in meta if item.get('data')}

    # 2. Data-dict overrides (fall back to descriptions() defaults)
    val_T_wo = data_dict.get('T_wo', default_map.get('T_wo', 10.0))
    val_u    = data_dict.get('u',    default_map.get('u',    0.0))

    # 3. Input
    if input is None:
        u = backend.symbols(f"u_washout_{name}")
        dae['u_ini_dict'][f"{u}"] = val_u
        dae['u_run_dict'][f"{u}"] = val_u
        u_expr = u
    else:
        u_expr = combine_inputs(input, dae, backend, val_u)

    # 4. Dynamic state
    x = backend.symbols(f"x_washout_{name}")

    # 5. Algebraic state (output)
    if output is None:
        z = backend.symbols(f"z_washout_{name}")
        output_aliased = False
    else:
        z = backend.symbols(output)
        output_aliased = True

    # 6. Parameter
    T_wo = backend.symbols(f"T_wo_washout_{name}")

    # 7. Equations: dx/dt = (u - x) / T_wo,  z = u - x
    dx  = (u_expr - x) / T_wo
    g_z = z - (u_expr - x)

    # 8. Promote an aliased output (was a global input) to algebraic state.
    if output_aliased:
        dae['u_ini_dict'].pop(str(z), None)
        dae['u_run_dict'].pop(str(z), None)

    # 9. Initialization hint: start at steady state (x = u, z = 0)
    dae['xy_0_dict'].update({f"{x}": val_u, f"{z}": 0.0})

    # 10. Assembly
    dae['f_list'] += [dx]
    dae['x_list'] += [x]
    dae['g_list'] += [g_z]
    dae['y_ini_list'] += [z]
    dae['y_run_list'] += [z]
    dae['h_dict'].update({f"{z}": z})
    dae['params_dict'].update({f"{T_wo}": val_T_wo})

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
def _build_washout_model(name='1', T_wo=2.0, u_0=0.0):
    """Assemble a standalone CasadiModel containing a single WASHOUT block."""
    from pydae.core.builder.casadi_builder import MathBackend, CasadiBuilder
    from pydae.core.model.casadi_model import CasadiModel

    dae = {'name': f'test_washout_{name}',
           'f_list': [], 'x_list': [],
           'g_list': [], 'y_ini_list': [], 'y_run_list': [],
           'u_ini_dict': {}, 'u_run_dict': {},
           'params_dict': {},
           'h_dict': {},
           'xy_0_dict': {}}

    backend = MathBackend(use_casadi=True)
    washout(dae, name, backend, {'T_wo': T_wo, 'u': u_0})

    model = CasadiModel(CasadiBuilder(dae).build())
    model.ini({}, dae['xy_0_dict'])

    return model


def test_steady_state_output_is_zero():
    """At steady state (du/dt = 0), the washout output z is zero,
    regardless of the operating-point value of u."""
    model = _build_washout_model(name='1', T_wo=2.0, u_0=0.7)

    assert abs(model.get_value('z_washout_1') - 0.0) < 1e-6
    assert abs(model.get_value('x_washout_1') - 0.7) < 1e-6


def test_step_response_decays_to_zero():
    """A step in u produces an immediate unity-gain jump in z, which
    then decays to zero with time constant T_wo.
    """
    import numpy as np

    name, T_wo = '1', 2.0
    model = _build_washout_model(name=name, T_wo=T_wo, u_0=0.0)

    model.run(5 * T_wo, {f'u_washout_{name}': 1.0})
    model.post()

    z = model.get_values(f'z_washout_{name}')

    assert abs(max(z) - 1.0) < 0.02  # initial unity-gain jump
    assert abs(z[-1] - 0.0) < 0.01   # washed out after 5*T_wo


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    name, T_wo = '1', 2.0
    model = _build_washout_model(name=name, T_wo=T_wo, u_0=0.0)

    model.report_u()
    model.report_x()
    model.report_z()
    model.report_params()

    model.run(5 * T_wo, {f'u_washout_{name}': 1.0})
    model.post()

    fig, ax = plt.subplots()
    ax.plot(model.Time, model.get_values(f'z_washout_{name}'), label='$z$ (washout output)')
    ax.axhline(0.0, color='k', ls='--', lw=0.8, label='steady state')
    ax.axvline(T_wo, color='gray', ls=':', lw=0.8, label='$t = T_{wo}$')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Output')
    ax.legend()
    fig.savefig('washout_step_response.svg')
    print("Test completed. Plot saved as 'washout_step_response.svg'.")
