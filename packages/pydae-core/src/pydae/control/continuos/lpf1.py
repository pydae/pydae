# -*- coding: utf-8 -*-
r"""
Created on Thu August 10 23:52:55 2026

@author: jmmauricio

First-order continuous low-pass filter (LPF1).

A single-state linear filter block, commonly used to smooth a
measurement (voltage, frequency, power, ...) before it is consumed by
a controller.

**Transfer function**

$$\frac{Z(s)}{U(s)} = \frac{1}{1 + sT}$$

**Dynamic equation**

$$\frac{dx}{dt} = \frac{1}{T}\left(u - x\right)$$

**Algebraic equation**

$$0 = z - x$$

The dynamic state $x$ already holds the filtered value of the input
$u$. The algebraic variable $z$ is an alias of $x$ exposed through
`h_dict`, so other components/observers can read the filter output by
name without depending on the internal state symbol.
"""

import io


def descriptions():
    """
    Single source of truth for model parameters, inputs, states, and outputs.
    """
    descriptions_list = []

    # Parameters
    descriptions_list += [{"type": "Parameter", "tex": "T", "data": "T", "model": "T", "default": 0.1, "description": "Filter time constant", "units": "s"}]

    # Inputs
    descriptions_list += [{"type": "Input", "tex": "u", "data": "u", "model": "u", "default": 0.0, "description": "Filter input signal", "units": "-"}]

    # Dynamic States
    descriptions_list += [{"type": "Dynamic State", "tex": "x", "data": "", "model": "x", "default": "", "description": "Filter state (filtered value of u)", "units": "-"}]

    # Algebraic States
    descriptions_list += [{"type": "Algebraic State", "tex": "z", "data": "", "model": "z", "default": "", "description": "Filtered output (alias of x, exposed in h_dict)", "units": "-"}]

    return descriptions_list


def lpf1(dae, name, data_dict, backend):
    """Append a first-order low-pass filter block to `dae`.

    Adds one dynamic state ``x_lpf1_{name}`` (the filtered value of the
    input), driven by ``u_lpf1_{name}`` with time constant
    ``T_lpf1_{name}``, plus an algebraic output ``z_lpf1_{name}`` that
    mirrors the state and is exposed through `h_dict`.
    """

    # 1. Defaults (from descriptions())
    meta = descriptions()
    default_map = {item['data']: item['default'] for item in meta if item.get('data')}

    # 2. Input
    u = backend.symbols(f"u_lpf1_{name}")

    # 3. Dynamic state
    x = backend.symbols(f"x_lpf1_{name}")

    # 4. Algebraic state (output alias)
    z = backend.symbols(f"z_lpf1_{name}")

    # 5. Parameter
    T = backend.symbols(f"T_lpf1_{name}")

    # 6. Dynamic equation: dx/dt = (u - x) / T
    dx = (u - x) / T

    # 7. Algebraic equation: z = x
    g_z = z - x

    # 8. Data-dict overrides (fall back to descriptions() defaults)
    val_T = data_dict.get('T', default_map.get('T', 0.1))
    val_u = data_dict.get('u', default_map.get('u', 0.0))

    # 9. Initialization hint: start at steady state (x = u)
    dae['xy_0_dict'].update({f"{x}": val_u})

    # 10. Assembly
    dae['f_list'] += [dx]
    dae['x_list'] += [x]
    dae['g_list'] += [g_z]
    dae['y_ini_list'] += [z]
    dae['y_run_list'] += [z]
    dae['h_dict'].update({f"{z}": z})  # Output
    dae['u_ini_dict'].update({f"{u}": val_u})
    dae['u_run_dict'].update({f"{u}": val_u})
    dae['params_dict'].update({f"{T}": val_T})

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
def _build_lpf1_model(name='1', T=0.5, u_0=0.0):
    """Assemble a standalone CasadiModel containing a single LPF1 block."""
    from pydae.core.builder.casadi_builder import MathBackend, CasadiBuilder
    from pydae.core.model.casadi_model import CasadiModel

    dae = {'name': f'test_lpf1_{name}',
           'f_list': [], 'x_list': [],
           'g_list': [], 'y_ini_list': [], 'y_run_list': [],
           'u_ini_dict': {}, 'u_run_dict': {},
           'params_dict': {},
           'h_dict': {},
           'xy_0_dict': {}}

    backend = MathBackend(use_casadi=True)
    lpf1(dae, name, {'T': T, 'u': u_0}, backend)

    model = CasadiModel(CasadiBuilder(dae).build())
    model.ini({}, dae['xy_0_dict'])

    return model


def test_steady_state_tracks_input():
    """At steady state (du/dt = 0), the filter output z equals the input u."""
    model = _build_lpf1_model(name='1', T=0.5, u_0=0.3)

    assert abs(model.get_value('z_lpf1_1') - 0.3) < 1e-6
    assert abs(model.get_value('x_lpf1_1') - 0.3) < 1e-6


def test_step_response_time_constant():
    """A step in u should produce the classic first-order response:
    ~63.2% of the step amplitude after one time constant T, settling
    to the new input value at steady state (t >> T).
    """
    import numpy as np

    name, T = '1', 0.5
    model = _build_lpf1_model(name=name, T=T, u_0=0.0)

    model.run(5 * T, {f'u_lpf1_{name}': 1.0})
    model.post()

    t = model.Time
    z = model.get_values(f'z_lpf1_{name}')

    idx_T = np.argmin(np.abs(t - T))
    assert abs(z[idx_T] - (1 - np.exp(-1))) < 0.02  # ~63.2% after one T
    assert abs(z[-1] - 1.0) < 0.01                   # settled after 5T


if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt

    name, T = '1', 0.5
    model = _build_lpf1_model(name=name, T=T, u_0=0.0)

    model.report_u()
    model.report_x()
    model.report_z()
    model.report_params()

    model.run(5 * T, {f'u_lpf1_{name}': 1.0})
    model.post()

    fig, ax = plt.subplots()
    ax.plot(model.Time, model.get_values(f'z_lpf1_{name}'), label='$z$ (filtered output)')
    ax.axhline(1.0, color='k', ls='--', lw=0.8, label='$u$ (step input)')
    ax.axvline(T, color='gray', ls=':', lw=0.8, label='$t = T$')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Output')
    ax.legend()
    fig.savefig('lpf1_step_response.svg')
    print("Test completed. Plot saved as 'lpf1_step_response.svg'.")
