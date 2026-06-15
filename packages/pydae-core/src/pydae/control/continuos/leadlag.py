# -*- coding: utf-8 -*-
r"""
Created on Thu August 10 23:52:55 2026

@author: jmmauricio

First-order continuous lead-lag compensator (LEADLAG).

A single-state linear block with one zero and one pole, commonly used
to shape the phase/gain of a control loop (e.g. PSS or AVR
compensation stages).

**Transfer function**

$$\frac{Z(s)}{U(s)} = \frac{1 + sT_{num}}{1 + sT_{den}}$$

**Dynamic equation**

$$\frac{dx}{dt} = \frac{1}{T_{den}}\left(u - x\right)$$

**Algebraic equation**

$$0 = z - x - \frac{T_{num}}{T_{den}}\left(u - x\right)$$

$x$ is the state of the lag pole $1/(1+sT_{den})$; the algebraic
variable $z$ combines $x$ with the instantaneous input $u$ to realize
the numerator zero $(1+sT_{num})$. $T_{den}$ must be strictly
positive. Setting $T_{num} = 0$ degenerates this block into a plain
first-order low-pass filter (see ``lpf1``) with time constant
$T_{den}$. Setting $T_{num} = T_{den}$ makes the block transparent
($z = u$).
"""

import io


def descriptions():
    """
    Single source of truth for model parameters, inputs, states, and outputs.
    """
    descriptions_list = []

    # Parameters
    descriptions_list += [{"type": "Parameter", "tex": "T_{num}", "data": "T_num", "model": "T_num", "default": 0.0, "description": "Numerator (lead) time constant", "units": "s"}]
    descriptions_list += [{"type": "Parameter", "tex": "T_{den}", "data": "T_den", "model": "T_den", "default": 0.1, "description": "Denominator (lag) time constant", "units": "s"}]

    # Inputs
    descriptions_list += [{"type": "Input", "tex": "u", "data": "u", "model": "u", "default": 0.0, "description": "Compensator input signal", "units": "-"}]

    # Dynamic States
    descriptions_list += [{"type": "Dynamic State", "tex": "x", "data": "", "model": "x", "default": "", "description": "Lag pole state", "units": "-"}]

    # Algebraic States
    descriptions_list += [{"type": "Algebraic State", "tex": "z", "data": "", "model": "z", "default": "", "description": "Compensator output, exposed in h_dict", "units": "-"}]

    return descriptions_list


def leadlag(dae, name, data_dict, backend):
    """Append a first-order lead-lag compensator block to `dae`.

    Adds one dynamic state ``x_leadlag_{name}`` (the lag-pole state),
    driven by ``u_leadlag_{name}`` with time constant
    ``T_den_leadlag_{name}``, plus an algebraic output
    ``z_leadlag_{name}`` that realizes the numerator zero
    ``T_num_leadlag_{name}`` and is exposed through `h_dict`.
    """

    # 1. Defaults (from descriptions())
    meta = descriptions()
    default_map = {item['data']: item['default'] for item in meta if item.get('data')}

    # 2. Input
    u = backend.symbols(f"u_leadlag_{name}")

    # 3. Dynamic state
    x = backend.symbols(f"x_leadlag_{name}")

    # 4. Algebraic state (output)
    z = backend.symbols(f"z_leadlag_{name}")

    # 5. Parameters
    T_num = backend.symbols(f"T_num_leadlag_{name}")
    T_den = backend.symbols(f"T_den_leadlag_{name}")

    # 6. Dynamic equation: dx/dt = (u - x) / T_den
    dx = (u - x) / T_den

    # 7. Algebraic equation: z = x + (T_num/T_den) * (u - x)
    g_z = z - x - (T_num / T_den) * (u - x)

    # 8. Data-dict overrides (fall back to descriptions() defaults)
    val_T_num = data_dict.get('T_num', default_map.get('T_num', 0.0))
    val_T_den = data_dict.get('T_den', default_map.get('T_den', 0.1))
    val_u = data_dict.get('u', default_map.get('u', 0.0))

    # 9. Initialization hint: start at steady state (x = u, z = u)
    dae['xy_0_dict'].update({f"{x}": val_u, f"{z}": val_u})

    # 10. Assembly
    dae['f_list'] += [dx]
    dae['x_list'] += [x]
    dae['g_list'] += [g_z]
    dae['y_ini_list'] += [z]
    dae['y_run_list'] += [z]
    dae['h_dict'].update({f"{z}": z})  # Output
    dae['u_ini_dict'].update({f"{u}": val_u})
    dae['u_run_dict'].update({f"{u}": val_u})
    dae['params_dict'].update({f"{T_num}": val_T_num, f"{T_den}": val_T_den})

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
def _build_leadlag_model(name='1', T_num=1.0, T_den=0.5, u_0=0.0):
    """Assemble a standalone CasadiModel containing a single LEADLAG block."""
    from pydae.core.builder.casadi_builder import MathBackend, CasadiBuilder
    from pydae.core.model.casadi_model import CasadiModel

    dae = {'name': f'test_leadlag_{name}',
           'f_list': [], 'x_list': [],
           'g_list': [], 'y_ini_list': [], 'y_run_list': [],
           'u_ini_dict': {}, 'u_run_dict': {},
           'params_dict': {},
           'h_dict': {},
           'xy_0_dict': {}}

    backend = MathBackend(use_casadi=True)
    leadlag(dae, name, {'T_num': T_num, 'T_den': T_den, 'u': u_0}, backend)

    model = CasadiModel(CasadiBuilder(dae).build())
    model.ini({}, dae['xy_0_dict'])

    return model


def test_steady_state_is_transparent():
    """At steady state (du/dt = 0), the output z equals the input u
    regardless of T_num and T_den."""
    model = _build_leadlag_model(name='1', T_num=1.0, T_den=0.5, u_0=0.3)

    assert abs(model.get_value('z_leadlag_1') - 0.3) < 1e-6
    assert abs(model.get_value('x_leadlag_1') - 0.3) < 1e-6


def test_step_response_lead_then_decay():
    """A step in u produces an initial jump of size T_num/T_den
    (the lead action), which then decays with time constant T_den to
    the new steady-state value u (z = u).
    """
    import numpy as np

    name, T_num, T_den = '1', 1.0, 0.5
    model = _build_leadlag_model(name=name, T_num=T_num, T_den=T_den, u_0=0.0)

    model.run(5 * T_den, {f'u_leadlag_{name}': 1.0})
    model.post()

    z = model.get_values(f'z_leadlag_{name}')

    assert abs(max(z) - T_num / T_den) < 0.05  # initial lead overshoot
    assert abs(z[-1] - 1.0) < 0.02              # settled after 5*T_den


def test_zero_T_num_matches_lpf1():
    """With T_num = 0 the block degenerates into a plain first-order
    low-pass filter with time constant T_den (no lead overshoot)."""
    name, T_den = '1', 0.5
    model = _build_leadlag_model(name=name, T_num=0.0, T_den=T_den, u_0=0.0)

    model.run(T_den, {f'u_leadlag_{name}': 1.0})
    model.post()

    z = model.get_values(f'z_leadlag_{name}')

    # No lead overshoot: monotonic rise, max value at the end of the run.
    assert z[-1] == max(z)


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    name, T_num, T_den = '1', 1.0, 0.5
    model = _build_leadlag_model(name=name, T_num=T_num, T_den=T_den, u_0=0.0)

    model.report_u()
    model.report_x()
    model.report_z()
    model.report_params()

    model.run(5 * T_den, {f'u_leadlag_{name}': 1.0})
    model.post()

    fig, ax = plt.subplots()
    ax.plot(model.Time, model.get_values(f'z_leadlag_{name}'), label='$z$ (compensator output)')
    ax.axhline(1.0, color='k', ls='--', lw=0.8, label='$u$ (step input)')
    ax.axvline(T_den, color='gray', ls=':', lw=0.8, label='$t = T_{den}$')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Output')
    ax.legend()
    fig.savefig('leadlag_step_response.svg')
    print("Test completed. Plot saved as 'leadlag_step_response.svg'.")
