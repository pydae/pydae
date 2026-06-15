# -*- coding: utf-8 -*-
r"""
Created on Thu August 10 23:52:55 2026

@author: jmmauricio

Second-order continuous low-pass filter (LPF2).

A two-state linear filter block implementing the standard
second-order system transfer function (no zeros), parametrized by its
natural frequency $\omega_n$ and damping ratio $\zeta$. Commonly used
to smooth a measurement with a sharper roll-off than ``lpf1``, or to
model a generic underdamped/overdamped second-order response.

**Transfer function**

$$\frac{Z(s)}{U(s)} = \frac{\omega_n^2}{s^2 + 2\zeta\omega_n s + \omega_n^2}$$

**Dynamic equations** (controllable canonical form)

$$\frac{dx}{dt} = v$$

$$\frac{dv}{dt} = \omega_n^2\left(u - x\right) - 2\zeta\omega_n v$$

**Algebraic equation**

$$0 = z - x$$

The dynamic state $x$ already holds the filtered value of the input
$u$; $v = \dot{x}$ is an auxiliary state with no direct physical
output. The algebraic variable $z$ is an alias of $x$ exposed through
`h_dict`, so other components/observers can read the filter output by
name without depending on the internal state symbol. For $\zeta < 1$
the step response is underdamped (overshoot); for $\zeta \geq 1$ it is
critically damped / overdamped (monotonic).
"""

import io


def descriptions():
    """
    Single source of truth for model parameters, inputs, states, and outputs.
    """
    descriptions_list = []

    # Parameters
    descriptions_list += [{"type": "Parameter", "tex": r"\omega_n", "data": "omega_n", "model": "omega_n", "default": 10.0, "description": "Natural frequency", "units": "rad/s"}]
    descriptions_list += [{"type": "Parameter", "tex": r"\zeta", "data": "zeta", "model": "zeta", "default": 0.7, "description": "Damping ratio", "units": "-"}]

    # Inputs
    descriptions_list += [{"type": "Input", "tex": "u", "data": "u", "model": "u", "default": 0.0, "description": "Filter input signal", "units": "-"}]

    # Dynamic States
    descriptions_list += [{"type": "Dynamic State", "tex": "x", "data": "", "model": "x", "default": "", "description": "Filter state (filtered value of u)", "units": "-"}]
    descriptions_list += [{"type": "Dynamic State", "tex": "v", "data": "", "model": "v", "default": "", "description": "Rate of change of x", "units": "1/s"}]

    # Algebraic States
    descriptions_list += [{"type": "Algebraic State", "tex": "z", "data": "", "model": "z", "default": "", "description": "Filtered output (alias of x, exposed in h_dict)", "units": "-"}]

    return descriptions_list


def lpf2(dae, name, data_dict, backend):
    """Append a second-order low-pass filter block to `dae`.

    Adds two dynamic states ``x_lpf2_{name}`` (the filtered value of
    the input) and ``v_lpf2_{name}`` (its rate of change), driven by
    ``u_lpf2_{name}`` with natural frequency ``omega_n_lpf2_{name}``
    and damping ratio ``zeta_lpf2_{name}``, plus an algebraic output
    ``z_lpf2_{name}`` that mirrors ``x_lpf2_{name}`` and is exposed
    through `h_dict`.
    """

    # 1. Defaults (from descriptions())
    meta = descriptions()
    default_map = {item['data']: item['default'] for item in meta if item.get('data')}

    # 2. Input
    u = backend.symbols(f"u_lpf2_{name}")

    # 3. Dynamic states
    x = backend.symbols(f"x_lpf2_{name}")
    v = backend.symbols(f"v_lpf2_{name}")

    # 4. Algebraic state (output alias)
    z = backend.symbols(f"z_lpf2_{name}")

    # 5. Parameters
    omega_n = backend.symbols(f"omega_n_lpf2_{name}")
    zeta = backend.symbols(f"zeta_lpf2_{name}")

    # 6. Dynamic equations (controllable canonical form):
    #    dx/dt = v
    #    dv/dt = omega_n^2*(u - x) - 2*zeta*omega_n*v
    dx = v
    dv = omega_n**2 * (u - x) - 2 * zeta * omega_n * v

    # 7. Algebraic equation: z = x
    g_z = z - x

    # 8. Data-dict overrides (fall back to descriptions() defaults)
    val_omega_n = data_dict.get('omega_n', default_map.get('omega_n', 10.0))
    val_zeta = data_dict.get('zeta', default_map.get('zeta', 0.7))
    val_u = data_dict.get('u', default_map.get('u', 0.0))

    # 9. Initialization hint: start at steady state (x = u, v = 0)
    dae['xy_0_dict'].update({f"{x}": val_u, f"{v}": 0.0})

    # 10. Assembly
    dae['f_list'] += [dx, dv]
    dae['x_list'] += [x, v]
    dae['g_list'] += [g_z]
    dae['y_ini_list'] += [z]
    dae['y_run_list'] += [z]
    dae['h_dict'].update({f"{z}": z})  # Output
    dae['u_ini_dict'].update({f"{u}": val_u})
    dae['u_run_dict'].update({f"{u}": val_u})
    dae['params_dict'].update({f"{omega_n}": val_omega_n, f"{zeta}": val_zeta})

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
def _build_lpf2_model(name='1', omega_n=10.0, zeta=0.5, u_0=0.0):
    """Assemble a standalone CasadiModel containing a single LPF2 block."""
    from pydae.core.builder.casadi_builder import MathBackend, CasadiBuilder
    from pydae.core.model.casadi_model import CasadiModel

    dae = {'name': f'test_lpf2_{name}',
           'f_list': [], 'x_list': [],
           'g_list': [], 'y_ini_list': [], 'y_run_list': [],
           'u_ini_dict': {}, 'u_run_dict': {},
           'params_dict': {},
           'h_dict': {},
           'xy_0_dict': {}}

    backend = MathBackend(use_casadi=True)
    lpf2(dae, name, {'omega_n': omega_n, 'zeta': zeta, 'u': u_0}, backend)

    model = CasadiModel(CasadiBuilder(dae).build())
    model.ini({}, dae['xy_0_dict'])

    return model


def test_steady_state_tracks_input():
    """At steady state (du/dt = 0), the filter output z equals the input u
    and the rate state v is zero."""
    model = _build_lpf2_model(name='1', omega_n=10.0, zeta=0.5, u_0=0.3)

    assert abs(model.get_value('z_lpf2_1') - 0.3) < 1e-6
    assert abs(model.get_value('x_lpf2_1') - 0.3) < 1e-6
    assert abs(model.get_value('v_lpf2_1') - 0.0) < 1e-6


def test_underdamped_step_response_overshoots():
    """For zeta < 1, a step in u produces an overshoot above the new
    steady-state value, which then settles back to u.
    """
    name, omega_n, zeta = '1', 10.0, 0.5
    model = _build_lpf2_model(name=name, omega_n=omega_n, zeta=zeta, u_0=0.0)

    model.run(2.0, {f'u_lpf2_{name}': 1.0})
    model.post()

    z = model.get_values(f'z_lpf2_{name}')

    assert max(z) > 1.05      # overshoot present
    assert max(z) < 1.30      # but within the expected range (~16% for zeta=0.5)
    assert abs(z[-1] - 1.0) < 0.01  # settled


def test_critically_damped_step_response_no_overshoot():
    """For zeta >= 1, a step in u produces a monotonic response with no
    overshoot, settling to u.
    """
    name, omega_n, zeta = '1', 10.0, 1.0
    model = _build_lpf2_model(name=name, omega_n=omega_n, zeta=zeta, u_0=0.0)

    model.run(2.0, {f'u_lpf2_{name}': 1.0})
    model.post()

    z = model.get_values(f'z_lpf2_{name}')

    assert z[-1] == max(z)           # monotonic rise, no overshoot
    assert abs(z[-1] - 1.0) < 0.01   # settled


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    name, omega_n, zeta = '1', 10.0, 0.5
    model = _build_lpf2_model(name=name, omega_n=omega_n, zeta=zeta, u_0=0.0)

    model.report_u()
    model.report_x()
    model.report_z()
    model.report_params()

    model.run(2.0, {f'u_lpf2_{name}': 1.0})
    model.post()

    fig, ax = plt.subplots()
    ax.plot(model.Time, model.get_values(f'z_lpf2_{name}'), label='$z$ (filtered output)')
    ax.axhline(1.0, color='k', ls='--', lw=0.8, label='$u$ (step input)')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Output')
    ax.legend()
    fig.savefig('lpf2_step_response.svg')
    print("Test completed. Plot saved as 'lpf2_step_response.svg'.")
