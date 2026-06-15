# -*- coding: utf-8 -*-
r"""
Created on Sun June 15 14:15:00 2026

@author: jmmauricio

Notch (bandstop) filter (NOTCH).

A second-order filter that lets all frequencies pass except a narrow
band around the centre frequency $\omega_0$:

$$H(s) = \frac{s^2 + \omega_0^2}{s^2 + \frac{\omega_0}{Q}\,s + \omega_0^2}$$

* DC gain $H(0) = 1$ -- a constant input passes through unchanged.
* Notch depth at $s = j\omega_0$ is zero -- a sinusoid at the centre
  frequency is fully rejected.
* Quality factor $Q$ sets the bandwidth of the rejection band; the
  -3 dB bandwidth is $\omega_0 / Q$.

Common power-system uses: damping subsynchronous resonance (SSR),
rejecting a specific harmonic component, suppressing torsional
modes. For an arbitrary biquad transfer function use
:mod:`pydae.control.continuos.tf` instead.

Internally realised in controllable canonical form (matching
:mod:`pydae.control.continuos.lpf2`): two dynamic states, one
algebraic output, direct feedthrough $D = 1$.
"""

import io


def descriptions():
    """
    Single source of truth for model parameters, inputs, states, and outputs.
    """
    descriptions_list = []

    descriptions_list += [{"type": "Parameter", "tex": r"\omega_0", "data": "omega_n", "model": "omega_n", "default": 314.159, "description": "Notch centre frequency", "units": "rad/s"}]
    descriptions_list += [{"type": "Parameter", "tex": "Q",         "data": "Q",       "model": "Q",       "default":  10.0,    "description": "Quality factor (sets -3 dB bandwidth = omega_n / Q)", "units": "-"}]

    descriptions_list += [{"type": "Input", "tex": "u", "data": "u", "model": "u", "default": 0.0, "description": "Filter input signal", "units": "-"}]

    descriptions_list += [{"type": "Dynamic State", "tex": "x_1", "data": "", "model": "x_1", "default": "", "description": "Canonical state 1 (settles to u / omega_n^2)", "units": "-"}]
    descriptions_list += [{"type": "Dynamic State", "tex": "x_2", "data": "", "model": "x_2", "default": "", "description": "Canonical state 2 (zero at steady state)",      "units": "-"}]
    descriptions_list += [{"type": "Algebraic State", "tex": "z", "data": "", "model": "z", "default": "", "description": "Notch output (exposed in h_dict)", "units": "-"}]

    return descriptions_list


def notch(dae, name, backend, data_dict=None, input=None, output=None):
    """Append a notch (bandstop) filter to `dae`.

    See :mod:`pydae.control.continuos._common` for the input/output
    aliasing semantics. A ``list[str]`` input is combined as
    ``u = input[0] - sum(input[1:])`` before being fed to the filter.
    """
    from pydae.control.continuos._common import combine_inputs

    data_dict = data_dict or {}

    meta = descriptions()
    default_map = {item['data']: item['default'] for item in meta if item.get('data')}
    val_omega_n = data_dict.get('omega_n', default_map.get('omega_n', 314.159))
    val_Q       = data_dict.get('Q',       default_map.get('Q',        10.0))
    val_u       = data_dict.get('u',       default_map.get('u',        0.0))

    if val_omega_n <= 0:
        raise ValueError("notch: omega_n must be strictly positive")
    if val_Q <= 0:
        raise ValueError("notch: Q must be strictly positive")

    if input is None:
        u = backend.symbols(f"u_notch_{name}")
        dae['u_ini_dict'][f"{u}"] = val_u
        dae['u_run_dict'][f"{u}"] = val_u
        u_expr = u
    else:
        u_expr = combine_inputs(input, dae, backend, val_u)

    x_1 = backend.symbols(f"x_1_notch_{name}")
    x_2 = backend.symbols(f"x_2_notch_{name}")

    if output is None:
        z = backend.symbols(f"z_notch_{name}")
        output_aliased = False
    else:
        z = backend.symbols(output)
        output_aliased = True

    omega_n = backend.symbols(f"omega_n_notch_{name}")
    Q       = backend.symbols(f"Q_notch_{name}")

    # CCF realisation with feedthrough:
    #   dx_1/dt = x_2
    #   dx_2/dt = u - omega_n^2 * x_1 - (omega_n / Q) * x_2
    #   z       = u - (omega_n / Q) * x_2
    # Derivation: H(s) = (s^2 + omega_n^2) / (s^2 + (omega_n/Q) s + omega_n^2),
    # so beta = [omega_n^2, 0, 1], alpha = [omega_n^2, omega_n/Q, 1], D = 1,
    # d_0 = beta_0 - D*alpha_0 = 0, d_1 = -omega_n / Q.
    dx_1 = x_2
    dx_2 = u_expr - omega_n**2 * x_1 - (omega_n / Q) * x_2
    g_z  = z - (u_expr - (omega_n / Q) * x_2)

    if output_aliased:
        dae['u_ini_dict'].pop(str(z), None)
        dae['u_run_dict'].pop(str(z), None)

    # Steady-state seed: x_1 = u / omega_n^2, x_2 = 0, z = u (unity DC gain).
    x_1_ss = val_u / (val_omega_n ** 2)
    dae['xy_0_dict'].update({f"{x_1}": x_1_ss, f"{x_2}": 0.0, f"{z}": val_u})

    dae['f_list']     += [dx_1, dx_2]
    dae['x_list']     += [x_1, x_2]
    dae['g_list']     += [g_z]
    dae['y_ini_list'] += [z]
    dae['y_run_list'] += [z]
    dae['h_dict'].update({f"{z}": z})
    dae['params_dict'].update({f"{omega_n}": val_omega_n, f"{Q}": val_Q})

    return None

# =============================================================================
# Sphinx Documentation Auto-Generator
# =============================================================================
def dict_list_to_aligned_markdown_table(data: list[dict]) -> str:
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
    docs = ""
    for cat in ["Parameter", "Input", "Dynamic State", "Algebraic State", "Output"]:
        cat_list = [{k: v for k, v in item.items() if k != 'type'}
                    for item in descriptions() if item.get('type') == cat]
        if cat_list:
            docs += f"\n### {cat}s\n\n"
            docs += dict_list_to_aligned_markdown_table(cat_list)
            docs += "\n"
    return docs

__doc__ += generate_sphinx_tables()

# =============================================================================
# Testing Block
# =============================================================================
def _build_notch_model(name='1', omega_n=10.0, Q=5.0, u_0=0.0):
    from pydae.core.builder.casadi_builder import MathBackend, CasadiBuilder
    from pydae.core.model.casadi_model import CasadiModel
    dae = {'name': f'test_notch_{name}',
           'f_list': [], 'x_list': [], 'g_list': [], 'y_ini_list': [], 'y_run_list': [],
           'u_ini_dict': {}, 'u_run_dict': {}, 'params_dict': {}, 'h_dict': {}, 'xy_0_dict': {}}
    backend = MathBackend(use_casadi=True)
    notch(dae, name, backend, {'omega_n': omega_n, 'Q': Q, 'u': u_0})
    model = CasadiModel(CasadiBuilder(dae).build())
    model.ini({}, dae['xy_0_dict'])
    return model


def test_dc_passes_through_unity_gain():
    """H(0) = 1: a constant input passes through unchanged."""
    model = _build_notch_model(name='1', omega_n=10.0, Q=5.0, u_0=0.7)
    assert abs(model.get_value('z_notch_1') - 0.7) < 1e-9


def test_sinusoid_at_centre_frequency_is_rejected():
    """Driving the filter with sin(omega_n * t) for many cycles leaves
    z bounded but well below the input amplitude (the notch rejects
    the centre-frequency component)."""
    import numpy as np
    name, omega_n, Q = '1', 10.0, 5.0
    model = _build_notch_model(name=name, omega_n=omega_n, Q=Q, u_0=0.0)
    # Run a couple of natural periods of forced sinusoid.
    T = 2 * np.pi / omega_n
    n_steps = 200
    dt = 8 * T / n_steps
    for k in range(n_steps):
        t = k * dt
        model.run(dt, {f'u_notch_{name}': np.sin(omega_n * t)})
    model.post()
    z = model.get_values(f'z_notch_{name}')
    # After the transient, the steady-state output magnitude should be
    # much smaller than the input amplitude of 1.
    assert max(abs(z[-50:])) < 0.5


def test_validation():
    from pydae.core.builder.casadi_builder import MathBackend
    for bad in ({'omega_n': 0.0}, {'omega_n': -1.0}, {'Q': 0.0}, {'Q': -3.0}):
        dae = {'h_dict': {}, 'g_list': [], 'y_ini_list': [], 'y_run_list': [],
               'u_ini_dict': {}, 'u_run_dict': {}, 'params_dict': {},
               'xy_0_dict': {}, 'x_list': [], 'f_list': []}
        try:
            notch(dae, 't', MathBackend(use_casadi=True), bad)
        except ValueError:
            continue
        raise AssertionError(f"should have rejected: {bad}")


if __name__ == '__main__':
    model = _build_notch_model(name='1', omega_n=10.0, Q=5.0, u_0=0.5)
    print(f"z(DC) = {model.get_value('z_notch_1'):.4f}  (expected 0.5)")
