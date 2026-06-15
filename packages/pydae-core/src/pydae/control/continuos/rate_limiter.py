# -*- coding: utf-8 -*-
r"""
Created on Sun June 15 14:00:00 2026

@author: jmmauricio

Rate limiter (RATE_LIMITER).

A first-order tracker whose tracking rate is clipped:

$$\frac{dz}{dt} = \mathrm{clip}\!\left(\frac{u - z}{T},\;
   dz_{min},\; dz_{max}\right)$$

When the input change is small enough that the unconstrained rate
$(u-z)/T$ stays inside $[dz_{min}, dz_{max}]$, the block behaves
exactly like a first-order low-pass filter
(:mod:`pydae.control.continuos.lpf1`) with time constant $T$. When
the rate limits saturate (large step in $u$), the output follows a
pure ramp at $dz_{max}$ (upward) or $dz_{min}$ (downward) until it
catches up.

Common in power-system models where an actuator setpoint must not
change faster than a known slew rate: AVR / governor reference
ramping, REGC_A's ``rrpwr`` (active-current up-ramp rate), REEC_B's
``dPmax`` / ``dPmin`` active-power ramps, etc.

For ``T -> 0`` the block becomes the classic discontinuous rate
limiter ($\dot z = \mathrm{clip}(\dot u, dz_{min}, dz_{max})$) up to
a small smoothing pole; keep ``T`` strictly positive (and typically
small, e.g. 0.01 s) so the Jacobian stays full-rank.
"""

import io


def descriptions():
    """
    Single source of truth for model parameters, inputs, states, and outputs.
    """
    descriptions_list = []

    descriptions_list += [{"type": "Parameter", "tex": "T",         "data": "T",       "model": "T",       "default": 0.01,    "description": "Tracker time constant (small positive)", "units": "s"}]
    descriptions_list += [{"type": "Parameter", "tex": "dz_{min}",  "data": "dz_min",  "model": "dz_min",  "default": -1.0e9, "description": "Lower rate limit (must be <= 0)",         "units": "1/s"}]
    descriptions_list += [{"type": "Parameter", "tex": "dz_{max}",  "data": "dz_max",  "model": "dz_max",  "default":  1.0e9, "description": "Upper rate limit (must be >= 0)",         "units": "1/s"}]

    descriptions_list += [{"type": "Input", "tex": "u", "data": "u", "model": "u", "default": 0.0, "description": "Reference signal to track", "units": "-"}]

    descriptions_list += [{"type": "Dynamic State",  "tex": "x", "data": "", "model": "x", "default": "", "description": "Rate-limited tracker state",         "units": "-"}]
    descriptions_list += [{"type": "Algebraic State", "tex": "z", "data": "", "model": "z", "default": "", "description": "Output alias of x (exposed in h_dict)", "units": "-"}]

    return descriptions_list


def rate_limiter(dae, name, backend, data_dict=None, input=None, output=None):
    """Append a rate limiter block to `dae`.

    Adds one dynamic state ``x_rate_limiter_{name}`` (the rate-limited
    tracker) plus an algebraic alias ``z_rate_limiter_{name}`` exposed
    through ``h_dict``. See :mod:`pydae.control.continuos._common`
    for the input/output aliasing semantics.
    """
    from pydae.control.continuos._common import combine_inputs

    data_dict = data_dict or {}

    meta = descriptions()
    default_map = {item['data']: item['default'] for item in meta if item.get('data')}
    val_T      = data_dict.get('T',      default_map.get('T',       0.01))
    val_dz_min = data_dict.get('dz_min', default_map.get('dz_min', -1.0e9))
    val_dz_max = data_dict.get('dz_max', default_map.get('dz_max',  1.0e9))
    val_u      = data_dict.get('u',      default_map.get('u',       0.0))

    if val_T <= 0:
        raise ValueError("rate_limiter: T must be strictly positive")

    if input is None:
        u = backend.symbols(f"u_rate_limiter_{name}")
        dae['u_ini_dict'][f"{u}"] = val_u
        dae['u_run_dict'][f"{u}"] = val_u
        u_expr = u
    else:
        u_expr = combine_inputs(input, dae, backend, val_u)

    x = backend.symbols(f"x_rate_limiter_{name}")

    if output is None:
        z = backend.symbols(f"z_rate_limiter_{name}")
        output_aliased = False
    else:
        z = backend.symbols(output)
        output_aliased = True

    T      = backend.symbols(f"T_rate_limiter_{name}")
    dz_min = backend.symbols(f"dz_min_rate_limiter_{name}")
    dz_max = backend.symbols(f"dz_max_rate_limiter_{name}")

    # dx/dt is the rate-limited pursuit of u; z is just an alias of x.
    dx  = backend.hard_limits((u_expr - x) / T, dz_min, dz_max)
    g_z = z - x

    if output_aliased:
        dae['u_ini_dict'].pop(str(z), None)
        dae['u_run_dict'].pop(str(z), None)

    # Steady state at rest: x = u, z = x = u.
    dae['xy_0_dict'].update({f"{x}": val_u, f"{z}": val_u})

    dae['f_list']     += [dx]
    dae['x_list']     += [x]
    dae['g_list']     += [g_z]
    dae['y_ini_list'] += [z]
    dae['y_run_list'] += [z]
    dae['h_dict'].update({f"{z}": z})
    dae['params_dict'].update({
        f"{T}":      val_T,
        f"{dz_min}": val_dz_min,
        f"{dz_max}": val_dz_max,
    })

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
def _build_rate_limiter_model(name='1', T=0.01, dz_min=-1.0, dz_max=1.0, u_0=0.0):
    """Standalone test model."""
    from pydae.core.builder.casadi_builder import MathBackend, CasadiBuilder
    from pydae.core.model.casadi_model import CasadiModel

    dae = {'name': f'test_rate_limiter_{name}',
           'f_list': [], 'x_list': [],
           'g_list': [], 'y_ini_list': [], 'y_run_list': [],
           'u_ini_dict': {}, 'u_run_dict': {},
           'params_dict': {}, 'h_dict': {}, 'xy_0_dict': {}}
    backend = MathBackend(use_casadi=True)
    rate_limiter(dae, name, backend,
                 {'T': T, 'dz_min': dz_min, 'dz_max': dz_max, 'u': u_0})
    model = CasadiModel(CasadiBuilder(dae).build())
    model.ini({}, dae['xy_0_dict'])
    return model


def test_steady_state_tracks_input():
    """At steady state z == u."""
    model = _build_rate_limiter_model(name='1', u_0=0.5)
    assert abs(model.get_value('z_rate_limiter_1') - 0.5) < 1e-9


def test_unrestricted_ramp_when_limits_loose():
    """Step from 0 to 1 with generous rate limits: behaves like lpf1
    with time constant T, settles to step within ~5T."""
    name, T = '1', 0.05
    model = _build_rate_limiter_model(name=name, T=T,
                                       dz_min=-1e6, dz_max=1e6, u_0=0.0)
    model.run(10 * T, {f'u_rate_limiter_{name}': 1.0})
    model.post()
    assert abs(model.get_value(f'z_rate_limiter_{name}') - 1.0) < 0.01


def test_rate_limited_ramp_when_step_too_large():
    """With dz_max = 0.5 and a step of 1.0, the output ramps at the
    limit for ~2 s before reaching the new setpoint."""
    name = '1'
    dz_max = 0.5
    model = _build_rate_limiter_model(name=name, T=0.001,
                                       dz_min=-1.0, dz_max=dz_max, u_0=0.0)
    # Step up; check the mid-ramp value via get_value (hot-path, no post).
    model.run(1.0, {f'u_rate_limiter_{name}': 1.0})
    assert abs(model.get_value(f'z_rate_limiter_{name}') - 0.5) < 0.05
    # Continue running to let it settle, then post() once at the end.
    model.run(2.0, {})
    model.post()
    assert abs(model.get_value(f'z_rate_limiter_{name}') - 1.0) < 0.01


def test_validation_rejects_nonpositive_T():
    """T must be strictly positive."""
    from pydae.core.builder.casadi_builder import MathBackend
    dae = {'h_dict': {}, 'g_list': [], 'y_ini_list': [], 'y_run_list': [],
           'u_ini_dict': {}, 'u_run_dict': {}, 'params_dict': {},
           'xy_0_dict': {}, 'x_list': [], 'f_list': []}
    try:
        rate_limiter(dae, 't', MathBackend(use_casadi=True), {'T': 0.0})
    except ValueError:
        return
    raise AssertionError("should have rejected T=0")


if __name__ == '__main__':
    model = _build_rate_limiter_model(name='1', dz_max=0.5, u_0=0.0)
    model.run(2.0, {'u_rate_limiter_1': 1.0})
    model.post()
    print(f"z(t=2) = {model.get_value('z_rate_limiter_1'):.4f}")
