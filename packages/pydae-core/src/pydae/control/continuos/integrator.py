# -*- coding: utf-8 -*-
r"""
Created on Sun June 15 14:00:30 2026

@author: jmmauricio

Integrator with output saturation and anti-windup (INTEGRATOR).

A single-state integrator with optional output clipping and
back-calculation anti-windup:

$$v = K\,\xi, \qquad
   \dot \xi = u + K_{aw}\,(z - v),
   \qquad z = \mathrm{clip}(v,\; z_{min},\; z_{max})$$

While the output is within the saturation limits, $z = v$ and the
anti-windup term vanishes, so $\dot \xi = u$ -- a clean
gain-$K$ integrator. When the output saturates ($z \neq v$), the
back-calculation gain $K_{aw}$ feeds the clamping error
$(z - v)$ back into the integrator so $\xi$ stops accumulating
past the value that keeps $K\xi = z_{limit}$. Setting $K_{aw} = 0$
recovers the plain (windup-prone) integrator.

This is the integral-only component of
:mod:`pydae.control.continuos.pi`. Use this block when you need a
bare accumulator (energy, charge, integrated frequency error,
totalizer) rather than a full PI / PID controller. ``K`` is the
gain on the integrator output (units per [u]-second); for the
typical "unity-gain integrator" $z = \int u\, dt$ use ``K = 1``.

Because the integrator pole is at the origin, an isolated
integrator has no unique equilibrium of its own: $\xi$ takes
whatever value the surrounding constraints / closed loop require
at initialisation. The ``xy_0_dict`` entry below is only a seed
for the Newton-Raphson solver.
"""

import io


def descriptions():
    """
    Single source of truth for model parameters, inputs, states, and outputs.
    """
    descriptions_list = []

    descriptions_list += [{"type": "Parameter", "tex": "K",        "data": "K",      "model": "K",       "default":  1.0,   "description": "Integrator gain (units per [u]-second)", "units": "1/s"}]
    descriptions_list += [{"type": "Parameter", "tex": "K_{aw}",   "data": "K_aw",   "model": "K_aw",    "default":  2.0,   "description": "Anti-windup (back-calculation) gain",     "units": "1/s"}]
    descriptions_list += [{"type": "Parameter", "tex": "z_{min}",  "data": "z_min",  "model": "z_min",   "default": -1.0e9, "description": "Lower output limit",                      "units": "-"}]
    descriptions_list += [{"type": "Parameter", "tex": "z_{max}",  "data": "z_max",  "model": "z_max",   "default":  1.0e9, "description": "Upper output limit",                      "units": "-"}]

    descriptions_list += [{"type": "Input", "tex": "u", "data": "u", "model": "u", "default": 0.0, "description": "Integrand", "units": "-"}]

    descriptions_list += [{"type": "Dynamic State",  "tex": r"\xi", "data": "", "model": "xi", "default": "", "description": "Integrator state", "units": "-"}]
    descriptions_list += [{"type": "Algebraic State", "tex": "z",   "data": "", "model": "z",  "default": "", "description": "Saturated output (K*xi clipped to [z_min, z_max], exposed in h_dict)", "units": "-"}]

    return descriptions_list


def integrator(dae, name, backend, data_dict=None, input=None, output=None):
    """Append a saturated integrator (with anti-windup) to `dae`.

    See :mod:`pydae.control.continuos._common` for the input/output
    aliasing semantics.
    """
    from pydae.control.continuos._common import combine_inputs

    data_dict = data_dict or {}

    meta = descriptions()
    default_map = {item['data']: item['default'] for item in meta if item.get('data')}
    val_K     = data_dict.get('K',     default_map.get('K',      1.0))
    val_K_aw  = data_dict.get('K_aw',  default_map.get('K_aw',   2.0))
    val_z_min = data_dict.get('z_min', default_map.get('z_min', -1.0e9))
    val_z_max = data_dict.get('z_max', default_map.get('z_max',  1.0e9))
    val_u     = data_dict.get('u',     default_map.get('u',      0.0))
    val_xi    = data_dict.get('xi_0',  0.0)

    if input is None:
        u_sym = backend.symbols(f"u_integrator_{name}")
        dae['u_ini_dict'][f"{u_sym}"] = val_u
        dae['u_run_dict'][f"{u_sym}"] = val_u
        u_expr = u_sym
    else:
        u_expr = combine_inputs(input, dae, backend, val_u)

    xi = backend.symbols(f"xi_integrator_{name}")

    if output is None:
        z = backend.symbols(f"z_integrator_{name}")
        output_aliased = False
    else:
        z = backend.symbols(output)
        output_aliased = True

    K     = backend.symbols(f"K_integrator_{name}")
    K_aw  = backend.symbols(f"K_aw_integrator_{name}")
    z_min = backend.symbols(f"z_min_integrator_{name}")
    z_max = backend.symbols(f"z_max_integrator_{name}")

    v   = K * xi
    dxi = u_expr + K_aw * (z - v)
    g_z = z - backend.hard_limits(v, z_min, z_max)

    if output_aliased:
        dae['u_ini_dict'].pop(str(z), None)
        dae['u_run_dict'].pop(str(z), None)

    val_v = val_K * val_xi
    val_z = min(max(val_v, val_z_min), val_z_max)
    dae['xy_0_dict'].update({f"{xi}": val_xi, f"{z}": val_z})

    dae['f_list'] += [dxi]
    dae['x_list'] += [xi]
    dae['g_list'] += [g_z]
    dae['y_ini_list'] += [z]
    dae['y_run_list'] += [z]
    dae['h_dict'].update({f"{z}": z})
    dae['params_dict'].update({
        f"{K}":     val_K,
        f"{K_aw}":  val_K_aw,
        f"{z_min}": val_z_min,
        f"{z_max}": val_z_max,
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
def _build_integrator_loop_model(name='1', K=1.0, K_aw=2.0,
                                   z_min=-1e9, z_max=1e9, r_0=0.0):
    """Close the integrator in unity negative feedback (u = r - z) so
    initialisation is well-posed -- an isolated integrator has no
    unique equilibrium when its input can be anything."""
    from pydae.core.builder.casadi_builder import MathBackend, CasadiBuilder
    from pydae.core.model.casadi_model import CasadiModel

    dae = {'name': f'test_integrator_loop_{name}',
           'f_list': [], 'x_list': [],
           'g_list': [], 'y_ini_list': [], 'y_run_list': [],
           'u_ini_dict': {}, 'u_run_dict': {},
           'params_dict': {}, 'h_dict': {}, 'xy_0_dict': {}}
    backend = MathBackend(use_casadi=True)
    integrator(dae, name, backend,
               {'K': K, 'K_aw': K_aw, 'z_min': z_min, 'z_max': z_max,
                'u': 0.0, 'xi_0': 0.0})

    # Close the loop: u_integrator = r - z_integrator.
    u_sym = backend.symbols(f"u_integrator_{name}")
    z_sym = backend.symbols(f"z_integrator_{name}")
    r_sym = backend.symbols(f"r_integrator_{name}")
    dae['u_ini_dict'].pop(str(u_sym))
    dae['u_run_dict'].pop(str(u_sym))
    dae['g_list'] += [u_sym - (r_sym - z_sym)]
    dae['y_ini_list'] += [u_sym]
    dae['y_run_list'] += [u_sym]
    dae['u_ini_dict'][f"{r_sym}"] = r_0
    dae['u_run_dict'][f"{r_sym}"] = r_0
    dae['xy_0_dict'][f"{u_sym}"] = 0.0

    model = CasadiModel(CasadiBuilder(dae).build())
    model.ini({}, dae['xy_0_dict'])
    return model


def test_closed_loop_settles_with_zero_steady_state_error():
    """The integrator drives the loop error to zero, so z = r and
    xi = r / K at steady state."""
    K, r_0 = 0.5, 0.0
    model = _build_integrator_loop_model(name='1', K=K, r_0=r_0)
    assert abs(model.get_value('z_integrator_1') - r_0) < 1e-9
    assert abs(model.get_value('u_integrator_1') - 0.0) < 1e-9


def test_step_response_settles_to_reference():
    """A step in r drives z toward r with first-order dynamics
    (the loop becomes a unity-gain LPF with time constant 1/K)."""
    name, K, r_1 = '1', 1.0, 1.0
    model = _build_integrator_loop_model(name=name, K=K, r_0=0.0)
    model.run(15.0, {f'r_integrator_{name}': r_1})
    model.post()
    assert abs(model.get_value(f'z_integrator_{name}') - r_1) < 0.02
    assert abs(model.get_value(f'xi_integrator_{name}') - r_1 / K) < 0.02


def test_anti_windup_bounds_xi():
    """Driving the loop reference outside [z_min, z_max] saturates
    the output but keeps xi bounded via the K_aw back-calculation.

    With K = 1, K_aw = 2, z_max = 1, r = 3, the closed-loop residual
    at saturation is dxi/dt = u + K_aw*(z - K*xi) = (r - z) + 2*(z - xi)
    = (3 - 1) + 2*(1 - xi) = 4 - 2*xi -> xi_ss = 2.0 (bounded).
    Without anti-windup (K_aw = 0) xi would grow without bound.
    """
    name = '1'
    K, K_aw = 1.0, 2.0
    z_min, z_max = -1.0, 1.0
    r_1 = 3.0
    model = _build_integrator_loop_model(name=name, K=K, K_aw=K_aw,
                                          z_min=z_min, z_max=z_max, r_0=0.0)
    model.run(20.0, {f'r_integrator_{name}': r_1})
    model.post()
    assert abs(model.get_value(f'z_integrator_{name}') - z_max) < 1e-3
    assert abs(model.get_value(f'xi_integrator_{name}') - 2.0) < 0.05


if __name__ == '__main__':
    model = _build_integrator_loop_model(name='1', K=0.5, r_0=0.0)
    model.report_x()
    model.report_z()
    model.run(20.0, {'r_integrator_1': 1.0})
    model.post()
    print(f"z = {model.get_value('z_integrator_1'):.4f}, "
          f"xi = {model.get_value('xi_integrator_1'):.4f}")
