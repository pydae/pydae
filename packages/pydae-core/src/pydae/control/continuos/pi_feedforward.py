# -*- coding: utf-8 -*-
r"""
Created on Sun June 15 14:18:00 2026

@author: jmmauricio

PI controller with feedforward (PI_FEEDFORWARD).

A two-input PI controller that adds a reference-feedforward term
on top of the error-driven PI action:

$$v = K_p\,e + K_i\,\xi + K_{ff}\,r,
   \qquad \frac{d\xi}{dt} = e + K_{aw}(z - v)$$

$$z = \mathrm{clip}(v,\, z_{min},\, z_{max})$$

When the reference $r$ is known a priori (e.g. an MPPT setpoint,
a measured DC-link voltage that the inner loop must duplicate, a
desired open-loop trajectory), the feedforward term $K_{ff}\,r$
takes the controller to roughly the right operating point without
waiting for the integrator to accumulate the steady-state offset
-- much faster response than a plain PI.

Same output-saturation and back-calculation anti-windup as
:mod:`pydae.control.continuos.pi`. Setting $K_{ff} = 0$ recovers a
plain PI on the error signal alone.

**Input convention** -- the block takes ``e`` (error / measurement
difference) and ``r`` (reference for feedforward) as two
**named slots**:

* ``input=None`` -- creates fresh ``e_pi_feedforward_{name}`` and
  ``r_pi_feedforward_{name}`` as global inputs.
* ``input=[e_name, r_name]`` -- must be a 2-list, first entry binds
  to ``e``, second binds to ``r``. Each resolved via
  :func:`pydae.control.continuos._common.resolve_input` (h_dict
  expression, existing-symbol alias, or new global input). The list
  is **not** combined with subtraction (unlike plain ``pi``); both
  entries are kept as separate signals because the block uses them
  in different terms of the PI law.
"""

import io


def descriptions():
    """
    Single source of truth for model parameters, inputs, states, and outputs.
    """
    descriptions_list = []

    descriptions_list += [{"type": "Parameter", "tex": "K_p",     "data": "K_p",    "model": "K_p",    "default":  1.0,    "description": "Proportional gain",                       "units": "-"}]
    descriptions_list += [{"type": "Parameter", "tex": "K_i",     "data": "K_i",    "model": "K_i",    "default":  1.0,    "description": "Integral gain",                           "units": "1/s"}]
    descriptions_list += [{"type": "Parameter", "tex": "K_{ff}",  "data": "K_ff",   "model": "K_ff",   "default":  0.0,    "description": "Reference-feedforward gain",               "units": "-"}]
    descriptions_list += [{"type": "Parameter", "tex": "K_{aw}",  "data": "K_aw",   "model": "K_aw",   "default":  2.0,    "description": "Anti-windup (back-calculation) gain",      "units": "1/s"}]
    descriptions_list += [{"type": "Parameter", "tex": "z_{min}", "data": "z_min",  "model": "z_min",  "default": -10.0,   "description": "Lower output limit",                       "units": "-"}]
    descriptions_list += [{"type": "Parameter", "tex": "z_{max}", "data": "z_max",  "model": "z_max",  "default":  10.0,   "description": "Upper output limit",                       "units": "-"}]

    descriptions_list += [{"type": "Input", "tex": "e", "data": "e", "model": "e", "default": 0.0, "description": "Error signal (driven into the PI law)", "units": "-"}]
    descriptions_list += [{"type": "Input", "tex": "r", "data": "r", "model": "r", "default": 0.0, "description": "Reference signal (driven into the feedforward term)", "units": "-"}]

    descriptions_list += [{"type": "Dynamic State",  "tex": r"\xi", "data": "", "model": "xi", "default": "", "description": "PI integrator state", "units": "-"}]
    descriptions_list += [{"type": "Algebraic State", "tex": "z",   "data": "", "model": "z",  "default": "", "description": "Saturated controller output (exposed in h_dict)", "units": "-"}]

    return descriptions_list


def pi_feedforward(dae, name, backend, data_dict=None, input=None, output=None):
    """Append a PI-with-feedforward controller to `dae`.

    See the module docstring for the input convention and the
    :mod:`pydae.control.continuos._common` module for output aliasing
    semantics.
    """
    from pydae.control.continuos._common import resolve_input

    data_dict = data_dict or {}

    meta = descriptions()
    default_map = {item['data']: item['default'] for item in meta if item.get('data')}
    val_K_p   = data_dict.get('K_p',   default_map.get('K_p',    1.0))
    val_K_i   = data_dict.get('K_i',   default_map.get('K_i',    1.0))
    val_K_ff  = data_dict.get('K_ff',  default_map.get('K_ff',   0.0))
    val_K_aw  = data_dict.get('K_aw',  default_map.get('K_aw',   2.0))
    val_z_min = data_dict.get('z_min', default_map.get('z_min', -10.0))
    val_z_max = data_dict.get('z_max', default_map.get('z_max',  10.0))
    val_e     = data_dict.get('e',     default_map.get('e',      0.0))
    val_r     = data_dict.get('r',     default_map.get('r',      0.0))
    val_xi    = data_dict.get('xi_0',  0.0)

    if input is None:
        e_sym = backend.symbols(f"e_pi_feedforward_{name}")
        r_sym = backend.symbols(f"r_pi_feedforward_{name}")
        dae['u_ini_dict'][f"{e_sym}"] = val_e
        dae['u_run_dict'][f"{e_sym}"] = val_e
        dae['u_ini_dict'][f"{r_sym}"] = val_r
        dae['u_run_dict'][f"{r_sym}"] = val_r
        e_expr, r_expr = e_sym, r_sym
    else:
        if isinstance(input, str):
            raise ValueError(
                "pi_feedforward: input must be a list of two names [e, r] "
                "(the block has two distinct input slots and the user must "
                "say which is which)."
            )
        names = list(input)
        if len(names) != 2:
            raise ValueError(
                f"pi_feedforward: input list has {len(names)} entries, "
                f"expected 2 [e, r]"
            )
        e_expr = resolve_input(names[0], dae, backend, val_e)
        r_expr = resolve_input(names[1], dae, backend, val_r)

    xi = backend.symbols(f"xi_pi_feedforward_{name}")

    if output is None:
        z = backend.symbols(f"z_pi_feedforward_{name}")
        output_aliased = False
    else:
        z = backend.symbols(output)
        output_aliased = True

    K_p   = backend.symbols(f"K_p_pi_feedforward_{name}")
    K_i   = backend.symbols(f"K_i_pi_feedforward_{name}")
    K_ff  = backend.symbols(f"K_ff_pi_feedforward_{name}")
    K_aw  = backend.symbols(f"K_aw_pi_feedforward_{name}")
    z_min = backend.symbols(f"z_min_pi_feedforward_{name}")
    z_max = backend.symbols(f"z_max_pi_feedforward_{name}")

    v   = K_p * e_expr + K_i * xi + K_ff * r_expr
    dxi = e_expr + K_aw * (z - v)
    g_z = z - backend.hard_limits(v, z_min, z_max)

    if output_aliased:
        dae['u_ini_dict'].pop(str(z), None)
        dae['u_run_dict'].pop(str(z), None)

    val_v = val_K_p * val_e + val_K_i * val_xi + val_K_ff * val_r
    val_z = min(max(val_v, val_z_min), val_z_max)
    dae['xy_0_dict'].update({f"{xi}": val_xi, f"{z}": val_z})

    dae['f_list']     += [dxi]
    dae['x_list']     += [xi]
    dae['g_list']     += [g_z]
    dae['y_ini_list'] += [z]
    dae['y_run_list'] += [z]
    dae['h_dict'].update({f"{z}": z})
    dae['params_dict'].update({
        f"{K_p}":   val_K_p,   f"{K_i}":   val_K_i,
        f"{K_ff}":  val_K_ff,  f"{K_aw}":  val_K_aw,
        f"{z_min}": val_z_min, f"{z_max}": val_z_max,
    })

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
def _build_pi_feedforward_loop(name='1', K_p=2.0, K_i=0.5, K_ff=0.0, K_aw=2.0,
                                 z_min=-10.0, z_max=10.0, r_0=0.0):
    """Close in unity negative feedback (e = r - z). For K_ff = 0 the
    loop reproduces the plain pi block; for K_ff > 0 the initial jump
    in z is larger."""
    from pydae.core.builder.casadi_builder import MathBackend, CasadiBuilder
    from pydae.core.model.casadi_model import CasadiModel

    dae = {'name': f'test_pi_ff_loop_{name}',
           'f_list': [], 'x_list': [], 'g_list': [], 'y_ini_list': [], 'y_run_list': [],
           'u_ini_dict': {}, 'u_run_dict': {}, 'params_dict': {}, 'h_dict': {}, 'xy_0_dict': {}}
    backend = MathBackend(use_casadi=True)
    pi_feedforward(dae, name, backend,
                   {'K_p': K_p, 'K_i': K_i, 'K_ff': K_ff, 'K_aw': K_aw,
                    'z_min': z_min, 'z_max': z_max,
                    'e': 0.0, 'r': r_0, 'xi_0': 0.0})

    # Wire e = r - z (negative-feedback error). r is shared with the
    # feedforward slot so the same external reference drives both.
    e_sym = backend.symbols(f"e_pi_feedforward_{name}")
    z_sym = backend.symbols(f"z_pi_feedforward_{name}")
    r_sym = backend.symbols(f"r_pi_feedforward_{name}")
    dae['u_ini_dict'].pop(str(e_sym))
    dae['u_run_dict'].pop(str(e_sym))
    dae['g_list'] += [e_sym - (r_sym - z_sym)]
    dae['y_ini_list'] += [e_sym]
    dae['y_run_list'] += [e_sym]
    dae['xy_0_dict'][f"{e_sym}"] = 0.0

    model = CasadiModel(CasadiBuilder(dae).build())
    model.ini({}, dae['xy_0_dict'])
    return model


def test_zero_feedforward_matches_plain_pi():
    """K_ff = 0: integral action drives steady-state error to zero,
    z == r."""
    K_p, K_i, r_0 = 2.0, 0.5, 0.0
    model = _build_pi_feedforward_loop(name='1', K_p=K_p, K_i=K_i, K_ff=0.0, r_0=r_0)
    assert abs(model.get_value('z_pi_feedforward_1') - r_0) < 1e-9


def test_feedforward_settles_to_reference_too():
    """With K_ff != 0 the loop still settles z = r (integrator absorbs
    any residual)."""
    name, K_p, K_i, K_ff, r_1 = '1', 2.0, 0.5, 1.0, 1.0
    model = _build_pi_feedforward_loop(name=name, K_p=K_p, K_i=K_i, K_ff=K_ff, r_0=0.0)
    model.run(30.0, {f'r_pi_feedforward_{name}': r_1})
    model.post()
    assert abs(model.get_value(f'z_pi_feedforward_{name}') - r_1) < 0.02


def test_feedforward_speeds_up_initial_response():
    """For the same K_p / K_i, a non-zero K_ff produces a larger jump
    in z right after a step in r than the plain PI (K_ff = 0)."""
    name, K_p, K_i, r_1 = '1', 2.0, 0.5, 1.0

    m_no_ff = _build_pi_feedforward_loop(name=name, K_p=K_p, K_i=K_i, K_ff=0.0, r_0=0.0)
    m_no_ff.run(0.1, {f'r_pi_feedforward_{name}': r_1})
    m_no_ff.post()
    z_no_ff = m_no_ff.get_values(f'z_pi_feedforward_{name}')

    m_ff = _build_pi_feedforward_loop(name=name, K_p=K_p, K_i=K_i, K_ff=1.0, r_0=0.0)
    m_ff.run(0.1, {f'r_pi_feedforward_{name}': r_1})
    m_ff.post()
    z_ff = m_ff.get_values(f'z_pi_feedforward_{name}')

    assert z_ff[1] > z_no_ff[1]


def test_validation_rejects_bad_input():
    from pydae.core.builder.casadi_builder import MathBackend
    for ia in ('just_one', ['a'], ['a', 'b', 'c']):
        dae = {'h_dict': {}, 'g_list': [], 'y_ini_list': [], 'y_run_list': [],
               'u_ini_dict': {}, 'u_run_dict': {}, 'params_dict': {},
               'xy_0_dict': {}, 'x_list': [], 'f_list': []}
        try:
            pi_feedforward(dae, 't', MathBackend(use_casadi=True), {}, input=ia)
        except ValueError:
            continue
        raise AssertionError(f"should have rejected input={ia!r}")


if __name__ == '__main__':
    model = _build_pi_feedforward_loop(name='1', K_p=2.0, K_i=0.5, K_ff=1.0, r_0=0.0)
    model.run(5.0, {'r_pi_feedforward_1': 1.0})
    model.post()
    print(f"z = {model.get_value('z_pi_feedforward_1'):.4f}  "
          f"xi = {model.get_value('xi_pi_feedforward_1'):.4f}")
