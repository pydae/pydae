# -*- coding: utf-8 -*-
r"""
Created on Sun June 15 14:16:30 2026

@author: jmmauricio

Inverse Park transformation, dq0 to abc (PARK_INV).

Reverses the amplitude-invariant Park transform of
:mod:`pydae.control.continuos.park`, mapping the synchronous-frame
components $(u_d, u_q, u_0)$ at instantaneous angle $\theta$ back
to the three-phase signals:

$$z_a = \cos\theta\,u_d - \sin\theta\,u_q + u_0$$

$$z_b = \cos\!\left(\theta - \tfrac{2\pi}{3}\right) u_d
   - \sin\!\left(\theta - \tfrac{2\pi}{3}\right) u_q + u_0$$

$$z_c = \cos\!\left(\theta + \tfrac{2\pi}{3}\right) u_d
   - \sin\!\left(\theta + \tfrac{2\pi}{3}\right) u_q + u_0$$

Purely algebraic, no dynamic state. Composing
``park(input=[a, b, c, theta]) -> park_inv(input=[z_d, z_q, z_0, theta])``
is the identity.
"""

import io


def descriptions():
    """
    Single source of truth for model parameters, inputs, states, and outputs.
    """
    descriptions_list = []

    descriptions_list += [{"type": "Input", "tex": "u_d",      "data": "u_d",   "model": "u_d",   "default": 0.0, "description": "Direct-axis component",      "units": "-"}]
    descriptions_list += [{"type": "Input", "tex": "u_q",      "data": "u_q",   "model": "u_q",   "default": 0.0, "description": "Quadrature-axis component",  "units": "-"}]
    descriptions_list += [{"type": "Input", "tex": "u_0",      "data": "u_0",   "model": "u_0",   "default": 0.0, "description": "Zero-sequence component",    "units": "-"}]
    descriptions_list += [{"type": "Input", "tex": r"\theta", "data": "theta", "model": "theta", "default": 0.0, "description": "Reference-frame angle",      "units": "rad"}]

    descriptions_list += [{"type": "Algebraic State", "tex": "z_a", "data": "", "model": "z_a", "default": "", "description": "Phase-a signal (exposed in h_dict)", "units": "-"}]
    descriptions_list += [{"type": "Algebraic State", "tex": "z_b", "data": "", "model": "z_b", "default": "", "description": "Phase-b signal (exposed in h_dict)", "units": "-"}]
    descriptions_list += [{"type": "Algebraic State", "tex": "z_c", "data": "", "model": "z_c", "default": "", "description": "Phase-c signal (exposed in h_dict)", "units": "-"}]

    return descriptions_list


def park_inv(dae, name, backend, data_dict=None, input=None, output=None):
    """Append an inverse Park (dq0 -> abc) transformation to `dae`.

    Parameters
    ----------
    input : list[str] or None
        ``None`` -- create four fresh global inputs ``u_d_park_inv_{name}``,
        ``u_q_park_inv_{name}``, ``u_0_park_inv_{name}``,
        ``theta_park_inv_{name}``.
        ``list[str]`` -- must contain exactly four names in order
        ``[u_d, u_q, u_0, theta]``.
    output : list[str] or None
        ``None`` -- expose the three outputs as fresh
        ``z_a_park_inv_{name}``, ``z_b_park_inv_{name}``,
        ``z_c_park_inv_{name}``.
        ``list[str]`` -- must contain exactly three names in order
        ``[z_a, z_b, z_c]``; aliased outputs that were previously
        global inputs are promoted to algebraic states.
    """
    import numpy as np
    from pydae.control.continuos._common import resolve_input

    data_dict = data_dict or {}

    meta = descriptions()
    default_map = {item['data']: item['default'] for item in meta if item.get('data')}
    val_u_d   = data_dict.get('u_d',   default_map.get('u_d',   0.0))
    val_u_q   = data_dict.get('u_q',   default_map.get('u_q',   0.0))
    val_u_0   = data_dict.get('u_0',   default_map.get('u_0',   0.0))
    val_theta = data_dict.get('theta', default_map.get('theta', 0.0))

    if input is None:
        u_d   = backend.symbols(f"u_d_park_inv_{name}")
        u_q   = backend.symbols(f"u_q_park_inv_{name}")
        u_0   = backend.symbols(f"u_0_park_inv_{name}")
        theta = backend.symbols(f"theta_park_inv_{name}")
        for sym, val in zip([u_d, u_q, u_0, theta], [val_u_d, val_u_q, val_u_0, val_theta]):
            dae['u_ini_dict'][f"{sym}"] = val
            dae['u_run_dict'][f"{sym}"] = val
        in_exprs = (u_d, u_q, u_0, theta)
    else:
        if isinstance(input, str) or len(list(input)) != 4:
            raise ValueError(
                "park_inv: input must be a list of 4 names [u_d, u_q, u_0, theta]"
            )
        names = list(input)
        defaults = [val_u_d, val_u_q, val_u_0, val_theta]
        in_exprs = tuple(resolve_input(nm, dae, backend, d) for nm, d in zip(names, defaults))

    u_d_x, u_q_x, u_0_x, theta_x = in_exprs

    if output is None:
        z_a = backend.symbols(f"z_a_park_inv_{name}")
        z_b = backend.symbols(f"z_b_park_inv_{name}")
        z_c = backend.symbols(f"z_c_park_inv_{name}")
        out_aliased = [False, False, False]
        out_syms = [z_a, z_b, z_c]
    else:
        if isinstance(output, str) or len(list(output)) != 3:
            raise ValueError(
                "park_inv: output must be a list of 3 names [z_a, z_b, z_c]"
            )
        out_syms = [backend.symbols(nm) for nm in list(output)]
        out_aliased = [True, True, True]
        z_a, z_b, z_c = out_syms

    cos = backend.cos
    sin = backend.sin
    two_pi_3 = 2.0 * np.pi / 3.0

    expr_a = cos(theta_x)             * u_d_x - sin(theta_x)             * u_q_x + u_0_x
    expr_b = cos(theta_x - two_pi_3)  * u_d_x - sin(theta_x - two_pi_3)  * u_q_x + u_0_x
    expr_c = cos(theta_x + two_pi_3)  * u_d_x - sin(theta_x + two_pi_3)  * u_q_x + u_0_x

    g_a = z_a - expr_a
    g_b = z_b - expr_b
    g_c = z_c - expr_c

    for out_sym, aliased in zip(out_syms, out_aliased):
        if aliased:
            dae['u_ini_dict'].pop(str(out_sym), None)
            dae['u_run_dict'].pop(str(out_sym), None)

    val_a = float(np.cos(val_theta)            * val_u_d - np.sin(val_theta)            * val_u_q + val_u_0)
    val_b = float(np.cos(val_theta - two_pi_3) * val_u_d - np.sin(val_theta - two_pi_3) * val_u_q + val_u_0)
    val_c = float(np.cos(val_theta + two_pi_3) * val_u_d - np.sin(val_theta + two_pi_3) * val_u_q + val_u_0)
    dae['xy_0_dict'].update({f"{z_a}": val_a, f"{z_b}": val_b, f"{z_c}": val_c})

    dae['g_list']     += [g_a, g_b, g_c]
    dae['y_ini_list'] += [z_a, z_b, z_c]
    dae['y_run_list'] += [z_a, z_b, z_c]
    for s in (z_a, z_b, z_c):
        dae['h_dict'][f"{s}"] = s

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
def _build_park_inv_model(name='1', u_d_0=0.0, u_q_0=0.0, u_0_0=0.0, theta_0=0.0):
    from pydae.core.builder.casadi_builder import MathBackend, CasadiBuilder
    from pydae.core.model.casadi_model import CasadiModel
    from pydae.control.continuos.lpf1 import lpf1
    dae = {'name': f'test_park_inv_{name}',
           'f_list': [], 'x_list': [], 'g_list': [], 'y_ini_list': [], 'y_run_list': [],
           'u_ini_dict': {}, 'u_run_dict': {}, 'params_dict': {}, 'h_dict': {}, 'xy_0_dict': {}}
    backend = MathBackend(use_casadi=True)
    lpf1(dae, '_dummy', backend, {'T': 1.0, 'u': 0.0})
    park_inv(dae, name, backend,
             {'u_d': u_d_0, 'u_q': u_q_0, 'u_0': u_0_0, 'theta': theta_0})
    model = CasadiModel(CasadiBuilder(dae).build())
    model.ini({}, dae['xy_0_dict'])
    return model


def test_d_only_at_theta_zero_gives_balanced_a_peak():
    """u_d = 1, u_q = u_0 = 0, theta = 0 -> a = 1, b = c = -0.5."""
    import numpy as np
    model = _build_park_inv_model(name='1', u_d_0=1.0, theta_0=0.0)
    assert abs(model.get_value('z_a_park_inv_1') - 1.0) < 1e-9
    assert abs(model.get_value('z_b_park_inv_1') - np.cos(-2 * np.pi / 3)) < 1e-9
    assert abs(model.get_value('z_c_park_inv_1') - np.cos( 2 * np.pi / 3)) < 1e-9


def test_zero_sequence_offsets_all_phases():
    """A non-zero u_0 adds the same offset to a, b, c."""
    model = _build_park_inv_model(name='1', u_0_0=0.3)
    assert abs(model.get_value('z_a_park_inv_1') - 0.3) < 1e-9
    assert abs(model.get_value('z_b_park_inv_1') - 0.3) < 1e-9
    assert abs(model.get_value('z_c_park_inv_1') - 0.3) < 1e-9


def test_park_inv_park_is_identity():
    """Compose park (abc -> dq0) and park_inv (dq0 -> abc) and verify
    the output equals the input."""
    import numpy as np
    from pydae.core.builder.casadi_builder import MathBackend, CasadiBuilder
    from pydae.core.model.casadi_model import CasadiModel
    from pydae.control.continuos.lpf1 import lpf1
    from pydae.control.continuos.park import park

    dae = {'name': 'test_park_round_trip',
           'f_list': [], 'x_list': [], 'g_list': [], 'y_ini_list': [], 'y_run_list': [],
           'u_ini_dict': {}, 'u_run_dict': {}, 'params_dict': {}, 'h_dict': {}, 'xy_0_dict': {}}
    backend = MathBackend(use_casadi=True)
    lpf1(dae, 'a', backend, {'T': 0.01, 'u':  0.7})
    lpf1(dae, 'b', backend, {'T': 0.01, 'u': -0.3})
    lpf1(dae, 'c', backend, {'T': 0.01, 'u': -0.1})
    park(dae, 'p', backend, {'theta': 0.4},
         input=['z_lpf1_a', 'z_lpf1_b', 'z_lpf1_c', 'theta_p'])
    park_inv(dae, 'inv', backend, {'theta': 0.4},
             input=['z_d_park_p', 'z_q_park_p', 'z_0_park_p', 'theta_p'])
    model = CasadiModel(CasadiBuilder(dae).build())
    model.ini({}, dae['xy_0_dict'])
    assert abs(model.get_value('z_a_park_inv_inv') -  0.7) < 1e-9
    assert abs(model.get_value('z_b_park_inv_inv') - (-0.3)) < 1e-9
    assert abs(model.get_value('z_c_park_inv_inv') - (-0.1)) < 1e-9


if __name__ == '__main__':
    model = _build_park_inv_model(name='1', u_d_0=1.0, theta_0=0.0)
    print(f"a = {model.get_value('z_a_park_inv_1'):+.4f}  "
          f"b = {model.get_value('z_b_park_inv_1'):+.4f}  "
          f"c = {model.get_value('z_c_park_inv_1'):+.4f}")
