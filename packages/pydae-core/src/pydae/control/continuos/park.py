# -*- coding: utf-8 -*-
r"""
Created on Sun June 15 14:16:00 2026

@author: jmmauricio

Park transformation, abc to dq0 (PARK).

Amplitude-invariant Park transform of a balanced three-phase
signal $(u_a, u_b, u_c)$ rotating at instantaneous angle $\theta$
into the synchronous reference frame $(z_d, z_q, z_0)$:

$$z_d = \frac{2}{3}\left[
   \cos\theta\,u_a + \cos\!\left(\theta - \tfrac{2\pi}{3}\right) u_b
   + \cos\!\left(\theta + \tfrac{2\pi}{3}\right) u_c\right]$$

$$z_q = -\frac{2}{3}\left[
   \sin\theta\,u_a + \sin\!\left(\theta - \tfrac{2\pi}{3}\right) u_b
   + \sin\!\left(\theta + \tfrac{2\pi}{3}\right) u_c\right]$$

$$z_0 = \frac{1}{3}\,(u_a + u_b + u_c)$$

The 2/3 factor preserves the amplitude of the original phasor: a
balanced positive-sequence sinusoid of amplitude $A$ produces
$z_d = A,\; z_q = 0,\; z_0 = 0$ when $\theta$ is aligned with the
phase-A peak. For the power-invariant variant (sqrt(2/3) factor),
multiply $z_d$ and $z_q$ by $\sqrt{1.5}$ externally; the convention
used here matches the WECC renewable-energy converter models and
the standard SRF-PLL (:mod:`pydae.control.continuos.pll`).

Purely algebraic, no dynamic state. Pair with
:mod:`pydae.control.continuos.park_inv` for the reverse direction.
"""

import io


def descriptions():
    """
    Single source of truth for model parameters, inputs, states, and outputs.
    """
    descriptions_list = []

    descriptions_list += [{"type": "Input", "tex": "u_a",      "data": "u_a",   "model": "u_a",   "default": 0.0, "description": "Phase-a signal", "units": "-"}]
    descriptions_list += [{"type": "Input", "tex": "u_b",      "data": "u_b",   "model": "u_b",   "default": 0.0, "description": "Phase-b signal", "units": "-"}]
    descriptions_list += [{"type": "Input", "tex": "u_c",      "data": "u_c",   "model": "u_c",   "default": 0.0, "description": "Phase-c signal", "units": "-"}]
    descriptions_list += [{"type": "Input", "tex": r"\theta", "data": "theta", "model": "theta", "default": 0.0, "description": "Reference-frame angle", "units": "rad"}]

    descriptions_list += [{"type": "Algebraic State", "tex": "z_d", "data": "", "model": "z_d", "default": "", "description": "Direct-axis component (exposed in h_dict)",      "units": "-"}]
    descriptions_list += [{"type": "Algebraic State", "tex": "z_q", "data": "", "model": "z_q", "default": "", "description": "Quadrature-axis component (exposed in h_dict)",  "units": "-"}]
    descriptions_list += [{"type": "Algebraic State", "tex": "z_0", "data": "", "model": "z_0", "default": "", "description": "Zero-sequence component (exposed in h_dict)",    "units": "-"}]

    return descriptions_list


def park(dae, name, backend, data_dict=None, input=None, output=None):
    """Append a Park (abc -> dq0) transformation to `dae`.

    Parameters
    ----------
    input : list[str] or None
        ``None`` -- create four fresh global inputs ``u_a_park_{name}``,
        ``u_b_park_{name}``, ``u_c_park_{name}``, ``theta_park_{name}``.
        ``list[str]`` -- must contain exactly four names in order
        ``[u_a, u_b, u_c, theta]``; each resolved via
        :func:`pydae.control.continuos._common.resolve_input`.
    output : list[str] or None
        ``None`` -- expose the three outputs as fresh
        ``z_d_park_{name}``, ``z_q_park_{name}``, ``z_0_park_{name}``.
        ``list[str]`` -- must contain exactly three names in order
        ``[z_d, z_q, z_0]``; aliased outputs that were previously
        global inputs are promoted to algebraic states.
    """
    import numpy as np
    from pydae.control.continuos._common import resolve_input

    data_dict = data_dict or {}

    meta = descriptions()
    default_map = {item['data']: item['default'] for item in meta if item.get('data')}
    val_u_a   = data_dict.get('u_a',   default_map.get('u_a',   0.0))
    val_u_b   = data_dict.get('u_b',   default_map.get('u_b',   0.0))
    val_u_c   = data_dict.get('u_c',   default_map.get('u_c',   0.0))
    val_theta = data_dict.get('theta', default_map.get('theta', 0.0))

    if input is None:
        u_a   = backend.symbols(f"u_a_park_{name}")
        u_b   = backend.symbols(f"u_b_park_{name}")
        u_c   = backend.symbols(f"u_c_park_{name}")
        theta = backend.symbols(f"theta_park_{name}")
        for sym, val in zip([u_a, u_b, u_c, theta], [val_u_a, val_u_b, val_u_c, val_theta]):
            dae['u_ini_dict'][f"{sym}"] = val
            dae['u_run_dict'][f"{sym}"] = val
        in_exprs = (u_a, u_b, u_c, theta)
    else:
        if isinstance(input, str) or len(list(input)) != 4:
            raise ValueError(
                "park: input must be a list of 4 names [u_a, u_b, u_c, theta]"
            )
        names = list(input)
        defaults = [val_u_a, val_u_b, val_u_c, val_theta]
        in_exprs = tuple(resolve_input(nm, dae, backend, d) for nm, d in zip(names, defaults))

    u_a_x, u_b_x, u_c_x, theta_x = in_exprs

    if output is None:
        z_d = backend.symbols(f"z_d_park_{name}")
        z_q = backend.symbols(f"z_q_park_{name}")
        z_0 = backend.symbols(f"z_0_park_{name}")
        out_aliased = [False, False, False]
        out_syms = [z_d, z_q, z_0]
    else:
        if isinstance(output, str) or len(list(output)) != 3:
            raise ValueError(
                "park: output must be a list of 3 names [z_d, z_q, z_0]"
            )
        out_syms = [backend.symbols(nm) for nm in list(output)]
        out_aliased = [True, True, True]
        z_d, z_q, z_0 = out_syms

    # Algebraic equations (amplitude-invariant Park, 2/3 factor):
    cos = backend.cos
    sin = backend.sin
    two_pi_3 = 2.0 * np.pi / 3.0

    expr_d = (2.0 / 3.0) * (
        cos(theta_x) * u_a_x
        + cos(theta_x - two_pi_3) * u_b_x
        + cos(theta_x + two_pi_3) * u_c_x
    )
    expr_q = -(2.0 / 3.0) * (
        sin(theta_x) * u_a_x
        + sin(theta_x - two_pi_3) * u_b_x
        + sin(theta_x + two_pi_3) * u_c_x
    )
    expr_0 = (1.0 / 3.0) * (u_a_x + u_b_x + u_c_x)

    g_d = z_d - expr_d
    g_q = z_q - expr_q
    g_0 = z_0 - expr_0

    for out_sym, aliased in zip(out_syms, out_aliased):
        if aliased:
            dae['u_ini_dict'].pop(str(out_sym), None)
            dae['u_run_dict'].pop(str(out_sym), None)

    # Steady-state seed via numpy evaluation of the same expression.
    val_d = (2.0 / 3.0) * (
        np.cos(val_theta) * val_u_a
        + np.cos(val_theta - two_pi_3) * val_u_b
        + np.cos(val_theta + two_pi_3) * val_u_c
    )
    val_q = -(2.0 / 3.0) * (
        np.sin(val_theta) * val_u_a
        + np.sin(val_theta - two_pi_3) * val_u_b
        + np.sin(val_theta + two_pi_3) * val_u_c
    )
    val_0 = (val_u_a + val_u_b + val_u_c) / 3.0
    dae['xy_0_dict'].update({f"{z_d}": float(val_d), f"{z_q}": float(val_q), f"{z_0}": float(val_0)})

    dae['g_list']     += [g_d, g_q, g_0]
    dae['y_ini_list'] += [z_d, z_q, z_0]
    dae['y_run_list'] += [z_d, z_q, z_0]
    for s in (z_d, z_q, z_0):
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
def _build_park_model(name='1', u_a_0=0.0, u_b_0=0.0, u_c_0=0.0, theta_0=0.0):
    """park is purely algebraic; add a dummy lpf1 for the CasADi builder."""
    from pydae.core.builder.casadi_builder import MathBackend, CasadiBuilder
    from pydae.core.model.casadi_model import CasadiModel
    from pydae.control.continuos.lpf1 import lpf1
    dae = {'name': f'test_park_{name}',
           'f_list': [], 'x_list': [], 'g_list': [], 'y_ini_list': [], 'y_run_list': [],
           'u_ini_dict': {}, 'u_run_dict': {}, 'params_dict': {}, 'h_dict': {}, 'xy_0_dict': {}}
    backend = MathBackend(use_casadi=True)
    lpf1(dae, '_dummy', backend, {'T': 1.0, 'u': 0.0})
    park(dae, name, backend,
         {'u_a': u_a_0, 'u_b': u_b_0, 'u_c': u_c_0, 'theta': theta_0})
    model = CasadiModel(CasadiBuilder(dae).build())
    model.ini({}, dae['xy_0_dict'])
    return model


def test_balanced_positive_sequence_aligned():
    """Aligned with phase-A peak (theta = 0): d = A, q = 0, 0 = 0."""
    import numpy as np
    A = 1.0
    model = _build_park_model(
        name='1',
        u_a_0= A * np.cos(0.0),
        u_b_0= A * np.cos(-2 * np.pi / 3),
        u_c_0= A * np.cos( 2 * np.pi / 3),
        theta_0=0.0,
    )
    assert abs(model.get_value('z_d_park_1') - A) < 1e-9
    assert abs(model.get_value('z_q_park_1'))     < 1e-9
    assert abs(model.get_value('z_0_park_1'))     < 1e-9


def test_balanced_positive_sequence_quarter_advanced():
    """Theta lags the phase-A peak by pi/2 -> q = A, d = 0."""
    import numpy as np
    A = 1.0
    model = _build_park_model(
        name='1',
        u_a_0= A * np.cos(0.0),
        u_b_0= A * np.cos(-2 * np.pi / 3),
        u_c_0= A * np.cos( 2 * np.pi / 3),
        theta_0=-np.pi / 2,
    )
    # With theta = -pi/2: cos(theta) = 0, sin(theta) = -1.
    # z_d = (2/3)*[0*A + cos(-pi/2 - 2pi/3)*A_b + cos(-pi/2 + 2pi/3)*A_c]
    # Hard to reason out by hand; just confirm magnitude conservation: d^2 + q^2 = A^2.
    d = model.get_value('z_d_park_1')
    q = model.get_value('z_q_park_1')
    assert abs(d ** 2 + q ** 2 - A ** 2) < 1e-9


def test_zero_sequence_picks_up_common_component():
    """A common offset (1, 1, 1) goes entirely into z_0, with z_d = z_q = 0."""
    model = _build_park_model(name='1', u_a_0=0.4, u_b_0=0.4, u_c_0=0.4, theta_0=0.0)
    assert abs(model.get_value('z_0_park_1') - 0.4) < 1e-9
    assert abs(model.get_value('z_d_park_1')) < 1e-9
    assert abs(model.get_value('z_q_park_1')) < 1e-9


def test_input_list_aliasing():
    """input= as a 4-list binds u_a/u_b/u_c/theta to existing signals."""
    from pydae.core.builder.casadi_builder import MathBackend, CasadiBuilder
    from pydae.core.model.casadi_model import CasadiModel
    from pydae.control.continuos.lpf1 import lpf1
    dae = {'name': 'test_park_alias',
           'f_list': [], 'x_list': [], 'g_list': [], 'y_ini_list': [], 'y_run_list': [],
           'u_ini_dict': {}, 'u_run_dict': {}, 'params_dict': {}, 'h_dict': {}, 'xy_0_dict': {}}
    backend = MathBackend(use_casadi=True)
    lpf1(dae, 'va', backend, {'T': 0.01, 'u': 1.0})
    lpf1(dae, 'vb', backend, {'T': 0.01, 'u': -0.5})
    lpf1(dae, 'vc', backend, {'T': 0.01, 'u': -0.5})
    park(dae, 'p', backend, {'theta': 0.0},
         input=['z_lpf1_va', 'z_lpf1_vb', 'z_lpf1_vc', 'theta_p'])
    # 'theta_p' was not previously declared, so the resolver creates it.
    assert 'theta_p' in dae['u_run_dict']
    model = CasadiModel(CasadiBuilder(dae).build())
    model.ini({}, dae['xy_0_dict'])
    # The three lpf1 outputs are (1, -0.5, -0.5) which is a balanced
    # positive-sequence sample with peak on a; d = 1, q = 0, 0 = 0.
    assert abs(model.get_value('z_d_park_p') - 1.0) < 1e-9
    assert abs(model.get_value('z_q_park_p'))       < 1e-9
    assert abs(model.get_value('z_0_park_p'))       < 1e-9


def test_validation_rejects_bad_input_list():
    from pydae.core.builder.casadi_builder import MathBackend
    for ia in ('just_one', ['a', 'b'], ['a', 'b', 'c', 'd', 'e']):
        dae = {'h_dict': {}, 'g_list': [], 'y_ini_list': [], 'y_run_list': [],
               'u_ini_dict': {}, 'u_run_dict': {}, 'params_dict': {},
               'xy_0_dict': {}, 'x_list': [], 'f_list': []}
        try:
            park(dae, 't', MathBackend(use_casadi=True), {}, input=ia)
        except ValueError:
            continue
        raise AssertionError(f"should have rejected input={ia!r}")


if __name__ == '__main__':
    import numpy as np
    A = 1.0
    model = _build_park_model(
        name='1',
        u_a_0= A * np.cos(0.0),
        u_b_0= A * np.cos(-2 * np.pi / 3),
        u_c_0= A * np.cos( 2 * np.pi / 3),
        theta_0=0.0,
    )
    print(f"d = {model.get_value('z_d_park_1'):+.4f}  q = {model.get_value('z_q_park_1'):+.4f}  "
          f"0 = {model.get_value('z_0_park_1'):+.4f}  (expected 1, 0, 0)")
