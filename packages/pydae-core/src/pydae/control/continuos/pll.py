# -*- coding: utf-8 -*-
r"""
Created on Sun June 15 14:17:00 2026

@author: jmmauricio

Synchronous-Reference-Frame Phase-Locked Loop (PLL).

Estimates the instantaneous angle $\hat\theta$ and frequency
$\hat\omega$ of a three-phase voltage from its $abc$ samples, using
the classical SRF-PLL topology:

1. Project the three-phase voltage onto the synchronous frame using
   the current angle estimate $\hat\theta$ (Park transform):

   $$v_q = -\tfrac{2}{3}\!\left[\sin\hat\theta\,v_a
     + \sin\!\left(\hat\theta - \tfrac{2\pi}{3}\right) v_b
     + \sin\!\left(\hat\theta + \tfrac{2\pi}{3}\right) v_c\right]$$

2. Drive $v_q$ to zero with a PI controller, producing a frequency
   correction $\Delta\hat\omega$:

   $$\frac{d\xi}{dt} = v_q,
     \qquad \Delta\hat\omega = K_p\,v_q + K_i\,\xi$$

3. Integrate the corrected frequency *as the deviation from the
   nominal-frequency reference* to update the angle estimate:

   $$\hat\omega = \omega_0 + \Delta\hat\omega,
     \qquad \frac{d\hat\theta}{dt} = \hat\omega - \omega_0
       = \Delta\hat\omega$$

The deviation formulation $d\hat\theta/dt = \Delta\hat\omega$ (rather
than the absolute $\hat\omega$) is what lets the DAE solver find a
``dx/dt = 0`` initialisation: at lock the PI is at rest and the
angle drift relative to the nominal reference is zero, so
$\hat\theta$ holds its bias value. Same convention as the rotor
angle $\delta$ in the pydae synchronous-machine models -- all
"absolute" rotation is implicit in the $\omega_0$ reference frame.

When locked, $v_q \to 0$ and $\hat\theta$ tracks the bias between
the actual voltage angle and the nominal reference frame;
$\hat\omega$ tracks the actual frequency. The bandwidth is set by
``K_p`` / ``K_i``.

**Dimensions.** Three inputs ($v_a, v_b, v_c$); two dynamic states
($\xi$, $\hat\theta$); outputs $\hat\omega$, $\hat\theta$, and the
internal $v_d$, $v_q$ -- all exposed via ``h_dict``.

For the underlying Park transform alone, use
:mod:`pydae.control.continuos.park`; this block is essentially that
transform wired into a PI + angle-integrator loop.
"""

import io


def descriptions():
    """
    Single source of truth for model parameters, inputs, states, and outputs.
    """
    descriptions_list = []

    descriptions_list += [{"type": "Parameter", "tex": "K_p",         "data": "K_p",     "model": "K_p",     "default":  10.0,    "description": "PLL PI proportional gain (rad/(s*pu))", "units": "rad/(s*pu)"}]
    descriptions_list += [{"type": "Parameter", "tex": "K_i",         "data": "K_i",     "model": "K_i",     "default":  50.0,    "description": "PLL PI integral gain (rad/s^2/pu)",      "units": "rad/s^2/pu"}]
    descriptions_list += [{"type": "Parameter", "tex": r"\omega_0",  "data": "omega_0", "model": "omega_0", "default": 314.159,  "description": "Nominal-frequency feedforward",          "units": "rad/s"}]

    descriptions_list += [{"type": "Input", "tex": "v_a", "data": "v_a", "model": "v_a", "default": 0.0, "description": "Phase-a voltage", "units": "-"}]
    descriptions_list += [{"type": "Input", "tex": "v_b", "data": "v_b", "model": "v_b", "default": 0.0, "description": "Phase-b voltage", "units": "-"}]
    descriptions_list += [{"type": "Input", "tex": "v_c", "data": "v_c", "model": "v_c", "default": 0.0, "description": "Phase-c voltage", "units": "-"}]

    descriptions_list += [{"type": "Dynamic State", "tex": r"\xi",         "data": "", "model": "xi",        "default": "", "description": "PI integrator state",        "units": "-"}]
    descriptions_list += [{"type": "Dynamic State", "tex": r"\hat\theta", "data": "", "model": "theta_hat", "default": "", "description": "Estimated voltage angle",    "units": "rad"}]

    descriptions_list += [{"type": "Algebraic State", "tex": r"\hat\omega", "data": "", "model": "omega_hat", "default": "", "description": "Estimated voltage frequency (exposed in h_dict)", "units": "rad/s"}]

    return descriptions_list


def pll(dae, name, backend, data_dict=None, input=None, output=None):
    """Append an SRF-PLL block to `dae`.

    Parameters
    ----------
    input : list[str] or None
        ``None`` -- create three fresh global inputs ``v_a_pll_{name}``,
        ``v_b_pll_{name}``, ``v_c_pll_{name}``.
        ``list[str]`` -- must contain exactly three names in order
        ``[v_a, v_b, v_c]``; each resolved via
        :func:`pydae.control.continuos._common.resolve_input`.
    output : list[str] or None
        ``None`` -- expose ``omega_hat`` as a fresh algebraic output;
        ``theta_hat`` is the dynamic state and is always available in
        ``h_dict`` under its canonical name.
        ``list[str]`` -- must contain exactly one name for ``omega_hat``;
        if it was a previously-declared global input it is promoted to
        an algebraic variable.
    """
    import numpy as np
    from pydae.control.continuos._common import resolve_input

    data_dict = data_dict or {}

    meta = descriptions()
    default_map = {item['data']: item['default'] for item in meta if item.get('data')}
    val_K_p     = data_dict.get('K_p',     default_map.get('K_p',      10.0))
    val_K_i     = data_dict.get('K_i',     default_map.get('K_i',      50.0))
    val_omega_0 = data_dict.get('omega_0', default_map.get('omega_0', 314.159))
    val_v_a     = data_dict.get('v_a',     default_map.get('v_a',      0.0))
    val_v_b     = data_dict.get('v_b',     default_map.get('v_b',      0.0))
    val_v_c     = data_dict.get('v_c',     default_map.get('v_c',      0.0))
    val_theta_0 = data_dict.get('theta_0', 0.0)
    val_xi_0    = data_dict.get('xi_0',    0.0)

    if input is None:
        v_a = backend.symbols(f"v_a_pll_{name}")
        v_b = backend.symbols(f"v_b_pll_{name}")
        v_c = backend.symbols(f"v_c_pll_{name}")
        for sym, val in zip([v_a, v_b, v_c], [val_v_a, val_v_b, val_v_c]):
            dae['u_ini_dict'][f"{sym}"] = val
            dae['u_run_dict'][f"{sym}"] = val
        v_a_x, v_b_x, v_c_x = v_a, v_b, v_c
    else:
        if isinstance(input, str) or len(list(input)) != 3:
            raise ValueError(
                "pll: input must be a list of 3 names [v_a, v_b, v_c]"
            )
        names = list(input)
        defaults = [val_v_a, val_v_b, val_v_c]
        v_a_x, v_b_x, v_c_x = (resolve_input(nm, dae, backend, d) for nm, d in zip(names, defaults))

    # States: PI integrator and angle integrator.
    xi        = backend.symbols(f"xi_pll_{name}")
    theta_hat = backend.symbols(f"theta_hat_pll_{name}")

    # Output: estimated frequency (algebraic, exposed via h_dict).
    if output is None:
        omega_hat = backend.symbols(f"omega_hat_pll_{name}")
        output_aliased = False
    else:
        if isinstance(output, str):
            omega_hat = backend.symbols(output)
        else:
            names_out = list(output)
            if len(names_out) != 1:
                raise ValueError(
                    "pll: output must be a string or 1-element list naming omega_hat"
                )
            omega_hat = backend.symbols(names_out[0])
        output_aliased = True

    K_p     = backend.symbols(f"K_p_pll_{name}")
    K_i     = backend.symbols(f"K_i_pll_{name}")
    omega_0 = backend.symbols(f"omega_0_pll_{name}")

    # Park projection (q-axis only is fed back; d-axis is the magnitude).
    two_pi_3 = 2.0 * np.pi / 3.0
    cos = backend.cos
    sin = backend.sin
    v_d = (2.0 / 3.0) * (
        cos(theta_hat)            * v_a_x
        + cos(theta_hat - two_pi_3) * v_b_x
        + cos(theta_hat + two_pi_3) * v_c_x
    )
    v_q = -(2.0 / 3.0) * (
        sin(theta_hat)            * v_a_x
        + sin(theta_hat - two_pi_3) * v_b_x
        + sin(theta_hat + two_pi_3) * v_c_x
    )

    # Equations:
    #   dxi/dt        = v_q                              (PI integrator)
    #   omega_hat     = omega_0 + K_p * v_q + K_i * xi  (algebraic)
    #   dtheta_hat/dt = omega_hat - omega_0              (deviation tracker)
    #
    # The deviation form for dtheta_hat is what makes ini() well-posed:
    # at lock omega_hat = omega_0 so dtheta_hat = 0, and theta_hat holds
    # its bias value while the absolute angle (= omega_0 * t + theta_hat)
    # is rotating at exactly omega_0.
    dxi        = v_q
    dtheta_hat = omega_hat - omega_0
    g_omega    = omega_hat - (omega_0 + K_p * v_q + K_i * xi)

    if output_aliased:
        dae['u_ini_dict'].pop(str(omega_hat), None)
        dae['u_run_dict'].pop(str(omega_hat), None)

    # Steady-state seed: assume the loop is locked at theta_0 with
    # omega_hat = omega_0 (so xi = 0 -- the PI is at rest).
    dae['xy_0_dict'].update({
        f"{xi}":        val_xi_0,
        f"{theta_hat}": val_theta_0,
        f"{omega_hat}": val_omega_0,
    })

    dae['f_list']     += [dxi, dtheta_hat]
    dae['x_list']     += [xi, theta_hat]
    dae['g_list']     += [g_omega]
    dae['y_ini_list'] += [omega_hat]
    dae['y_run_list'] += [omega_hat]
    dae['h_dict'][f"{omega_hat}"] = omega_hat
    dae['h_dict'][f"theta_hat_pll_{name}"] = theta_hat
    dae['h_dict'][f"v_d_pll_{name}"] = v_d
    dae['h_dict'][f"v_q_pll_{name}"] = v_q
    dae['params_dict'].update({
        f"{K_p}":     val_K_p,
        f"{K_i}":     val_K_i,
        f"{omega_0}": val_omega_0,
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
def _build_locked_pll_model(name='1', K_p=200.0, K_i=10000.0, omega_0=314.159,
                              A=1.0, theta_0_plant=0.0):
    """Seed a PLL with its angle perfectly aligned to a steady balanced
    three-phase input so v_q == 0 at init and the loop is at rest."""
    from pydae.core.builder.casadi_builder import MathBackend, CasadiBuilder
    from pydae.core.model.casadi_model import CasadiModel
    import numpy as np
    dae = {'name': f'test_pll_{name}',
           'f_list': [], 'x_list': [], 'g_list': [], 'y_ini_list': [], 'y_run_list': [],
           'u_ini_dict': {}, 'u_run_dict': {}, 'params_dict': {}, 'h_dict': {}, 'xy_0_dict': {}}
    backend = MathBackend(use_casadi=True)
    pll(dae, name, backend,
        {'K_p': K_p, 'K_i': K_i, 'omega_0': omega_0,
         'v_a': A * np.cos(theta_0_plant),
         'v_b': A * np.cos(theta_0_plant - 2 * np.pi / 3),
         'v_c': A * np.cos(theta_0_plant + 2 * np.pi / 3),
         'theta_0': theta_0_plant, 'xi_0': 0.0})
    model = CasadiModel(CasadiBuilder(dae).build())
    model.ini({}, dae['xy_0_dict'])
    return model


def test_locked_steady_state_v_q_is_zero():
    """At a perfectly aligned operating point, v_q == 0 and the PLL
    output frequency equals omega_0."""
    omega_0 = 314.159
    model = _build_locked_pll_model(name='1', omega_0=omega_0, theta_0_plant=0.4)
    assert abs(model.get_value('v_q_pll_1')) < 1e-9
    assert abs(model.get_value('omega_hat_pll_1') - omega_0) < 1e-9


def test_locked_v_d_equals_amplitude():
    """At lock, v_d = A (the input amplitude)."""
    A = 0.95
    model = _build_locked_pll_model(name='1', A=A, theta_0_plant=0.0)
    assert abs(model.get_value('v_d_pll_1') - A) < 1e-9


def test_input_list_aliasing():
    """input= as a 3-list binds v_a / v_b / v_c to existing signals."""
    from pydae.core.builder.casadi_builder import MathBackend, CasadiBuilder
    from pydae.core.model.casadi_model import CasadiModel
    from pydae.control.continuos.lpf1 import lpf1
    import numpy as np
    dae = {'name': 'test_pll_alias',
           'f_list': [], 'x_list': [], 'g_list': [], 'y_ini_list': [], 'y_run_list': [],
           'u_ini_dict': {}, 'u_run_dict': {}, 'params_dict': {}, 'h_dict': {}, 'xy_0_dict': {}}
    backend = MathBackend(use_casadi=True)
    A = 1.0
    lpf1(dae, 'va', backend, {'T': 0.001, 'u': A * np.cos(0.0)})
    lpf1(dae, 'vb', backend, {'T': 0.001, 'u': A * np.cos(-2 * np.pi / 3)})
    lpf1(dae, 'vc', backend, {'T': 0.001, 'u': A * np.cos( 2 * np.pi / 3)})
    pll(dae, '1', backend,
        {'K_p': 200.0, 'K_i': 10000.0, 'omega_0': 314.159,
         'theta_0': 0.0, 'xi_0': 0.0},
        input=['z_lpf1_va', 'z_lpf1_vb', 'z_lpf1_vc'])
    model = CasadiModel(CasadiBuilder(dae).build())
    model.ini({}, dae['xy_0_dict'])
    assert abs(model.get_value('v_q_pll_1')) < 1e-9
    assert abs(model.get_value('omega_hat_pll_1') - 314.159) < 1e-9


def test_validation_rejects_bad_input():
    from pydae.core.builder.casadi_builder import MathBackend
    for ia in ('just_one', ['a'], ['a', 'b', 'c', 'd']):
        dae = {'h_dict': {}, 'g_list': [], 'y_ini_list': [], 'y_run_list': [],
               'u_ini_dict': {}, 'u_run_dict': {}, 'params_dict': {},
               'xy_0_dict': {}, 'x_list': [], 'f_list': []}
        try:
            pll(dae, 't', MathBackend(use_casadi=True), {}, input=ia)
        except ValueError:
            continue
        raise AssertionError(f"should have rejected input={ia!r}")


if __name__ == '__main__':
    model = _build_locked_pll_model(name='1')
    print(f"v_d = {model.get_value('v_d_pll_1'):+.4f}  "
          f"v_q = {model.get_value('v_q_pll_1'):+.4e}  "
          f"omega_hat = {model.get_value('omega_hat_pll_1'):+.4f}")
