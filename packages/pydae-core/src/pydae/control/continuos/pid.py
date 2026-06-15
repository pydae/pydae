# -*- coding: utf-8 -*-
r"""
Created on Thu August 10 23:52:55 2026

@author: jmmauricio

Proportional-Integral-Derivative controller with filtered derivative,
output saturation, and back-calculation anti-windup (PID).

A two-state controller block extending :mod:`pydae.control.continuos.pi`
with derivative action. The derivative term is realised through a
first-order low-pass filter (a "filtered" or "real" derivative), which
is required for a well-posed DAE: a pure derivative $K_d s$ cannot be
implemented without an internal state.

**Unsaturated transfer function**

$$\frac{Z(s)}{U(s)} = K_p + \frac{K_i}{s} + \frac{K_d s}{1 + s T_f}$$

**Dynamic equations (anti-windup)**

$$\frac{d\xi}{dt} = u + K_{aw}\left(z - v\right)$$

$$\frac{dx_d}{dt} = \frac{1}{T_f}\left(u - x_d\right)$$

$$v = K_p u + K_i \xi + \frac{K_d}{T_f}\left(u - x_d\right)$$

**Algebraic equation (output saturation)**

$$0 = z - \mathrm{clip}\left(v, z_{min}, z_{max}\right)$$

$\xi$ is the integrator state (as in the plain PI). $x_d$ is the
derivative-filter state: it tracks $u$ with time constant $T_f$, so
$(u - x_d)/T_f$ approximates $\dot u$ for $T_f$ small, and the
derivative term $\frac{K_d}{T_f}(u-x_d)$ vanishes at steady state
($\dot u = 0$). $v$ is the unsaturated PID output; $z$ is the
saturated output exposed through `h_dict`. While $z = v$ the
back-calculation term vanishes; while the output is saturated
($z \neq v$), the gain $K_{aw}$ feeds the clamping error $(z-v)$ back
into the integrator, preventing unbounded windup -- exactly as in
:mod:`pydae.control.continuos.pi`. Setting $K_{aw}=0$ recovers the
plain (windup-prone) PID, and $K_d=0$ recovers the plain PI. Because
the integrator has a pole at $s=0$, $\xi$ has no unique steady-state
value of its own when $u=0$ and $z=v$; at initialization it takes
whatever value the rest of the system requires (the `xy_0_dict`
entries below are only an initial guess).
"""

import io


def descriptions():
    """
    Single source of truth for model parameters, inputs, states, and outputs.
    """
    descriptions_list = []

    # Parameters
    descriptions_list += [{"type": "Parameter", "tex": "K_p", "data": "K_p", "model": "K_p", "default": 1.0, "description": "Proportional gain", "units": "-"}]
    descriptions_list += [{"type": "Parameter", "tex": "K_i", "data": "K_i", "model": "K_i", "default": 1.0, "description": "Integral gain", "units": "1/s"}]
    descriptions_list += [{"type": "Parameter", "tex": "K_d", "data": "K_d", "model": "K_d", "default": 0.1, "description": "Derivative gain", "units": "s"}]
    descriptions_list += [{"type": "Parameter", "tex": "T_f", "data": "T_f", "model": "T_f", "default": 0.1, "description": "Derivative filter time constant", "units": "s"}]
    descriptions_list += [{"type": "Parameter", "tex": "K_{aw}", "data": "K_aw", "model": "K_aw", "default": 2.0, "description": "Anti-windup (back-calculation) gain", "units": "1/s"}]
    descriptions_list += [{"type": "Parameter", "tex": "z_{min}", "data": "z_min", "model": "z_min", "default": -10.0, "description": "Lower output limit", "units": "-"}]
    descriptions_list += [{"type": "Parameter", "tex": "z_{max}", "data": "z_max", "model": "z_max", "default": 10.0, "description": "Upper output limit", "units": "-"}]

    # Inputs
    descriptions_list += [{"type": "Input", "tex": "u", "data": "u", "model": "u", "default": 0.0, "description": "Controller input (error) signal", "units": "-"}]

    # Dynamic States
    descriptions_list += [{"type": "Dynamic State", "tex": r"\xi", "data": "", "model": "xi", "default": "", "description": "Integrator state", "units": "-"}]
    descriptions_list += [{"type": "Dynamic State", "tex": "x_d", "data": "", "model": "x_d", "default": "", "description": "Derivative-filter state (tracks u)", "units": "-"}]

    # Algebraic States
    descriptions_list += [{"type": "Algebraic State", "tex": "z", "data": "", "model": "z", "default": "", "description": "Saturated controller output, exposed in h_dict", "units": "-"}]

    return descriptions_list


def pid(dae, name, backend, data_dict=None, input=None, output=None):
    """Append a PID controller block with filtered derivative, output
    saturation, and back-calculation anti-windup to `dae`.

    Adds two dynamic states ``xi_pid_{name}`` (the integrator) and
    ``x_d_pid_{name}`` (the derivative filter) plus an algebraic output
    ``z_pid_{name}`` clamped to ``[z_min_pid_{name}, z_max_pid_{name}]``
    and exposed through `h_dict`. The gain ``K_aw_pid_{name}`` feeds
    the clamping error back into the integrator to prevent windup.

    See :mod:`pydae.control.continuos._common` for the input/output
    aliasing semantics (``input`` may be ``None`` / ``str`` /
    ``list[str]``; ``output`` may alias an existing actuator setpoint
    and is promoted to an algebraic state if it was previously a global
    input).
    """
    from pydae.control.continuos._common import combine_inputs

    data_dict = data_dict or {}

    # 1. Defaults (from descriptions())
    meta = descriptions()
    default_map = {item['data']: item['default'] for item in meta if item.get('data')}

    # 2. Data-dict overrides (fall back to descriptions() defaults)
    val_K_p   = data_dict.get('K_p',   default_map.get('K_p',   1.0))
    val_K_i   = data_dict.get('K_i',   default_map.get('K_i',   1.0))
    val_K_d   = data_dict.get('K_d',   default_map.get('K_d',   0.1))
    val_T_f   = data_dict.get('T_f',   default_map.get('T_f',   0.1))
    val_K_aw  = data_dict.get('K_aw',  default_map.get('K_aw',  2.0))
    val_z_min = data_dict.get('z_min', default_map.get('z_min', -10.0))
    val_z_max = data_dict.get('z_max', default_map.get('z_max', 10.0))
    val_u     = data_dict.get('u',     default_map.get('u',     0.0))
    val_xi    = data_dict.get('xi_0',  0.0)
    val_x_d   = data_dict.get('x_d_0', val_u)

    # 3. Input
    if input is None:
        u_sym = backend.symbols(f"u_pid_{name}")
        dae['u_ini_dict'][f"{u_sym}"] = val_u
        dae['u_run_dict'][f"{u_sym}"] = val_u
        u_expr = u_sym
    else:
        u_expr = combine_inputs(input, dae, backend, val_u)

    # 4. Dynamic states
    xi  = backend.symbols(f"xi_pid_{name}")
    x_d = backend.symbols(f"x_d_pid_{name}")

    # 5. Algebraic state (saturated output): fresh or aliased.
    if output is None:
        z = backend.symbols(f"z_pid_{name}")
        output_aliased = False
    else:
        z = backend.symbols(output)
        output_aliased = True

    # 6. Parameters
    K_p   = backend.symbols(f"K_p_pid_{name}")
    K_i   = backend.symbols(f"K_i_pid_{name}")
    K_d   = backend.symbols(f"K_d_pid_{name}")
    T_f   = backend.symbols(f"T_f_pid_{name}")
    K_aw  = backend.symbols(f"K_aw_pid_{name}")
    z_min = backend.symbols(f"z_min_pid_{name}")
    z_max = backend.symbols(f"z_max_pid_{name}")

    # 7. Unsaturated PID output, integrator with back-calculation
    #    anti-windup, derivative-filter state, and output saturation.
    v    = K_p * u_expr + K_i * xi + K_d * (u_expr - x_d) / T_f
    dxi  = u_expr + K_aw * (z - v)
    dx_d = (u_expr - x_d) / T_f
    g_z  = z - backend.hard_limits(v, z_min, z_max)

    # 8. Promote an aliased output (was a global input) to algebraic state.
    if output_aliased:
        dae['u_ini_dict'].pop(str(z), None)
        dae['u_run_dict'].pop(str(z), None)

    # 9. Initialization hints (integrator at xi_0, derivative filter at
    #    x_d_0 = u by default so the D term starts at zero, output at the
    #    clipped unsaturated value).
    val_v = val_K_p * val_u + val_K_i * val_xi + val_K_d * (val_u - val_x_d) / val_T_f
    val_z = min(max(val_v, val_z_min), val_z_max)
    dae['xy_0_dict'].update({f"{xi}": val_xi, f"{x_d}": val_x_d, f"{z}": val_z})

    # 10. Assembly
    dae['f_list']     += [dxi, dx_d]
    dae['x_list']     += [xi, x_d]
    dae['g_list']     += [g_z]
    dae['y_ini_list'] += [z]
    dae['y_run_list'] += [z]
    dae['h_dict'].update({f"{z}": z})
    dae['params_dict'].update({
        f"{K_p}":   val_K_p,   f"{K_i}":   val_K_i,   f"{K_d}":   val_K_d,
        f"{T_f}":   val_T_f,   f"{K_aw}":  val_K_aw,
        f"{z_min}": val_z_min, f"{z_max}": val_z_max,
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
def _build_pid_loop_model(name='1', K_p=2.0, K_i=0.5, K_d=0.1, T_f=0.1, K_aw=2.0,
                           z_min=-10.0, z_max=10.0, r_0=0.0):
    """Assemble a standalone CasadiModel containing a single PID block in
    unity negative feedback: ``u_pid = r - z_pid``.

    A bare integrator has no equilibrium of its own (any constant value
    of ``xi`` satisfies ``dxi/dt = 0`` whenever ``u = 0``), so
    ``model.ini()`` for an isolated PID block is structurally singular.
    Closing the loop around a reference ``r`` — exactly how a PID block
    is always used in practice — makes the initialization well-posed
    and lets the integral and derivative actions be exercised: at
    steady state ``z_pid = r``, ``xi_pid = r / K_i`` and
    ``x_d_pid = 0`` (while ``z_pid`` is unsaturated).
    """
    from pydae.core.builder.casadi_builder import MathBackend, CasadiBuilder
    from pydae.core.model.casadi_model import CasadiModel

    dae = {'name': f'test_pid_loop_{name}',
           'f_list': [], 'x_list': [],
           'g_list': [], 'y_ini_list': [], 'y_run_list': [],
           'u_ini_dict': {}, 'u_run_dict': {},
           'params_dict': {},
           'h_dict': {},
           'xy_0_dict': {}}

    backend = MathBackend(use_casadi=True)
    pid(dae, name, backend, {'K_p': K_p, 'K_i': K_i, 'K_d': K_d, 'T_f': T_f, 'K_aw': K_aw,
                             'z_min': z_min, 'z_max': z_max,
                             'u': 0.0, 'xi_0': 0.0})

    xi  = backend.symbols(f"xi_pid_{name}")
    x_d = backend.symbols(f"x_d_pid_{name}")
    z   = backend.symbols(f"z_pid_{name}")
    u   = backend.symbols(f"u_pid_{name}")

    # u_pid becomes an algebraic variable, closed onto z_pid via r.
    dae['u_ini_dict'].pop(str(u))
    dae['u_run_dict'].pop(str(u))

    r = backend.symbols(f'r_pid_{name}')
    g_loop = u - (r - z)

    dae['g_list'] += [g_loop]
    dae['y_ini_list'] += [u]
    dae['y_run_list'] += [u]
    dae['u_ini_dict'].update({f"{r}": r_0})
    dae['u_run_dict'].update({f"{r}": r_0})
    dae['xy_0_dict'].update({f"{u}": 0.0, f"{xi}": 0.0})

    model = CasadiModel(CasadiBuilder(dae).build())
    model.ini({}, dae['xy_0_dict'])

    return model


def test_closed_loop_zero_steady_state_error():
    """At steady state, integral action drives the error to zero, so the
    PID output z equals the reference r, xi = r / K_i, and the
    derivative-filter state x_d = 0 (du/dt = 0)."""
    K_p, K_i, r_0 = 2.0, 0.5, 0.0
    model = _build_pid_loop_model(name='1', K_p=K_p, K_i=K_i, r_0=r_0)

    assert abs(model.get_value('z_pid_1') - r_0) < 1e-6
    assert abs(model.get_value('u_pid_1') - 0.0) < 1e-6
    assert abs(model.get_value('xi_pid_1') - r_0 / K_i) < 1e-6
    assert abs(model.get_value('x_d_pid_1') - 0.0) < 1e-6


def test_step_response_derivative_kick_then_zero_error():
    """A step in the reference r produces an immediate jump in z driven
    by both K_p and the derivative kick K_d/T_f, then settles with zero
    steady-state error driven by K_i:
    z(0+) = (K_p + K_d/T_f) / (1 + K_p + K_d/T_f) * r,
    z(inf) = r, xi(inf) = r / K_i, x_d(inf) = 0.
    """
    name, K_p, K_i, K_d, T_f, r_1 = '1', 2.0, 0.5, 0.1, 0.1, 1.0
    model = _build_pid_loop_model(name=name, K_p=K_p, K_i=K_i, K_d=K_d, T_f=T_f, r_0=0.0)

    model.run(30.0, {f'r_pid_{name}': r_1})
    model.post()

    z = model.get_values(f'z_pid_{name}')
    xi = model.get_values(f'xi_pid_{name}')
    x_d = model.get_values(f'x_d_pid_{name}')

    K_eff = K_p + K_d / T_f
    assert abs(z[1] - K_eff / (1 + K_eff) * r_1) < 0.05  # immediate (derivative) jump
    assert abs(z[-1] - r_1) < 0.02                        # zero steady-state error
    assert abs(xi[-1] - r_1 / K_i) < 0.05
    assert abs(x_d[-1] - 0.0) < 0.02                       # filter settles to final u (=0)


def test_derivative_term_increases_initial_jump():
    """Increasing K_d increases the immediate jump in z for a step in r,
    compared to the plain-PI (K_d=0) case -- the classic derivative kick.
    """
    name, K_p, K_i, T_f, r_1 = '1', 2.0, 0.5, 0.1, 1.0

    model_pi = _build_pid_loop_model(name=name, K_p=K_p, K_i=K_i, K_d=0.0, T_f=T_f, r_0=0.0)
    model_pi.run(1.0, {f'r_pid_{name}': r_1})
    model_pi.post()
    z_pi = model_pi.get_values(f'z_pid_{name}')

    model_pid = _build_pid_loop_model(name=name, K_p=K_p, K_i=K_i, K_d=0.1, T_f=T_f, r_0=0.0)
    model_pid.run(1.0, {f'r_pid_{name}': r_1})
    model_pid.post()
    z_pid = model_pid.get_values(f'z_pid_{name}')

    assert z_pid[1] > z_pi[1]


def test_anti_windup_limits_integrator_at_saturation():
    """When the reference drives the unsaturated output beyond
    [z_min, z_max], the back-calculation term ``K_aw*(z - v)`` keeps
    the integrator state ``xi`` bounded at a finite value instead of
    winding up without limit, and ``z`` stays clamped at ``z_max``.
    """
    name = '1'
    K_p, K_i, K_aw = 2.0, 0.5, 2.0
    z_min, z_max = -1.0, 1.0
    r_1 = 2.0  # outside [z_min, z_max] -> forces saturation

    model = _build_pid_loop_model(name=name, K_p=K_p, K_i=K_i, K_aw=K_aw,
                                   z_min=z_min, z_max=z_max, r_0=0.0)

    model.run(5.0, {f'r_pid_{name}': r_1})
    model.post()

    z = model.get_values(f'z_pid_{name}')
    xi = model.get_values(f'xi_pid_{name}')

    assert abs(z[1] - z_max) < 1e-3    # output saturates immediately
    assert abs(z[-1] - z_max) < 1e-3   # remains saturated
    assert abs(xi[-1] - (-1.0)) < 0.02  # integrator settles at a finite value


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    name, K_p, K_i, K_d, T_f, r_1 = '1', 2.0, 0.5, 0.1, 0.1, 1.0
    model = _build_pid_loop_model(name=name, K_p=K_p, K_i=K_i, K_d=K_d, T_f=T_f, r_0=0.0)

    model.report_u()
    model.report_x()
    model.report_z()
    model.report_params()

    model.run(2.0, {f'r_pid_{name}': r_1})
    model.post()

    fig, ax = plt.subplots()
    ax.plot(model.Time, model.get_values(f'z_pid_{name}'), label='$z$ (PID output)')
    ax.plot(model.Time, model.get_values(f'xi_pid_{name}'), label=r'$\xi$ (integrator state)')
    ax.plot(model.Time, model.get_values(f'x_d_pid_{name}'), label='$x_d$ (derivative filter)')
    ax.axhline(r_1, color='k', ls='--', lw=0.8, label='$r$ (reference)')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Output')
    ax.legend()
    fig.savefig('pid_step_response.svg')
    print("Test completed. Plot saved as 'pid_step_response.svg'.")
