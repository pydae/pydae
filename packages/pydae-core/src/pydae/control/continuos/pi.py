# -*- coding: utf-8 -*-
r"""
Created on Thu August 10 23:52:55 2026

@author: jmmauricio

Proportional-Integral controller with output saturation and
back-calculation anti-windup (PI).

A single-state linear controller block, the workhorse of feedback
control loops (AVRs, governors, AGC, droop/secondary controllers, ...).
It produces an output proportional to its input plus the integral of
its input, so a non-zero input is eventually corrected -- as long as
the output is not saturated.

**Unsaturated transfer function**

$$\frac{Z(s)}{U(s)} = K_p + \frac{K_i}{s} = \frac{K_p s + K_i}{s}$$

**Dynamic equation (anti-windup)**

$$\frac{d\xi}{dt} = u + K_{aw}\left(z - v\right), \qquad v = K_p u + K_i \xi$$

**Algebraic equation (output saturation)**

$$0 = z - \mathrm{clip}\left(v, z_{min}, z_{max}\right)$$

$v$ is the unsaturated PI output; $z$ is the saturated output exposed
through `h_dict`. While $z = v$ (output within limits) the
back-calculation term vanishes and $\xi$ integrates $u$ as a plain PI.
While the output is saturated ($z \neq v$), the gain $K_{aw}$ feeds the
clamping error $(z - v)$ back into the integrator, driving $\xi$ toward
the value that keeps $v$ consistent with the active limit, preventing
unbounded windup. Setting $K_{aw}=0$ recovers the plain (windup-prone)
PI. Because the integrator has a pole at $s=0$, $\xi$ has no unique
steady-state value of its own when $u=0$ and $z=v$; at initialization
it takes whatever value the rest of the system requires (the
`xy_0_dict` entry below is only an initial guess).
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
    descriptions_list += [{"type": "Parameter", "tex": "K_{aw}", "data": "K_aw", "model": "K_aw", "default": 2.0, "description": "Anti-windup (back-calculation) gain", "units": "1/s"}]
    descriptions_list += [{"type": "Parameter", "tex": "z_{min}", "data": "z_min", "model": "z_min", "default": -10.0, "description": "Lower output limit", "units": "-"}]
    descriptions_list += [{"type": "Parameter", "tex": "z_{max}", "data": "z_max", "model": "z_max", "default": 10.0, "description": "Upper output limit", "units": "-"}]

    # Inputs
    descriptions_list += [{"type": "Input", "tex": "u", "data": "u", "model": "u", "default": 0.0, "description": "Controller input (error) signal", "units": "-"}]

    # Dynamic States
    descriptions_list += [{"type": "Dynamic State", "tex": r"\xi", "data": "", "model": "xi", "default": "", "description": "Integrator state", "units": "-"}]

    # Algebraic States
    descriptions_list += [{"type": "Algebraic State", "tex": "z", "data": "", "model": "z", "default": "", "description": "Saturated controller output, exposed in h_dict", "units": "-"}]

    return descriptions_list


def _resolve_pi_input(input_name, dae, backend, default_val):
    """Resolve an `input=` name to an expression usable in the PI equation.

    Resolution order:
      1. ``dae['h_dict'][input_name]`` -- use the derived expression
         directly (no new free symbol is introduced, the constraints
         that produced the h_dict entry stay in force).
      2. Name already declared as a state, algebraic variable, input,
         or parameter -- alias the cached backend symbol.
      3. Otherwise -- create the name as a new global input, added to
         both ``u_ini_dict`` and ``u_run_dict`` with `default_val`.
    """
    if input_name in dae['h_dict']:
        return dae['h_dict'][input_name]

    known = set()
    known.update(str(s) for s in dae.get('x_list', []))
    known.update(str(s) for s in dae.get('y_run_list', []))
    known.update(dae.get('u_run_dict', {}).keys())
    known.update(dae.get('params_dict', {}).keys())

    sym = backend.symbols(input_name)
    if input_name in known:
        return sym

    dae['u_ini_dict'][input_name] = default_val
    dae['u_run_dict'][input_name] = default_val
    return sym


def pi(dae, name, backend, data_dict=None, input=None, output=None):
    """Append a PI controller block with output saturation and
    back-calculation anti-windup to `dae`.

    Adds one dynamic state ``xi_pi_{name}`` (the integrator) plus an
    algebraic output ``z_pi_{name}`` clamped to ``[z_min_pi_{name},
    z_max_pi_{name}]``. The gain ``K_aw_pi_{name}`` feeds the clamping
    error back into the integrator to prevent windup.

    Parameters
    ----------
    input : str, list[str], or None
        ``None`` (default) -- create a fresh ``u_pi_{name}`` symbol and
        add it as a global input.
        ``str`` -- wire the PI input to an existing signal: either an
        entry in ``dae['h_dict']`` (its expression is substituted in
        directly), an existing state / algebraic variable / input /
        parameter (the cached backend symbol is aliased), or an
        otherwise-unknown name (created as a new global input).
        ``list[str]`` -- the inputs are combined as the conventional PI
        error signal ``u = input[0] - sum(input[1:])`` (measurement
        minus one or more reference signals). Each name is resolved
        independently via the rule above.
    output : str or None
        ``None`` (default) -- expose the saturated output as a fresh
        ``z_pi_{name}`` symbol.
        ``str`` -- alias an existing symbol; if that name was previously
        declared as a global input (e.g. an actuator setpoint such as
        ``Qext_5``), it is removed from ``u_ini_dict`` / ``u_run_dict``
        and promoted to an algebraic variable driven by the PI's
        saturation.
    """
    data_dict = data_dict or {}

    # 1. Defaults (from descriptions())
    meta = descriptions()
    default_map = {item['data']: item['default'] for item in meta if item.get('data')}

    # 2. Data-dict overrides (fall back to descriptions() defaults)
    val_K_p   = data_dict.get('K_p',   default_map.get('K_p',   1.0))
    val_K_i   = data_dict.get('K_i',   default_map.get('K_i',   1.0))
    val_K_aw  = data_dict.get('K_aw',  default_map.get('K_aw',  2.0))
    val_z_min = data_dict.get('z_min', default_map.get('z_min', -10.0))
    val_z_max = data_dict.get('z_max', default_map.get('z_max', 10.0))
    val_u     = data_dict.get('u',     default_map.get('u',     0.0))
    val_xi    = data_dict.get('xi_0', 0.0)

    # 3. PI input expression: fresh symbol, single alias, or
    #    error = input[0] - sum(input[1:]) over a list of names.
    if input is None:
        u_sym = backend.symbols(f"u_pi_{name}")
        dae['u_ini_dict'][f"{u_sym}"] = val_u
        dae['u_run_dict'][f"{u_sym}"] = val_u
        u_expr = u_sym
    else:
        names = [input] if isinstance(input, str) else list(input)
        u_expr = _resolve_pi_input(names[0], dae, backend, val_u)
        for extra in names[1:]:
            u_expr = u_expr - _resolve_pi_input(extra, dae, backend, 0.0)

    # 4. Dynamic state
    xi = backend.symbols(f"xi_pi_{name}")

    # 5. Algebraic state (saturated output): fresh symbol or aliased.
    if output is None:
        z = backend.symbols(f"z_pi_{name}")
        output_aliased = False
    else:
        z = backend.symbols(output)
        output_aliased = True

    # 6. Parameters
    K_p   = backend.symbols(f"K_p_pi_{name}")
    K_i   = backend.symbols(f"K_i_pi_{name}")
    K_aw  = backend.symbols(f"K_aw_pi_{name}")
    z_min = backend.symbols(f"z_min_pi_{name}")
    z_max = backend.symbols(f"z_max_pi_{name}")

    # 7. Unsaturated PI output, integrator with back-calculation
    #    anti-windup, and output-saturation algebraic constraint.
    v   = K_p * u_expr + K_i * xi
    dxi = u_expr + K_aw * (z - v)
    g_z = z - backend.hard_limits(v, z_min, z_max)

    # 8. Promote an aliased output (was a global input) to algebraic state.
    if output_aliased:
        dae['u_ini_dict'].pop(str(z), None)
        dae['u_run_dict'].pop(str(z), None)

    # 9. Initialization hints (integrator at xi_0, output at clipped u-only)
    val_v = val_K_p * val_u + val_K_i * val_xi
    val_z = min(max(val_v, val_z_min), val_z_max)
    dae['xy_0_dict'].update({f"{xi}": val_xi, f"{z}": val_z})

    # 10. Assembly
    dae['f_list']     += [dxi]
    dae['x_list']     += [xi]
    dae['g_list']     += [g_z]
    dae['y_ini_list'] += [z]
    dae['y_run_list'] += [z]
    dae['h_dict'].update({f"{z}": z})
    dae['params_dict'].update({
        f"{K_p}":   val_K_p,   f"{K_i}":   val_K_i,   f"{K_aw}":  val_K_aw,
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
def _build_pi_loop_model(name='1', K_p=2.0, K_i=0.5, K_aw=2.0,
                          z_min=-10.0, z_max=10.0, r_0=0.0):
    """Assemble a standalone CasadiModel containing a single PI block in
    unity negative feedback: ``u_pi = r - z_pi``.

    A bare integrator has no equilibrium of its own (any constant value
    of ``xi`` satisfies ``dxi/dt = 0`` whenever ``u = 0``), so
    ``model.ini()`` for an isolated PI block is structurally singular.
    Closing the loop around a reference ``r`` — exactly how a PI block
    is always used in practice — makes the initialization well-posed
    and lets the integral action be exercised: at steady state
    ``z_pi = r`` and ``xi_pi = r / K_i`` (while ``z_pi`` is unsaturated).
    """
    from pydae.core.builder.casadi_builder import MathBackend, CasadiBuilder
    from pydae.core.model.casadi_model import CasadiModel

    dae = {'name': f'test_pi_loop_{name}',
           'f_list': [], 'x_list': [],
           'g_list': [], 'y_ini_list': [], 'y_run_list': [],
           'u_ini_dict': {}, 'u_run_dict': {},
           'params_dict': {},
           'h_dict': {},
           'xy_0_dict': {}}

    backend = MathBackend(use_casadi=True)
    pi(dae, name, backend, {'K_p': K_p, 'K_i': K_i, 'K_aw': K_aw,
                            'z_min': z_min, 'z_max': z_max,
                            'u': 0.0, 'xi_0': 0.0})

    xi = backend.symbols(f"xi_pi_{name}")
    z = backend.symbols(f"z_pi_{name}")
    u = backend.symbols(f"u_pi_{name}")

    # u_pi becomes an algebraic variable, closed onto z_pi via r.
    dae['u_ini_dict'].pop(str(u))
    dae['u_run_dict'].pop(str(u))

    r = backend.symbols(f'r_pi_{name}')
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
    PI output z equals the reference r and xi = r / K_i."""
    K_p, K_i, r_0 = 2.0, 0.5, 0.0
    model = _build_pi_loop_model(name='1', K_p=K_p, K_i=K_i, r_0=r_0)

    assert abs(model.get_value('z_pi_1') - r_0) < 1e-6
    assert abs(model.get_value('u_pi_1') - 0.0) < 1e-6
    assert abs(model.get_value('xi_pi_1') - r_0 / K_i) < 1e-6


def test_step_response_proportional_jump_then_zero_error():
    """A step in the reference r produces an immediate jump in z driven
    by K_p, then settles with zero steady-state error driven by K_i:
    z(0+) = K_p/(1+K_p) * r, z(inf) = r, xi(inf) = r / K_i.
    """
    name, K_p, K_i, r_1 = '1', 2.0, 0.5, 1.0
    model = _build_pi_loop_model(name=name, K_p=K_p, K_i=K_i, r_0=0.0)

    model.run(30.0, {f'r_pi_{name}': r_1})
    model.post()

    z = model.get_values(f'z_pi_{name}')
    xi = model.get_values(f'xi_pi_{name}')

    assert abs(z[1] - K_p / (1 + K_p) * r_1) < 0.05  # immediate jump
    assert abs(z[-1] - r_1) < 0.02                    # zero steady-state error
    assert abs(xi[-1] - r_1 / K_i) < 0.05


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

    model = _build_pi_loop_model(name=name, K_p=K_p, K_i=K_i, K_aw=K_aw,
                                  z_min=z_min, z_max=z_max, r_0=0.0)

    model.run(5.0, {f'r_pi_{name}': r_1})
    model.post()

    z = model.get_values(f'z_pi_{name}')
    xi = model.get_values(f'xi_pi_{name}')

    assert abs(z[1] - z_max) < 1e-3    # output saturates immediately
    assert abs(z[-1] - z_max) < 1e-3   # remains saturated
    assert abs(xi[-1] - (-1.0)) < 0.02  # integrator settles at a finite value


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    name, K_p, K_i, K_aw = '1', 2.0, 0.5, 2.0
    z_min, z_max, r_1 = -1.0, 1.0, 2.0
    model = _build_pi_loop_model(name=name, K_p=K_p, K_i=K_i, K_aw=K_aw,
                                  z_min=z_min, z_max=z_max, r_0=0.0)

    model.report_u()
    model.report_x()
    model.report_z()
    model.report_params()

    model.run(5.0, {f'r_pi_{name}': r_1})
    model.post()

    fig, ax = plt.subplots()
    ax.plot(model.Time, model.get_values(f'z_pi_{name}'), label='$z$ (saturated output)')
    ax.plot(model.Time, model.get_values(f'xi_pi_{name}'), label=r'$\xi$ (integrator state)')
    ax.axhline(r_1, color='k', ls='--', lw=0.8, label='$r$ (reference)')
    ax.axhline(z_max, color='gray', ls=':', lw=0.8, label='$z_{max}$')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Output')
    ax.legend()
    fig.savefig('pi_anti_windup_response.svg')
    print("Test completed. Plot saved as 'pi_anti_windup_response.svg'.")
