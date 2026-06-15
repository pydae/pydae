# -*- coding: utf-8 -*-
r"""
Created on Sun June 15 13:27:00 2026

@author: jmmauricio

Generic continuous transfer function (TF).

Realises an arbitrary proper rational transfer function

$$H(s) = \frac{N(s)}{D(s)} =
   \frac{b_m s^m + b_{m-1} s^{m-1} + \dots + b_1 s + b_0}
        {a_n s^n + a_{n-1} s^{n-1} + \dots + a_1 s + a_0}$$

specified by two lists of coefficients (the same convention as
MATLAB ``tf`` and ``scipy.signal.TransferFunction`` -- **descending**
powers of $s$):

* ``NUM = [b_m, b_{m-1}, ..., b_1, b_0]``
* ``DEN = [a_n, a_{n-1}, ..., a_1, a_0]``

The block must be proper (``len(NUM) <= len(DEN)``) and ``DEN[0]``
must be non-zero. Both lists are first normalised so ``DEN`` becomes
monic (every coefficient divided by ``DEN[0]``), then realised in
**controllable canonical form** with ``n = len(DEN) - 1`` chain
states $x_1, \dots, x_n$:

$$\dot x_i = x_{i+1}\;\;(i<n), \qquad
   \dot x_n = u - \sum_{i=0}^{n-1} \alpha_i\,x_{i+1}$$

$$z = D\,u + \sum_{i=0}^{n-1} d_i\,x_{i+1}$$

where $\alpha_i$ are the (monic) denominator coefficients in
**ascending** order ($\alpha_n = 1$), $D$ is the direct
feedthrough (non-zero only for $m=n$), and the output weights
$d_i = \beta_i - D\,\alpha_i$ come from $\beta_i$, the (normalised)
numerator coefficients padded to length $n+1$ in ascending order.

**Examples**

* First-order LPF $1/(Ts+1)$ -- ``NUM=[1]``, ``DEN=[T, 1]``
* Lead-lag $(T_{num}s+1)/(T_{den}s+1)$ -- ``NUM=[T_num, 1]``, ``DEN=[T_den, 1]``
* Washout $T_{wo}s/(T_{wo}s+1)$ -- ``NUM=[T_wo, 0]``, ``DEN=[T_wo, 1]``
* Second-order LPF $\omega_n^2/(s^2 + 2\zeta\omega_n s + \omega_n^2)$
  -- ``NUM=[omega_n**2]``, ``DEN=[1, 2*zeta*omega_n, omega_n**2]``

Coefficients are baked into the symbolic equations at build time --
changing ``NUM`` or ``DEN`` requires a model rebuild. For runtime
tuning of a single time constant or gain, use the purpose-built
``lpf1`` / ``lpf2`` / ``leadlag`` / ``washout`` blocks instead.
"""

import io


def descriptions():
    """
    Single source of truth for model parameters, inputs, states, and outputs.

    The structural coefficients ``NUM`` / ``DEN`` are listed under
    *Parameters* but are **structural** -- they define the block's
    order at build time and are not exposed as runtime-tunable
    ``params_dict`` entries. The number of dynamic states is
    ``n = len(DEN) - 1``; the entry below names the first state for
    documentation purposes.
    """
    descriptions_list = []

    # Structural parameters (set at build time, baked into the equations)
    descriptions_list += [{"type": "Parameter", "tex": "\\mathrm{NUM}", "data": "NUM", "model": "NUM", "default": [1.0], "description": "Numerator coefficients, descending powers of s: [b_m, ..., b_0]", "units": "-"}]
    descriptions_list += [{"type": "Parameter", "tex": "\\mathrm{DEN}", "data": "DEN", "model": "DEN", "default": [1.0, 1.0], "description": "Denominator coefficients, descending powers of s: [a_n, ..., a_0]", "units": "-"}]

    # Inputs
    descriptions_list += [{"type": "Input", "tex": "u", "data": "u", "model": "u", "default": 0.0, "description": "Transfer function input signal", "units": "-"}]

    # Dynamic States (n states, x_1 named here for documentation)
    descriptions_list += [{"type": "Dynamic State", "tex": "x_i", "data": "", "model": "x_i (i=1..n)", "default": "", "description": "Controllable-canonical chain state (n = len(DEN)-1)", "units": "-"}]

    # Algebraic States
    descriptions_list += [{"type": "Algebraic State", "tex": "z", "data": "", "model": "z", "default": "", "description": "Filter output H(s)*u (exposed in h_dict)", "units": "-"}]

    return descriptions_list


def tf(dae, name, backend, data_dict=None, input=None, output=None):
    """Append a generic continuous transfer function block to `dae`.

    Adds ``n = len(DEN) - 1`` dynamic states ``x_1_tf_{name}`` ..
    ``x_n_tf_{name}`` (the controllable-canonical chain), driven by
    ``u_tf_{name}`` (or the aliased ``input``), plus an algebraic
    output ``z_tf_{name}`` (or the aliased ``output``) exposed
    through `h_dict`.

    Parameters
    ----------
    data_dict : dict
        Must contain ``NUM`` and ``DEN`` coefficient lists (descending
        powers of $s$); both default to a trivial pass-through if
        omitted. Optional ``u`` sets the steady-state input used for
        the xy_0 hint.
    input : str, list[str], or None
        See :mod:`pydae.control.continuos._common`.
    output : str or None
        See :mod:`pydae.control.continuos._common`. If ``output``
        aliases an existing global input, it is promoted to an
        algebraic variable driven by this block's output.
    """
    from pydae.control.continuos._common import combine_inputs

    data_dict = data_dict or {}

    # 1. Defaults (from descriptions())
    meta = descriptions()
    default_map = {item['data']: item['default'] for item in meta if item.get('data')}

    NUM = list(data_dict.get('NUM', default_map.get('NUM', [1.0])))
    DEN = list(data_dict.get('DEN', default_map.get('DEN', [1.0, 1.0])))
    val_u = data_dict.get('u', default_map.get('u', 0.0))

    # 2. Validate
    if not DEN:
        raise ValueError("DEN must be a non-empty list of coefficients")
    if DEN[0] == 0:
        raise ValueError("DEN leading coefficient must be non-zero")
    if not NUM:
        raise ValueError("NUM must be a non-empty list of coefficients")
    if len(NUM) > len(DEN):
        raise ValueError(
            f"transfer function must be proper: len(NUM)={len(NUM)} > "
            f"len(DEN)={len(DEN)}"
        )
    if len(DEN) < 2:
        raise ValueError(
            "tf requires order >= 1 (len(DEN) >= 2). For a pure gain, "
            "use the summation block with K_2 = 0."
        )

    n = len(DEN) - 1   # order

    # 3. Normalise to monic DEN, then convert to ascending order so
    #    the recursion alpha[i]*x_{i+1} reads naturally.
    a_n = float(DEN[0])
    den_norm = [d / a_n for d in DEN]
    num_norm = [b / a_n for b in NUM]
    # Pad NUM to length n+1 with leading zeros (strictly proper).
    num_padded = [0.0] * (n + 1 - len(num_norm)) + num_norm
    alpha = list(reversed(den_norm))   # alpha[0]=a_0/a_n, ..., alpha[n]=1
    beta  = list(reversed(num_padded)) # beta[0]=b_0/a_n,  ..., beta[n]=b_n/a_n
    D = beta[n]                        # direct feedthrough (0 unless m == n)
    d_coeffs = [beta[i] - D * alpha[i] for i in range(n)]

    # 4. Input: fresh u_tf_{name}, single alias, or combined list.
    if input is None:
        u = backend.symbols(f"u_tf_{name}")
        dae['u_ini_dict'][f"{u}"] = val_u
        dae['u_run_dict'][f"{u}"] = val_u
        u_expr = u
    else:
        u_expr = combine_inputs(input, dae, backend, val_u)

    # 5. Chain of dynamic states (CCF). The degenerate pure-gain case
    #    (n = 0) is rejected up front in step 2 above.
    x_syms = [backend.symbols(f"x_{i + 1}_tf_{name}") for i in range(n)]

    # 6. Output: fresh z_tf_{name} or aliased.
    if output is None:
        z = backend.symbols(f"z_tf_{name}")
        output_aliased = False
    else:
        z = backend.symbols(output)
        output_aliased = True

    # 7. Equations.
    #    f-chain (CCF):
    #      dx_i/dt = x_{i+1}                     for i in [1, n-1]
    #      dx_n/dt = u - sum_{i=0..n-1} alpha_i * x_{i+1}
    #    g (output):
    #      0 = z - (D * u + sum_{i=0..n-1} d_i * x_{i+1})
    f_eqs = []
    for i in range(n - 1):
        f_eqs.append(x_syms[i + 1])
    rhs_last = u_expr
    for j in range(n):
        rhs_last = rhs_last - alpha[j] * x_syms[j]
    f_eqs.append(rhs_last)

    out_expr = D * u_expr
    for i in range(n):
        out_expr = out_expr + d_coeffs[i] * x_syms[i]
    g_z = z - out_expr

    # 8. Promote an aliased output (was a global input) to algebraic state.
    if output_aliased:
        dae['u_ini_dict'].pop(str(z), None)
        dae['u_run_dict'].pop(str(z), None)

    # 9. Steady-state init hints.
    #    From f-chain at rest: x_2 = ... = x_n = 0, alpha_0*x_1 = u
    #    -> x_1 = u / alpha_0 (well defined: alpha_0 = a_0/a_n; the
    #    only way alpha_0 = 0 is a pole at the origin, in which case
    #    the steady-state value of x_1 is undetermined and we fall
    #    back to 0 -- the closed loop / external constraints decide).
    x_1_ss = val_u / alpha[0] if alpha[0] != 0 else 0.0
    val_x_list = [x_1_ss] + [0.0] * (n - 1)
    # z at steady state = u * H(0) = u * beta[0] / alpha[0] (when alpha_0 != 0)
    val_z = D * val_u + d_coeffs[0] * x_1_ss
    for x_sym, val in zip(x_syms, val_x_list):
        dae['xy_0_dict'][f"{x_sym}"] = val
    dae['xy_0_dict'][f"{z}"] = val_z

    # 10. Assembly. Coefficients are baked into the equations as
    #     numeric constants; no params_dict entries are added.
    dae['f_list']     += f_eqs
    dae['x_list']     += x_syms
    dae['g_list']     += [g_z]
    dae['y_ini_list'] += [z]
    dae['y_run_list'] += [z]
    dae['h_dict'].update({f"{z}": z})

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
def _build_tf_model(name='1', NUM=None, DEN=None, u_0=0.0):
    """Assemble a standalone CasadiModel containing a single TF block."""
    from pydae.core.builder.casadi_builder import MathBackend, CasadiBuilder
    from pydae.core.model.casadi_model import CasadiModel

    NUM = NUM if NUM is not None else [1.0]
    DEN = DEN if DEN is not None else [1.0, 1.0]

    dae = {'name': f'test_tf_{name}',
           'f_list': [], 'x_list': [],
           'g_list': [], 'y_ini_list': [], 'y_run_list': [],
           'u_ini_dict': {}, 'u_run_dict': {},
           'params_dict': {},
           'h_dict': {},
           'xy_0_dict': {}}

    backend = MathBackend(use_casadi=True)
    tf(dae, name, backend, {'NUM': NUM, 'DEN': DEN, 'u': u_0})

    model = CasadiModel(CasadiBuilder(dae).build())
    model.ini({}, dae['xy_0_dict'])

    return model


def test_first_order_lpf_matches_lpf1():
    """H(s) = 1/(T*s+1): steady-state z = u; step response reaches
    ~63.2% of the step amplitude after one time constant.
    """
    import numpy as np
    name, T = '1', 0.5
    model = _build_tf_model(name=name, NUM=[1.0], DEN=[T, 1.0], u_0=0.0)

    assert abs(model.get_value(f'z_tf_{name}')) < 1e-9

    model.run(5 * T, {f'u_tf_{name}': 1.0})
    model.post()
    t = model.Time
    z = model.get_values(f'z_tf_{name}')
    idx_T = np.argmin(np.abs(t - T))
    assert abs(z[idx_T] - (1 - np.exp(-1))) < 0.02
    assert abs(z[-1] - 1.0) < 0.01


def test_pure_gain_via_lead_with_equal_time_constants():
    """H(s) = (T*s+1)/(T*s+1) = 1: a proper (m = n = 1) transfer
    function with direct feedthrough D = 1. Steady-state z = u and a
    step in u produces an immediate jump to the new value (no lag)."""
    name = '1'
    model = _build_tf_model(name=name, NUM=[1.0, 1.0], DEN=[1.0, 1.0], u_0=0.4)
    assert abs(model.get_value(f'z_tf_{name}') - 0.4) < 1e-9
    model.run(1.0, {f'u_tf_{name}': 0.9})
    model.post()
    z = model.get_values(f'z_tf_{name}')
    assert abs(z[1] - 0.9) < 1e-3      # immediate jump (D-feedthrough)
    assert abs(z[-1] - 0.9) < 1e-9


def test_washout_settles_to_zero():
    """H(s) = T*s/(T*s+1): a constant input has zero steady-state
    output; a step excites a transient that decays back to zero."""
    name, T = '1', 1.0
    # Steady-state value of u doesn't matter (washout passes only AC),
    # but the block initialises with u_0 = 0 so x_1 starts at 0 too.
    model = _build_tf_model(name=name, NUM=[T, 0.0], DEN=[T, 1.0], u_0=0.0)
    assert abs(model.get_value(f'z_tf_{name}')) < 1e-9
    model.run(10 * T, {f'u_tf_{name}': 1.0})
    model.post()
    z = model.get_values(f'z_tf_{name}')
    assert max(abs(z)) > 0.5            # transient excitation
    assert abs(z[-1]) < 0.02            # washes out


def test_second_order_lpf_steady_state_unity_gain():
    """H(s) = omega_n^2 / (s^2 + 2*zeta*omega_n*s + omega_n^2): unity
    DC gain, so steady-state z = u for any well-damped omega_n / zeta."""
    name, omega_n, zeta, u_0 = '1', 5.0, 0.7, 0.3
    model = _build_tf_model(
        name=name,
        NUM=[omega_n ** 2],
        DEN=[1.0, 2 * zeta * omega_n, omega_n ** 2],
        u_0=u_0,
    )
    assert abs(model.get_value(f'z_tf_{name}') - u_0) < 1e-9


def test_input_aliasing_via_h_dict():
    """`input=` referring to an h_dict expression substitutes that
    expression directly (no new free symbol)."""
    from pydae.core.builder.casadi_builder import MathBackend, CasadiBuilder
    from pydae.core.model.casadi_model import CasadiModel
    from pydae.control.continuos.lpf1 import lpf1

    dae = {'name': 'test_tf_alias',
           'f_list': [], 'x_list': [],
           'g_list': [], 'y_ini_list': [], 'y_run_list': [],
           'u_ini_dict': {}, 'u_run_dict': {},
           'params_dict': {}, 'h_dict': {}, 'xy_0_dict': {}}
    backend = MathBackend(use_casadi=True)

    lpf1(dae, 'meas', backend, {'T': 0.05, 'u': 0.6})
    # Chain a TF (a lead) onto the LPF output via h_dict aliasing.
    tf(dae, 'cmp', backend, {'NUM': [0.2, 1.0], 'DEN': [0.05, 1.0], 'u': 0.6},
       input='z_lpf1_meas')

    model = CasadiModel(CasadiBuilder(dae).build())
    model.ini({}, dae['xy_0_dict'])
    # DC gain of (0.2 s + 1) / (0.05 s + 1) is 1 -> z_tf_cmp = 0.6.
    assert abs(model.get_value('z_lpf1_meas') - 0.6) < 1e-9
    assert abs(model.get_value('z_tf_cmp') - 0.6) < 1e-9


if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt

    name, T = '1', 0.5
    model = _build_tf_model(name=name, NUM=[1.0], DEN=[T, 1.0], u_0=0.0)
    model.report_x()
    model.report_z()
    model.run(5 * T, {f'u_tf_{name}': 1.0})
    model.post()

    fig, ax = plt.subplots()
    ax.plot(model.Time, model.get_values(f'z_tf_{name}'), label='$z$ (TF output)')
    ax.axhline(1.0, color='k', ls='--', lw=0.8, label='$u$ (step input)')
    ax.axvline(T, color='gray', ls=':', lw=0.8, label='$t = T$')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Output')
    ax.legend()
    fig.savefig('tf_step_response.svg')
    print("Test completed. Plot saved as 'tf_step_response.svg'.")
