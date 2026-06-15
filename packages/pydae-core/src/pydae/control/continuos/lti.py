# -*- coding: utf-8 -*-
r"""
Created on Sun June 15 13:32:00 2026

@author: jmmauricio

Generic continuous LTI state-space system (LTI).

Realises an arbitrary multivariable linear time-invariant system

$$\dot x = A\,x + B\,u, \qquad z = C\,x + D\,u$$

specified by the four real matrices ``A``, ``B``, ``C``, ``D`` passed
as nested lists in ``data_dict``. The block's dimensions are derived
from the shapes:

* ``A`` is :math:`n \times n` (square),
* ``B`` is :math:`n \times m`,
* ``C`` is :math:`p \times n`,
* ``D`` is :math:`p \times m`.

There are then ``n`` dynamic states ``x_1_lti_{name}`` ..
``x_n_lti_{name}``, ``m`` inputs ``u_1_lti_{name}`` ..
``u_m_lti_{name}`` (when ``input=None``), and ``p`` algebraic outputs
``z_1_lti_{name}`` .. ``z_p_lti_{name}`` (when ``output=None``).

**Equivalence to a transfer function.** A SISO realisation of
:math:`H(s) = 1/(Ts+1)` is ``A=[[-1/T]]``, ``B=[[1/T]]``,
``C=[[1]]``, ``D=[[0]]``. Use the dedicated
:mod:`pydae.control.continuos.tf` block when a frequency-domain
description is more natural; ``lti`` is the right choice for
truly multivariable systems and for plants already given in
state-space form (e.g. a reduced-order model).

Matrix entries are baked into the symbolic equations at build time
as numeric constants -- changing ``A`` / ``B`` / ``C`` / ``D``
requires a model rebuild.
"""

import io


def descriptions():
    """
    Single source of truth for model parameters, inputs, states, and outputs.

    The structural matrices ``A`` / ``B`` / ``C`` / ``D`` are listed
    under *Parameters* but are **structural** -- they fix the block's
    dimensions at build time and are not exposed as runtime-tunable
    ``params_dict`` entries. The count of states / inputs / outputs
    derives from the matrix shapes; the entries below name the first
    of each kind for documentation purposes.
    """
    descriptions_list = []

    # Structural parameters (set at build time, baked into the equations)
    descriptions_list += [{"type": "Parameter", "tex": "A", "data": "A", "model": "A", "default": [[-1.0]], "description": "State matrix, n x n (nested list of floats)", "units": "-"}]
    descriptions_list += [{"type": "Parameter", "tex": "B", "data": "B", "model": "B", "default": [[1.0]],  "description": "Input matrix, n x m",                                "units": "-"}]
    descriptions_list += [{"type": "Parameter", "tex": "C", "data": "C", "model": "C", "default": [[1.0]],  "description": "Output matrix, p x n",                               "units": "-"}]
    descriptions_list += [{"type": "Parameter", "tex": "D", "data": "D", "model": "D", "default": [[0.0]],  "description": "Direct-feedthrough matrix, p x m",                   "units": "-"}]

    # Inputs (m channels, u_1 named here for documentation)
    descriptions_list += [{"type": "Input", "tex": "u_k", "data": "u_k", "model": "u_k (k=1..m)", "default": 0.0, "description": "Input channel k",  "units": "-"}]

    # Dynamic States (n states, x_1 named here for documentation)
    descriptions_list += [{"type": "Dynamic State", "tex": "x_i", "data": "", "model": "x_i (i=1..n)", "default": "", "description": "State component i", "units": "-"}]

    # Algebraic States (p outputs, z_1 named here for documentation)
    descriptions_list += [{"type": "Algebraic State", "tex": "z_j", "data": "", "model": "z_j (j=1..p)", "default": "", "description": "Output component j (z = C*x + D*u, exposed in h_dict)", "units": "-"}]

    return descriptions_list


def _check_matrix(M, name, expected_rows, expected_cols):
    """Sanity-check a ``data_dict`` matrix and return it as a nested list."""
    if M is None:
        raise ValueError(f"lti: matrix {name} is required")
    rows = [list(r) for r in M]
    if not rows:
        raise ValueError(f"lti: matrix {name} is empty")
    n_rows = len(rows)
    n_cols = len(rows[0])
    if any(len(r) != n_cols for r in rows):
        raise ValueError(f"lti: matrix {name} has rows of unequal length")
    if expected_rows is not None and n_rows != expected_rows:
        raise ValueError(
            f"lti: matrix {name} has {n_rows} rows, expected {expected_rows}"
        )
    if expected_cols is not None and n_cols != expected_cols:
        raise ValueError(
            f"lti: matrix {name} has {n_cols} cols, expected {expected_cols}"
        )
    return rows


def lti(dae, name, backend, data_dict=None, input=None, output=None):
    """Append a generic LTI state-space block to `dae`.

    Adds ``n`` dynamic states (the components of ``x``), ``m`` global
    inputs (when ``input=None``) and ``p`` algebraic outputs (when
    ``output=None``).

    Parameters
    ----------
    data_dict : dict
        Required keys ``A``, ``B``, ``C``, ``D`` as nested lists of
        floats. Optional ``u_k`` (``k = 1..m``) sets the steady-state
        value of input ``k`` used for the xy_0 hint.
    input : str, list[str], or None
        ``None`` -- create ``m`` fresh global inputs ``u_1_lti_{name}``,
        ``..``, ``u_m_lti_{name}``.
        ``str`` -- only valid when ``m == 1`` (SISO/MISO with a single
        input column); resolved via
        :func:`pydae.control.continuos._common.resolve_input`.
        ``list[str]`` -- must contain exactly ``m`` names; each is
        resolved independently and bound to the corresponding ``u_k``.
        Note that, unlike the PI block, the list **is not** combined
        as an error signal -- each entry is one channel.
    output : str, list[str], or None
        Same convention as ``input`` but for the ``p`` output channels.
        When an output name aliases an existing global input, it is
        popped from ``u_*_dict`` and promoted to an algebraic variable.
    """
    import numpy as np
    from pydae.control.continuos._common import resolve_input

    data_dict = data_dict or {}

    # 1. Defaults (from descriptions())
    meta = descriptions()
    default_map = {item['data']: item['default'] for item in meta if item.get('data')}

    A_in = data_dict.get('A', default_map.get('A'))
    B_in = data_dict.get('B', default_map.get('B'))
    C_in = data_dict.get('C', default_map.get('C'))
    D_in = data_dict.get('D', default_map.get('D'))

    # 2. Validate shapes. A defines n; B's column count defines m;
    #    C's row count defines p; D must match (p, m).
    A = _check_matrix(A_in, 'A', expected_rows=None, expected_cols=None)
    n = len(A)
    if len(A[0]) != n:
        raise ValueError(f"lti: A must be square, got {n} x {len(A[0])}")
    B = _check_matrix(B_in, 'B', expected_rows=n, expected_cols=None)
    m = len(B[0])
    C = _check_matrix(C_in, 'C', expected_rows=None, expected_cols=n)
    p = len(C)
    D = _check_matrix(D_in, 'D', expected_rows=p, expected_cols=m)

    if n < 1:
        raise ValueError("lti: at least one state required (n >= 1)")

    val_u = [float(data_dict.get(f'u_{k + 1}', 0.0)) for k in range(m)]

    # 3. Input channels: fresh symbols (one per column of B), aliased
    #    list, or a single aliased name (only valid for m == 1).
    u_exprs = []
    if input is None:
        for k in range(m):
            u_sym = backend.symbols(f"u_{k + 1}_lti_{name}")
            dae['u_ini_dict'][f"{u_sym}"] = val_u[k]
            dae['u_run_dict'][f"{u_sym}"] = val_u[k]
            u_exprs.append(u_sym)
    elif isinstance(input, str):
        if m != 1:
            raise ValueError(
                f"lti: a single input name was given but B has m = {m} "
                f"columns; pass a list of {m} names"
            )
        u_exprs = [resolve_input(input, dae, backend, val_u[0])]
    else:
        names = list(input)
        if len(names) != m:
            raise ValueError(
                f"lti: input list has {len(names)} entries, B has m = {m} "
                f"columns"
            )
        for k, nm in enumerate(names):
            u_exprs.append(resolve_input(nm, dae, backend, val_u[k]))

    # 4. Dynamic states (one per row of A).
    x_syms = [backend.symbols(f"x_{i + 1}_lti_{name}") for i in range(n)]

    # 5. Output channels: fresh z symbols, aliased list, or a single
    #    aliased name (only valid for p == 1).
    z_syms = []
    z_aliased_flags = []
    if output is None:
        for j in range(p):
            z_syms.append(backend.symbols(f"z_{j + 1}_lti_{name}"))
            z_aliased_flags.append(False)
    elif isinstance(output, str):
        if p != 1:
            raise ValueError(
                f"lti: a single output name was given but C has p = {p} "
                f"rows; pass a list of {p} names"
            )
        z_syms.append(backend.symbols(output))
        z_aliased_flags.append(True)
    else:
        names_out = list(output)
        if len(names_out) != p:
            raise ValueError(
                f"lti: output list has {len(names_out)} entries, C has "
                f"p = {p} rows"
            )
        for nm in names_out:
            z_syms.append(backend.symbols(nm))
            z_aliased_flags.append(True)

    # 6. Equations:
    #      dx_i/dt = sum_j A[i][j] * x_j + sum_k B[i][k] * u_k
    #      0       = z_j - (sum_i C[j][i] * x_i + sum_k D[j][k] * u_k)
    #
    #    Zero coefficients are skipped from the symbolic sum so the
    #    generated expression doesn't carry useless `+ 0 * x` terms.
    def _row_combination(coeffs, syms):
        terms = [c * s for c, s in zip(coeffs, syms) if c != 0]
        if not terms:
            return 0
        expr = terms[0]
        for t in terms[1:]:
            expr = expr + t
        return expr

    f_eqs = []
    for i in range(n):
        rhs = _row_combination(A[i], x_syms) + _row_combination(B[i], u_exprs)
        f_eqs.append(rhs)

    g_eqs = []
    for j in range(p):
        out_expr = _row_combination(C[j], x_syms) + _row_combination(D[j], u_exprs)
        g_eqs.append(z_syms[j] - out_expr)

    # 7. Promote any aliased outputs that were previously global inputs.
    for z_sym, aliased in zip(z_syms, z_aliased_flags):
        if aliased:
            dae['u_ini_dict'].pop(str(z_sym), None)
            dae['u_run_dict'].pop(str(z_sym), None)

    # 8. Steady-state init hints.
    #    From A*x + B*u_0 = 0 we get x_ss = -inv(A) * B * u_0 when A is
    #    invertible; otherwise fall back to x_ss = 0 (an integrator's
    #    steady-state value is set by the surrounding constraints, not
    #    by the block in isolation).
    u_0 = np.array(val_u, dtype=float)
    A_np = np.array(A, dtype=float)
    B_np = np.array(B, dtype=float)
    C_np = np.array(C, dtype=float)
    D_np = np.array(D, dtype=float)
    try:
        x_ss = -np.linalg.solve(A_np, B_np @ u_0)
    except np.linalg.LinAlgError:
        x_ss = np.zeros(n)
    z_ss = C_np @ x_ss + D_np @ u_0

    for x_sym, val in zip(x_syms, x_ss):
        dae['xy_0_dict'][f"{x_sym}"] = float(val)
    for z_sym, val in zip(z_syms, z_ss):
        dae['xy_0_dict'][f"{z_sym}"] = float(val)

    # 9. Assembly. Coefficients are baked into the equations as numeric
    #    constants; no params_dict entries are added.
    dae['f_list']     += f_eqs
    dae['x_list']     += x_syms
    dae['g_list']     += g_eqs
    dae['y_ini_list'] += list(z_syms)
    dae['y_run_list'] += list(z_syms)
    for z_sym in z_syms:
        dae['h_dict'][f"{z_sym}"] = z_sym

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
def _build_lti_model(name='1', A=None, B=None, C=None, D=None, u_0=None):
    """Assemble a standalone CasadiModel containing a single LTI block."""
    from pydae.core.builder.casadi_builder import MathBackend, CasadiBuilder
    from pydae.core.model.casadi_model import CasadiModel

    if A is None: A = [[-1.0]]
    if B is None: B = [[1.0]]
    if C is None: C = [[1.0]]
    if D is None: D = [[0.0]]
    if u_0 is None: u_0 = [0.0] * len(B[0])

    dae = {'name': f'test_lti_{name}',
           'f_list': [], 'x_list': [],
           'g_list': [], 'y_ini_list': [], 'y_run_list': [],
           'u_ini_dict': {}, 'u_run_dict': {},
           'params_dict': {},
           'h_dict': {},
           'xy_0_dict': {}}

    backend = MathBackend(use_casadi=True)
    dd = {'A': A, 'B': B, 'C': C, 'D': D}
    for k, val in enumerate(u_0):
        dd[f'u_{k+1}'] = val
    lti(dae, name, backend, dd)

    model = CasadiModel(CasadiBuilder(dae).build())
    model.ini({}, dae['xy_0_dict'])

    return model


def test_siso_first_order_matches_lpf1():
    """A=[[-1/T]], B=[[1/T]], C=[[1]], D=[[0]] realises H(s)=1/(Ts+1)."""
    import numpy as np
    name, T = '1', 0.5
    model = _build_lti_model(
        name=name,
        A=[[-1.0 / T]], B=[[1.0 / T]], C=[[1.0]], D=[[0.0]], u_0=[0.0],
    )
    assert abs(model.get_value(f'z_1_lti_{name}')) < 1e-9
    model.run(5 * T, {f'u_1_lti_{name}': 1.0})
    model.post()
    z = model.get_values(f'z_1_lti_{name}')
    t = model.Time
    idx_T = np.argmin(np.abs(t - T))
    assert abs(z[idx_T] - (1 - np.exp(-1))) < 0.02
    assert abs(z[-1] - 1.0) < 0.01


def test_feedthrough_d_term_immediate_jump():
    """D != 0 produces an immediate output jump on input change, with no
    state required. A=[[-1]], B=[[1]], C=[[0]], D=[[2]]: z = 2*u."""
    name = '1'
    model = _build_lti_model(
        name=name,
        A=[[-1.0]], B=[[1.0]], C=[[0.0]], D=[[2.0]], u_0=[0.3],
    )
    assert abs(model.get_value(f'z_1_lti_{name}') - 0.6) < 1e-9
    model.run(1.0, {f'u_1_lti_{name}': 0.5})
    model.post()
    assert abs(model.get_value(f'z_1_lti_{name}') - 1.0) < 1e-9


def test_mimo_two_parallel_lpfs():
    """Block-diagonal 2-input 2-output system. Each channel is an
    independent 1/(Ts+1); cross-coupling stays zero."""
    name = '1'
    T1, T2 = 0.4, 0.7
    model = _build_lti_model(
        name=name,
        A=[[-1.0 / T1, 0.0], [0.0, -1.0 / T2]],
        B=[[1.0 / T1, 0.0], [0.0, 1.0 / T2]],
        C=[[1.0, 0.0], [0.0, 1.0]],
        D=[[0.0, 0.0], [0.0, 0.0]],
        u_0=[0.0, 0.0],
    )
    model.run(10 * max(T1, T2),
              {f'u_1_lti_{name}': 0.5, f'u_2_lti_{name}': 1.0})
    model.post()
    assert abs(model.get_value(f'z_1_lti_{name}') - 0.5) < 0.01
    assert abs(model.get_value(f'z_2_lti_{name}') - 1.0) < 0.01


def test_input_aliasing_via_h_dict_list():
    """`input=` as a list of m names binds each channel independently.
    Mix an h_dict expression (from a parallel lpf1) with a new global."""
    from pydae.core.builder.casadi_builder import MathBackend, CasadiBuilder
    from pydae.core.model.casadi_model import CasadiModel
    from pydae.control.continuos.lpf1 import lpf1

    dae = {'name': 'test_lti_alias',
           'f_list': [], 'x_list': [],
           'g_list': [], 'y_ini_list': [], 'y_run_list': [],
           'u_ini_dict': {}, 'u_run_dict': {},
           'params_dict': {}, 'h_dict': {}, 'xy_0_dict': {}}
    backend = MathBackend(use_casadi=True)

    lpf1(dae, 'meas', backend, {'T': 0.05, 'u': 0.4})  # h_dict['z_lpf1_meas']
    # Block-diagonal MIMO LPF; channel 1 driven by the lpf1 output,
    # channel 2 by a new global input `r`.
    lti(dae, '1', backend,
        {'A': [[-2.0, 0.0], [0.0, -2.0]],
         'B': [[ 2.0, 0.0], [0.0,  2.0]],
         'C': [[ 1.0, 0.0], [0.0,  1.0]],
         'D': [[ 0.0, 0.0], [0.0,  0.0]],
         'u_1': 0.4, 'u_2': 0.0},
        input=['z_lpf1_meas', 'r'])

    assert 'r' in dae['u_run_dict']
    model = CasadiModel(CasadiBuilder(dae).build())
    model.ini({}, dae['xy_0_dict'])
    # Both channels are unity DC gain; z_1 == u_1 == 0.4, z_2 == 0.
    assert abs(model.get_value('z_1_lti_1') - 0.4) < 1e-9
    assert abs(model.get_value('z_2_lti_1') - 0.0) < 1e-9


def test_output_aliasing_promotes_global_input():
    """`output=` aliasing an existing u_run input promotes that input
    to an algebraic state driven by this block's output."""
    from pydae.core.builder.casadi_builder import MathBackend, CasadiBuilder
    from pydae.core.model.casadi_model import CasadiModel

    dae = {'name': 'test_lti_out_alias',
           'f_list': [], 'x_list': [],
           'g_list': [], 'y_ini_list': [], 'y_run_list': [],
           'u_ini_dict': {}, 'u_run_dict': {},
           'params_dict': {}, 'h_dict': {}, 'xy_0_dict': {}}
    backend = MathBackend(use_casadi=True)

    # A standalone global input we'll later have the LTI block drive.
    setpoint = backend.symbols('setpoint')
    dae['u_ini_dict']['setpoint'] = 0.0
    dae['u_run_dict']['setpoint'] = 0.0

    lti(dae, '1', backend,
        {'A': [[-1.0]], 'B': [[1.0]], 'C': [[3.0]], 'D': [[0.0]], 'u_1': 0.7},
        output='setpoint')

    # The previous global input is gone -- promoted to algebraic.
    assert 'setpoint' not in dae['u_run_dict']

    model = CasadiModel(CasadiBuilder(dae).build())
    model.ini({}, dae['xy_0_dict'])
    # DC gain of A=-1, B=1, C=3, D=0 is C * (-A^{-1}) * B = 3. u_1 = 0.7
    # so setpoint should equal 0.7 * 3 = 2.1.
    assert abs(model.get_value('setpoint') - 2.1) < 1e-9


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    name, T = '1', 0.5
    model = _build_lti_model(
        name=name, A=[[-1.0 / T]], B=[[1.0 / T]], C=[[1.0]], D=[[0.0]], u_0=[0.0]
    )
    model.report_x()
    model.report_z()
    model.run(5 * T, {f'u_1_lti_{name}': 1.0})
    model.post()

    fig, ax = plt.subplots()
    ax.plot(model.Time, model.get_values(f'z_1_lti_{name}'), label='$z$ (LTI output)')
    ax.axhline(1.0, color='k', ls='--', lw=0.8, label='$u$ (step input)')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Output')
    ax.legend()
    fig.savefig('lti_step_response.svg')
    print("Test completed. Plot saved as 'lti_step_response.svg'.")
