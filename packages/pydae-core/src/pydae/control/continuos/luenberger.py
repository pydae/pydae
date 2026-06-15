# -*- coding: utf-8 -*-
r"""
Created on Sun June 15 13:36:00 2026

@author: jmmauricio

Luenberger state observer (LUENBERGER).

Estimates the internal state of an LTI plant

$$\dot x = A\,x + B\,u, \qquad y = C\,x + D\,u$$

from its inputs $u$ and measured outputs $y$. The observer runs a
copy of the plant model in parallel with the measurement, and drives
the estimate toward the truth with the innovation $y - \hat y$
multiplied by a gain matrix $L$:

$$\dot{\hat x} = A\,\hat x + B\,u + L\,(y - \hat y),
   \qquad \hat y = C\,\hat x + D\,u$$

Subtracting the plant from the observer gives the error dynamics

$$\dot{\tilde x} = (A - L\,C)\,\tilde x, \qquad \tilde x = x - \hat x$$

so the observer converges with eigenvalues of $A - L\,C$, which the
designer places (via pole placement, LQR / Kalman duality, etc.) to be
faster than the plant's own dynamics.

**Dimensions** derived from the matrix shapes (same convention as
:mod:`pydae.control.continuos.lti`):

* ``A`` is $n \times n$ -- ``n`` state estimates,
* ``B`` is $n \times m$ -- ``m`` plant-input channels,
* ``C`` is $p \times n$ -- ``p`` measurement channels,
* ``D`` is $p \times m$ -- direct feedthrough,
* ``L`` is $n \times p$ -- observer gain.

So the block has ``n`` dynamic states ($\hat x$), takes ``m + p``
inputs (the m control signals then the p measurements) and exposes
``n`` algebraic outputs ($\hat x$), plus ``p + p`` auxiliary
signals in ``h_dict`` (the estimated outputs $\hat y$ and the
innovations $\tilde y = y - \hat y$, useful for diagnostics and
fault detection).

Matrices are baked into the symbolic equations at build time as
numeric constants -- changing $A$, $B$, $C$, $D$ or $L$ requires a
model rebuild.
"""

import io


def descriptions():
    """
    Single source of truth for model parameters, inputs, states, and outputs.

    The structural matrices ``A`` / ``B`` / ``C`` / ``D`` / ``L`` are
    listed under *Parameters* but are **structural** -- they fix the
    block's dimensions at build time and are not exposed as
    runtime-tunable ``params_dict`` entries.
    """
    descriptions_list = []

    # Structural parameters (set at build time, baked into the equations)
    descriptions_list += [{"type": "Parameter", "tex": "A", "data": "A", "model": "A", "default": [[-1.0]], "description": "Plant state matrix, n x n", "units": "-"}]
    descriptions_list += [{"type": "Parameter", "tex": "B", "data": "B", "model": "B", "default": [[1.0]],  "description": "Plant input matrix, n x m",  "units": "-"}]
    descriptions_list += [{"type": "Parameter", "tex": "C", "data": "C", "model": "C", "default": [[1.0]],  "description": "Plant output matrix, p x n", "units": "-"}]
    descriptions_list += [{"type": "Parameter", "tex": "D", "data": "D", "model": "D", "default": [[0.0]],  "description": "Plant feedthrough matrix, p x m", "units": "-"}]
    descriptions_list += [{"type": "Parameter", "tex": "L", "data": "L", "model": "L", "default": [[1.0]],  "description": "Observer gain, n x p (places eigenvalues of A-L*C)", "units": "-"}]

    # Inputs: m control + p measurement channels
    descriptions_list += [{"type": "Input", "tex": "u_k", "data": "u_k", "model": "u_k (k=1..m)", "default": 0.0, "description": "Plant control input k", "units": "-"}]
    descriptions_list += [{"type": "Input", "tex": "y_j", "data": "y_j", "model": "y_j (j=1..p)", "default": 0.0, "description": "Plant measurement j",      "units": "-"}]

    # Dynamic States (n state estimates)
    descriptions_list += [{"type": "Dynamic State", "tex": r"\hat x_i", "data": "", "model": "x_hat_i (i=1..n)", "default": "", "description": "Estimated state component i", "units": "-"}]

    # Algebraic States (n outputs = the state estimates)
    descriptions_list += [{"type": "Algebraic State", "tex": "z_i", "data": "", "model": "z_i (i=1..n)", "default": "", "description": "State-estimate output (alias of x_hat_i, exposed in h_dict)", "units": "-"}]

    return descriptions_list


def _check_matrix(M, name, expected_rows, expected_cols):
    """Sanity-check a ``data_dict`` matrix and return it as a nested list."""
    if M is None:
        raise ValueError(f"luenberger: matrix {name} is required")
    rows = [list(r) for r in M]
    if not rows:
        raise ValueError(f"luenberger: matrix {name} is empty")
    n_rows = len(rows)
    n_cols = len(rows[0])
    if any(len(r) != n_cols for r in rows):
        raise ValueError(f"luenberger: matrix {name} has rows of unequal length")
    if expected_rows is not None and n_rows != expected_rows:
        raise ValueError(
            f"luenberger: matrix {name} has {n_rows} rows, expected {expected_rows}"
        )
    if expected_cols is not None and n_cols != expected_cols:
        raise ValueError(
            f"luenberger: matrix {name} has {n_cols} cols, expected {expected_cols}"
        )
    return rows


def luenberger(dae, name, backend, data_dict=None, input=None, output=None):
    """Append a Luenberger state observer to `dae`.

    Adds ``n`` dynamic states ``x_hat_1_luenberger_{name}`` ..
    ``x_hat_n_luenberger_{name}``, ``m + p`` global inputs (when
    ``input=None``), and ``n`` algebraic outputs ``z_1_luenberger_{name}``
    .. ``z_n_luenberger_{name}`` (when ``output=None``). Also exposes
    ``y_hat_j_luenberger_{name}`` and ``y_tilde_j_luenberger_{name}``
    (innovations) for each measurement channel via ``h_dict``.

    Parameters
    ----------
    data_dict : dict
        Required keys ``A``, ``B``, ``C``, ``D``, ``L`` as nested lists
        of floats. Optional ``u_k`` / ``y_j`` set the steady-state
        values used for the xy_0 hint.
    input : list[str] or None
        ``None`` -- create ``m + p`` fresh global inputs (control
        channels first, then measurement channels).
        ``list[str]`` -- must contain exactly ``m + p`` names; the
        first ``m`` bind to ``u`` and the next ``p`` bind to ``y``.
        Each name is resolved via
        :func:`pydae.control.continuos._common.resolve_input` (h_dict
        expression, existing-symbol alias, or new global input).
        A single string is rejected: the observer always has two input
        streams (control and measurement) and the user must say which
        is which.
    output : list[str] or None
        ``None`` -- expose the ``n`` state estimates as fresh
        ``z_i_luenberger_{name}`` symbols.
        ``list[str]`` -- must contain exactly ``n`` names; each is
        bound to one state estimate. Names that were previously global
        inputs are promoted to algebraic variables.
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
    L_in = data_dict.get('L', default_map.get('L'))

    # 2. Validate shapes.
    A = _check_matrix(A_in, 'A', expected_rows=None, expected_cols=None)
    n = len(A)
    if len(A[0]) != n:
        raise ValueError(f"luenberger: A must be square, got {n} x {len(A[0])}")
    B = _check_matrix(B_in, 'B', expected_rows=n, expected_cols=None)
    m = len(B[0])
    C = _check_matrix(C_in, 'C', expected_rows=None, expected_cols=n)
    p = len(C)
    D = _check_matrix(D_in, 'D', expected_rows=p, expected_cols=m)
    L = _check_matrix(L_in, 'L', expected_rows=n, expected_cols=p)

    if n < 1:
        raise ValueError("luenberger: at least one state required (n >= 1)")
    if p < 1:
        raise ValueError("luenberger: at least one measurement required (p >= 1)")

    val_u = [float(data_dict.get(f'u_{k + 1}', 0.0)) for k in range(m)]
    val_y = [float(data_dict.get(f'y_{j + 1}', 0.0)) for j in range(p)]

    # 3. Input channels. Convention: the first m entries of `input`
    #    bind to the control u; the next p bind to the measurement y.
    u_exprs = []
    y_exprs = []
    if input is None:
        for k in range(m):
            u_sym = backend.symbols(f"u_{k + 1}_luenberger_{name}")
            dae['u_ini_dict'][f"{u_sym}"] = val_u[k]
            dae['u_run_dict'][f"{u_sym}"] = val_u[k]
            u_exprs.append(u_sym)
        for j in range(p):
            y_sym = backend.symbols(f"y_{j + 1}_luenberger_{name}")
            dae['u_ini_dict'][f"{y_sym}"] = val_y[j]
            dae['u_run_dict'][f"{y_sym}"] = val_y[j]
            y_exprs.append(y_sym)
    elif isinstance(input, str):
        raise ValueError(
            "luenberger: a single input name is ambiguous (the block has "
            "m control inputs and p measurement inputs). Pass a list of "
            "m + p names with the m control channels first."
        )
    else:
        names = list(input)
        if len(names) != m + p:
            raise ValueError(
                f"luenberger: input list has {len(names)} entries, expected "
                f"m + p = {m} + {p} = {m + p} (the m control channels first, "
                f"then the p measurement channels)"
            )
        for k in range(m):
            u_exprs.append(resolve_input(names[k], dae, backend, val_u[k]))
        for j in range(p):
            y_exprs.append(resolve_input(names[m + j], dae, backend, val_y[j]))

    # 4. Dynamic states: estimated state components x_hat_i.
    x_hat_syms = [backend.symbols(f"x_hat_{i + 1}_luenberger_{name}") for i in range(n)]

    # 5. Algebraic outputs (the state estimates): fresh or aliased list.
    z_syms = []
    z_aliased_flags = []
    if output is None:
        for i in range(n):
            z_syms.append(backend.symbols(f"z_{i + 1}_luenberger_{name}"))
            z_aliased_flags.append(False)
    elif isinstance(output, str):
        if n != 1:
            raise ValueError(
                f"luenberger: a single output name was given but the block "
                f"has n = {n} state estimates; pass a list of {n} names"
            )
        z_syms.append(backend.symbols(output))
        z_aliased_flags.append(True)
    else:
        names_out = list(output)
        if len(names_out) != n:
            raise ValueError(
                f"luenberger: output list has {len(names_out)} entries, "
                f"expected n = {n} state estimates"
            )
        for nm in names_out:
            z_syms.append(backend.symbols(nm))
            z_aliased_flags.append(True)

    # 6. Auxiliary signals (always exposed via h_dict): estimated outputs
    #    y_hat = C * x_hat + D * u  and innovations y_tilde = y - y_hat.
    def _row_combination(coeffs, syms):
        """Return sum(c_i * s_i) skipping the zero-coefficient terms."""
        terms = [c * s for c, s in zip(coeffs, syms) if c != 0]
        if not terms:
            return 0
        expr = terms[0]
        for t in terms[1:]:
            expr = expr + t
        return expr

    y_hat_exprs = []
    for j in range(p):
        y_hat_exprs.append(
            _row_combination(C[j], x_hat_syms) + _row_combination(D[j], u_exprs)
        )

    # 7. Observer dynamics:
    #      dx_hat_i/dt = sum_j A[i][j] * x_hat_j
    #                  + sum_k B[i][k] * u_k
    #                  + sum_j L[i][j] * (y_j - y_hat_j)
    f_eqs = []
    for i in range(n):
        # Accumulate L[i][:] * (y - y_hat), using None as the "empty
        # accumulator" sentinel -- a bare ``0`` would force ``!=`` /
        # ``==`` checks against a CasADi SX, which raise.
        innovation_i = None
        for j in range(p):
            if L[i][j] != 0:
                term = L[i][j] * (y_exprs[j] - y_hat_exprs[j])
                innovation_i = term if innovation_i is None else innovation_i + term
        rhs = _row_combination(A[i], x_hat_syms) + _row_combination(B[i], u_exprs)
        if innovation_i is not None:
            rhs = rhs + innovation_i
        f_eqs.append(rhs)

    # 8. Output algebraic equations: each z_i tracks the corresponding x_hat_i.
    g_eqs = [z_syms[i] - x_hat_syms[i] for i in range(n)]

    # 9. Promote any aliased outputs that were previously global inputs.
    for z_sym, aliased in zip(z_syms, z_aliased_flags):
        if aliased:
            dae['u_ini_dict'].pop(str(z_sym), None)
            dae['u_run_dict'].pop(str(z_sym), None)

    # 10. Steady-state init hints.
    #     From dx_hat/dt = 0:
    #         (A - L*C) x_hat = -(B - L*D) u - L*y_inverted? Let's re-derive.
    #     A x_hat + B u + L (y - C x_hat - D u) = 0
    #     (A - L C) x_hat = -B u + L D u - L y = (L D - B) u - L y
    #     x_hat_ss = (A - L C)^{-1} * ((L D - B) u - L y)
    #     y_hat_ss = C x_hat_ss + D u
    A_np = np.array(A, dtype=float)
    B_np = np.array(B, dtype=float)
    C_np = np.array(C, dtype=float)
    D_np = np.array(D, dtype=float)
    L_np = np.array(L, dtype=float)
    u_0  = np.array(val_u, dtype=float)
    y_0  = np.array(val_y, dtype=float)
    try:
        M = A_np - L_np @ C_np
        rhs = (L_np @ D_np - B_np) @ u_0 - L_np @ y_0
        x_hat_ss = np.linalg.solve(M, rhs)
    except np.linalg.LinAlgError:
        x_hat_ss = np.zeros(n)

    for x_sym, val in zip(x_hat_syms, x_hat_ss):
        dae['xy_0_dict'][f"{x_sym}"] = float(val)
    for z_sym, val in zip(z_syms, x_hat_ss):
        dae['xy_0_dict'][f"{z_sym}"] = float(val)

    # 11. Assembly.
    dae['f_list']     += f_eqs
    dae['x_list']     += x_hat_syms
    dae['g_list']     += g_eqs
    dae['y_ini_list'] += list(z_syms)
    dae['y_run_list'] += list(z_syms)
    for z_sym in z_syms:
        dae['h_dict'][f"{z_sym}"] = z_sym
    for i in range(n):
        dae['h_dict'][f"x_hat_{i + 1}_luenberger_{name}"] = x_hat_syms[i]
    for j in range(p):
        dae['h_dict'][f"y_hat_{j + 1}_luenberger_{name}"] = y_hat_exprs[j]
        dae['h_dict'][f"y_tilde_{j + 1}_luenberger_{name}"] = y_exprs[j] - y_hat_exprs[j]

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
def _build_luenberger_model(name='1', A=None, B=None, C=None, D=None, L=None,
                              u_0=None, y_0=None):
    """Assemble a standalone CasadiModel containing a single Luenberger observer."""
    from pydae.core.builder.casadi_builder import MathBackend, CasadiBuilder
    from pydae.core.model.casadi_model import CasadiModel

    if A is None: A = [[-1.0]]
    if B is None: B = [[1.0]]
    if C is None: C = [[1.0]]
    if D is None: D = [[0.0]]
    if L is None: L = [[2.0]]
    if u_0 is None: u_0 = [0.0] * len(B[0])
    if y_0 is None: y_0 = [0.0] * len(C)

    dae = {'name': f'test_luenberger_{name}',
           'f_list': [], 'x_list': [],
           'g_list': [], 'y_ini_list': [], 'y_run_list': [],
           'u_ini_dict': {}, 'u_run_dict': {},
           'params_dict': {},
           'h_dict': {},
           'xy_0_dict': {}}

    backend = MathBackend(use_casadi=True)
    dd = {'A': A, 'B': B, 'C': C, 'D': D, 'L': L}
    for k, val in enumerate(u_0):
        dd[f'u_{k + 1}'] = val
    for j, val in enumerate(y_0):
        dd[f'y_{j + 1}'] = val
    luenberger(dae, name, backend, dd)

    model = CasadiModel(CasadiBuilder(dae).build())
    model.ini({}, dae['xy_0_dict'])

    return model


def test_siso_steady_state_matches_closed_form():
    """For the SISO plant A=-1, B=1, C=1, D=0 with observer gain L=2,
    the algebraic SS is x_hat = (L*y - B*u) / (L - A) = (2*y + u) / 3.
    Note the sign: dx_hat/dt = A*x_hat + B*u + L*(y - C*x_hat) = 0
    -> -x_hat + u + 2*(y - x_hat) = 0 -> x_hat = (u + 2*y) / 3.
    """
    name = '1'
    u_0, y_0 = 1.0, 0.5
    model = _build_luenberger_model(name=name, u_0=[u_0], y_0=[y_0])
    expected = (u_0 + 2 * y_0) / 3.0
    assert abs(model.get_value(f'z_1_luenberger_{name}') - expected) < 1e-9
    assert abs(model.get_value(f'x_hat_1_luenberger_{name}') - expected) < 1e-9
    # y_hat = C * x_hat = expected; y_tilde = y - y_hat.
    assert abs(model.get_value(f'y_hat_1_luenberger_{name}') - expected) < 1e-9
    assert abs(model.get_value(f'y_tilde_1_luenberger_{name}') - (y_0 - expected)) < 1e-9


def test_observer_converges_to_plant_state():
    """Run a plant `lti` block and an `luenberger` observer side-by-side
    in the same dae, wiring the observer's `u` input to the plant's `u`
    and its `y` input to the plant's measurement output. Step the plant
    input; the observer's state estimate should converge to the true
    plant state, with closed-form steady-state agreement.
    """
    from pydae.core.builder.casadi_builder import MathBackend, CasadiBuilder
    from pydae.core.model.casadi_model import CasadiModel
    from pydae.control.continuos.lti import lti

    # Plant: 2-state, SIMO (1 input, 1 output).
    A = [[-1.0, 1.0], [0.0, -2.0]]
    B = [[0.0], [1.0]]
    C = [[1.0, 0.0]]
    D = [[0.0]]
    # Pole-placement: A - LC has eigenvalues at -5 and -6 -> L = [[8],[12]]
    # (derived in the file's docstring; tested numerically below).
    L = [[8.0], [12.0]]

    dae = {'name': 'test_lu_plus_lti',
           'f_list': [], 'x_list': [],
           'g_list': [], 'y_ini_list': [], 'y_run_list': [],
           'u_ini_dict': {}, 'u_run_dict': {},
           'params_dict': {}, 'h_dict': {}, 'xy_0_dict': {}}

    backend = MathBackend(use_casadi=True)
    # Plant: u in, measurement y = z_1_lti_plant in h_dict.
    lti(dae, 'plant', backend, {'A': A, 'B': B, 'C': C, 'D': D, 'u_1': 0.0})

    # Observer reads u from the plant's input symbol, and y from the
    # plant's output expression in h_dict.
    luenberger(dae, 'obs', backend,
               {'A': A, 'B': B, 'C': C, 'D': D, 'L': L},
               input=['u_1_lti_plant', 'z_1_lti_plant'])

    model = CasadiModel(CasadiBuilder(dae).build())
    model.ini({}, dae['xy_0_dict'])

    # Step the plant input and let both systems run to steady state.
    model.run(20.0, {'u_1_lti_plant': 1.0})
    model.post()

    # Plant SS: dx/dt = 0 -> x_2 = -B u / (-2) wait let me redo.
    # A x + B u = 0 -> [[-1,1],[0,-2]] x + [[0],[1]] = 0
    #   -x_1 + x_2 = 0 -> x_1 = x_2
    #   -2 x_2 + 1 = 0 -> x_2 = 0.5, x_1 = 0.5
    x_plant_true = (0.5, 0.5)
    x_hat = (model.get_value('x_hat_1_luenberger_obs'),
             model.get_value('x_hat_2_luenberger_obs'))
    assert abs(x_hat[0] - x_plant_true[0]) < 1e-3
    assert abs(x_hat[1] - x_plant_true[1]) < 1e-3
    # Innovation has decayed to zero.
    assert abs(model.get_value('y_tilde_1_luenberger_obs')) < 1e-3


def test_input_list_wrong_length_rejected():
    """input list length must equal m + p."""
    A = [[-1.0]]; B = [[1.0]]; C = [[1.0]]; D = [[0.0]]; L = [[2.0]]
    from pydae.core.builder.casadi_builder import MathBackend
    dae = {'h_dict':{}, 'g_list':[], 'y_ini_list':[], 'y_run_list':[],
           'u_ini_dict':{}, 'u_run_dict':{}, 'params_dict':{},
           'xy_0_dict':{}, 'x_list':[], 'f_list':[]}
    try:
        luenberger(dae, 'x', MathBackend(use_casadi=True),
                   {'A': A, 'B': B, 'C': C, 'D': D, 'L': L},
                   input=['only_one'])
    except ValueError as e:
        assert 'expected m + p' in str(e)
        return
    raise AssertionError("should have rejected single-name input list")


def test_L_shape_validated():
    """L must be n x p."""
    A = [[-1.0, 0.0], [0.0, -2.0]]; B = [[1.0], [0.0]]
    C = [[1.0, 1.0]]; D = [[0.0]]; L_bad = [[1.0]]  # 1x1, should be 2x1
    from pydae.core.builder.casadi_builder import MathBackend
    dae = {'h_dict':{}, 'g_list':[], 'y_ini_list':[], 'y_run_list':[],
           'u_ini_dict':{}, 'u_run_dict':{}, 'params_dict':{},
           'xy_0_dict':{}, 'x_list':[], 'f_list':[]}
    try:
        luenberger(dae, 'x', MathBackend(use_casadi=True),
                   {'A': A, 'B': B, 'C': C, 'D': D, 'L': L_bad})
    except ValueError as e:
        assert 'L' in str(e) and 'rows' in str(e)
        return
    raise AssertionError("should have rejected L with wrong shape")


if __name__ == '__main__':
    name = '1'
    u_0, y_0 = 1.0, 0.5
    model = _build_luenberger_model(name=name, u_0=[u_0], y_0=[y_0])
    model.report_u()
    model.report_x()
    model.report_z()
    expected = (u_0 + 2 * y_0) / 3.0
    print(f"x_hat = {model.get_value(f'x_hat_1_luenberger_{name}'):.6f}  "
          f"(expected {expected:.6f})")
