# -*- coding: utf-8 -*-
r"""
Created on Sun June 15 13:40:00 2026

@author: jmmauricio

Steady-state continuous-time Kalman filter (KALMAN).

The optimal linear observer for an LTI plant

$$\dot x = A\,x + B\,u + w, \qquad y = C\,x + D\,u + v$$

driven by zero-mean white process noise $w \sim \mathcal{N}(0, Q)$
and measurement noise $v \sim \mathcal{N}(0, R)$. The filter has the
exact same structure as the Luenberger observer
(:mod:`pydae.control.continuos.luenberger`),

$$\dot{\hat x} = A\,\hat x + B\,u + K\,(y - \hat y),
   \qquad \hat y = C\,\hat x + D\,u$$

but with the gain $K$ computed at build time as the minimum-variance
solution of the continuous algebraic Riccati equation (CARE):

$$A P + P A^T - P C^T R^{-1} C P + Q = 0,
   \qquad K = P\, C^T R^{-1}$$

where $P$ is the steady-state estimation-error covariance. The
Riccati solve is done with :func:`scipy.linalg.solve_continuous_are`
(via its dual to the LQR problem) and the resulting numeric gain
$K$ is baked into the symbolic equations -- no extra
$\tfrac{n(n+1)}{2}$ covariance states are introduced at runtime, so
the block is as cheap as the Luenberger observer.

**Dimensions** derived from the matrix shapes (same convention as
:mod:`pydae.control.continuos.lti` and
:mod:`pydae.control.continuos.luenberger`):

* ``A`` is $n \times n$, ``B`` is $n \times m$, ``C`` is $p \times n$,
  ``D`` is $p \times m$,
* ``Q`` is $n \times n$ (symmetric positive-semi-definite),
* ``R`` is $p \times p$ (symmetric positive-definite).

The block then has ``n`` dynamic states ($\hat x$), takes ``m + p``
inputs (m control then p measurement) and exposes ``n`` algebraic
outputs (the state estimates) plus the ``y_hat`` and ``y_tilde``
(innovation) signals in ``h_dict``.

Matrices are baked into the symbolic equations at build time as
numeric constants. Changing $A$, $B$, $C$, $D$, $Q$ or $R$ requires
a model rebuild (which re-solves the Riccati equation).
"""

import io


def descriptions():
    """
    Single source of truth for model parameters, inputs, states, and outputs.

    The structural matrices are **set at build time** and not exposed
    as runtime-tunable ``params_dict`` entries. ``Q`` and ``R`` set
    the Kalman gain via an offline Riccati solve; tuning them is a
    rebuild, not a runtime parameter change.
    """
    descriptions_list = []

    # Structural parameters (baked into the equations at build time)
    descriptions_list += [{"type": "Parameter", "tex": "A", "data": "A", "model": "A", "default": [[-1.0]], "description": "Plant state matrix, n x n", "units": "-"}]
    descriptions_list += [{"type": "Parameter", "tex": "B", "data": "B", "model": "B", "default": [[1.0]],  "description": "Plant input matrix, n x m",  "units": "-"}]
    descriptions_list += [{"type": "Parameter", "tex": "C", "data": "C", "model": "C", "default": [[1.0]],  "description": "Plant output matrix, p x n", "units": "-"}]
    descriptions_list += [{"type": "Parameter", "tex": "D", "data": "D", "model": "D", "default": [[0.0]],  "description": "Plant feedthrough matrix, p x m", "units": "-"}]
    descriptions_list += [{"type": "Parameter", "tex": "Q", "data": "Q", "model": "Q", "default": [[1.0]],  "description": "Process-noise covariance, n x n (symmetric PSD)",       "units": "-"}]
    descriptions_list += [{"type": "Parameter", "tex": "R", "data": "R", "model": "R", "default": [[1.0]],  "description": "Measurement-noise covariance, p x p (symmetric PD)",   "units": "-"}]

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
        raise ValueError(f"kalman: matrix {name} is required")
    rows = [list(r) for r in M]
    if not rows:
        raise ValueError(f"kalman: matrix {name} is empty")
    n_rows = len(rows)
    n_cols = len(rows[0])
    if any(len(r) != n_cols for r in rows):
        raise ValueError(f"kalman: matrix {name} has rows of unequal length")
    if expected_rows is not None and n_rows != expected_rows:
        raise ValueError(
            f"kalman: matrix {name} has {n_rows} rows, expected {expected_rows}"
        )
    if expected_cols is not None and n_cols != expected_cols:
        raise ValueError(
            f"kalman: matrix {name} has {n_cols} cols, expected {expected_cols}"
        )
    return rows


def _solve_kalman_gain(A_np, C_np, Q_np, R_np):
    """Compute the steady-state Kalman gain K = P C^T R^{-1}.

    P is the unique symmetric positive-semi-definite solution of the
    continuous algebraic Riccati equation
        A P + P A^T - P C^T R^{-1} C P + Q = 0
    obtained as the dual of the LQR problem via
    :func:`scipy.linalg.solve_continuous_are`.
    """
    import numpy as np
    from scipy.linalg import solve_continuous_are

    # LQR-dual: solve_continuous_are(A.T, C.T, Q, R) returns P such that
    # A P + P A^T + Q - P C^T R^{-1} C P = 0 (real-symmetric A means
    # the .T on the first argument plays both roles correctly).
    P = solve_continuous_are(A_np.T, C_np.T, Q_np, R_np)
    K = P @ C_np.T @ np.linalg.inv(R_np)
    return K, P


def kalman(dae, name, backend, data_dict=None, input=None, output=None):
    """Append a steady-state continuous-time Kalman filter to `dae`.

    Same runtime structure as
    :func:`pydae.control.continuos.luenberger.luenberger`; the
    difference is the gain matrix, which here is the steady-state
    Kalman gain computed at build time from ``Q`` and ``R`` rather
    than user-specified.

    Adds ``n`` dynamic states ``x_hat_1_kalman_{name}`` ..
    ``x_hat_n_kalman_{name}``, ``m + p`` global inputs (when
    ``input=None``), and ``n`` algebraic outputs
    ``z_1_kalman_{name}`` .. ``z_n_kalman_{name}`` (when
    ``output=None``). Also exposes ``y_hat_j_kalman_{name}`` and
    ``y_tilde_j_kalman_{name}`` (innovations) via ``h_dict``, plus
    a one-shot informational h_dict entry per gain element
    ``K_ij_kalman_{name}`` -- these are constants, not signals, but
    they let the user read the chosen gain back from the built model
    without re-solving the Riccati equation.

    Parameters
    ----------
    data_dict : dict
        Required keys ``A``, ``B``, ``C``, ``D``, ``Q``, ``R`` as
        nested lists of floats. Optional ``u_k`` / ``y_j`` set the
        steady-state values used for the xy_0 hint.
    input : list[str] or None
        Same semantics as ``luenberger``: ``m + p`` names, control
        channels first.
    output : list[str] or None
        Same semantics as ``luenberger``: ``n`` names binding the
        state estimates.
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
    Q_in = data_dict.get('Q', default_map.get('Q'))
    R_in = data_dict.get('R', default_map.get('R'))

    # 2. Validate shapes.
    A = _check_matrix(A_in, 'A', expected_rows=None, expected_cols=None)
    n = len(A)
    if len(A[0]) != n:
        raise ValueError(f"kalman: A must be square, got {n} x {len(A[0])}")
    B = _check_matrix(B_in, 'B', expected_rows=n, expected_cols=None)
    m = len(B[0])
    C = _check_matrix(C_in, 'C', expected_rows=None, expected_cols=n)
    p = len(C)
    D = _check_matrix(D_in, 'D', expected_rows=p, expected_cols=m)
    Q = _check_matrix(Q_in, 'Q', expected_rows=n, expected_cols=n)
    R = _check_matrix(R_in, 'R', expected_rows=p, expected_cols=p)

    if n < 1:
        raise ValueError("kalman: at least one state required (n >= 1)")
    if p < 1:
        raise ValueError("kalman: at least one measurement required (p >= 1)")

    val_u = [float(data_dict.get(f'u_{k + 1}', 0.0)) for k in range(m)]
    val_y = [float(data_dict.get(f'y_{j + 1}', 0.0)) for j in range(p)]

    # 3. Solve the CARE for the steady-state Kalman gain K (n x p).
    A_np = np.array(A, dtype=float)
    B_np = np.array(B, dtype=float)
    C_np = np.array(C, dtype=float)
    D_np = np.array(D, dtype=float)
    Q_np = np.array(Q, dtype=float)
    R_np = np.array(R, dtype=float)
    try:
        K_np, P_np = _solve_kalman_gain(A_np, C_np, Q_np, R_np)
    except Exception as e:
        raise ValueError(
            f"kalman: failed to solve the continuous algebraic Riccati "
            f"equation -- check that (A, C) is detectable and that Q is "
            f"PSD, R is PD. Underlying error: {e}"
        )
    L = K_np.tolist()  # bake K as the observer gain; downstream code mirrors luenberger.

    # 4. Input channels: first m control, then p measurement.
    u_exprs = []
    y_exprs = []
    if input is None:
        for k in range(m):
            u_sym = backend.symbols(f"u_{k + 1}_kalman_{name}")
            dae['u_ini_dict'][f"{u_sym}"] = val_u[k]
            dae['u_run_dict'][f"{u_sym}"] = val_u[k]
            u_exprs.append(u_sym)
        for j in range(p):
            y_sym = backend.symbols(f"y_{j + 1}_kalman_{name}")
            dae['u_ini_dict'][f"{y_sym}"] = val_y[j]
            dae['u_run_dict'][f"{y_sym}"] = val_y[j]
            y_exprs.append(y_sym)
    elif isinstance(input, str):
        raise ValueError(
            "kalman: a single input name is ambiguous (the block has m "
            "control inputs and p measurement inputs). Pass a list of "
            "m + p names with the m control channels first."
        )
    else:
        names = list(input)
        if len(names) != m + p:
            raise ValueError(
                f"kalman: input list has {len(names)} entries, expected "
                f"m + p = {m} + {p} = {m + p} (the m control channels first, "
                f"then the p measurement channels)"
            )
        for k in range(m):
            u_exprs.append(resolve_input(names[k], dae, backend, val_u[k]))
        for j in range(p):
            y_exprs.append(resolve_input(names[m + j], dae, backend, val_y[j]))

    # 5. Dynamic states: estimated state components x_hat_i.
    x_hat_syms = [backend.symbols(f"x_hat_{i + 1}_kalman_{name}") for i in range(n)]

    # 6. Algebraic outputs (the state estimates): fresh or aliased list.
    z_syms = []
    z_aliased_flags = []
    if output is None:
        for i in range(n):
            z_syms.append(backend.symbols(f"z_{i + 1}_kalman_{name}"))
            z_aliased_flags.append(False)
    elif isinstance(output, str):
        if n != 1:
            raise ValueError(
                f"kalman: a single output name was given but the block "
                f"has n = {n} state estimates; pass a list of {n} names"
            )
        z_syms.append(backend.symbols(output))
        z_aliased_flags.append(True)
    else:
        names_out = list(output)
        if len(names_out) != n:
            raise ValueError(
                f"kalman: output list has {len(names_out)} entries, "
                f"expected n = {n} state estimates"
            )
        for nm in names_out:
            z_syms.append(backend.symbols(nm))
            z_aliased_flags.append(True)

    # 7. Auxiliary expressions: estimated outputs y_hat = C x_hat + D u
    #    and innovations y_tilde = y - y_hat (always exposed via h_dict).
    def _row_combination(coeffs, syms):
        """Return sum(c_i * s_i), skipping the zero-coefficient terms."""
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

    # 8. Observer dynamics with the Kalman gain L = K:
    #      dx_hat_i/dt = sum_j A[i][j] * x_hat_j
    #                  + sum_k B[i][k] * u_k
    #                  + sum_j L[i][j] * (y_j - y_hat_j)
    f_eqs = []
    for i in range(n):
        innovation_i = None  # None sentinel: avoid SX-vs-0 truthiness
        for j in range(p):
            if L[i][j] != 0:
                term = L[i][j] * (y_exprs[j] - y_hat_exprs[j])
                innovation_i = term if innovation_i is None else innovation_i + term
        rhs = _row_combination(A[i], x_hat_syms) + _row_combination(B[i], u_exprs)
        if innovation_i is not None:
            rhs = rhs + innovation_i
        f_eqs.append(rhs)

    # 9. Output algebraic equations: each z_i tracks the corresponding x_hat_i.
    g_eqs = [z_syms[i] - x_hat_syms[i] for i in range(n)]

    # 10. Promote any aliased outputs that were previously global inputs.
    for z_sym, aliased in zip(z_syms, z_aliased_flags):
        if aliased:
            dae['u_ini_dict'].pop(str(z_sym), None)
            dae['u_run_dict'].pop(str(z_sym), None)

    # 11. Steady-state init hints (same closed form as the Luenberger block).
    #     (A - K C) x_hat + (B - K D) u + K y = 0
    #     => x_hat_ss = (A - K C)^{-1} * ((K D - B) u - K y)
    u_0 = np.array(val_u, dtype=float)
    y_0 = np.array(val_y, dtype=float)
    try:
        M = A_np - K_np @ C_np
        rhs_ss = (K_np @ D_np - B_np) @ u_0 - K_np @ y_0
        x_hat_ss = np.linalg.solve(M, rhs_ss)
    except np.linalg.LinAlgError:
        x_hat_ss = np.zeros(n)

    for x_sym, val in zip(x_hat_syms, x_hat_ss):
        dae['xy_0_dict'][f"{x_sym}"] = float(val)
    for z_sym, val in zip(z_syms, x_hat_ss):
        dae['xy_0_dict'][f"{z_sym}"] = float(val)

    # 12. Assembly. Equations baked with the computed K; gain entries
    #     are also exposed in h_dict as numeric "constants" so the user
    #     can read back what the Riccati solve produced.
    dae['f_list']     += f_eqs
    dae['x_list']     += x_hat_syms
    dae['g_list']     += g_eqs
    dae['y_ini_list'] += list(z_syms)
    dae['y_run_list'] += list(z_syms)
    for z_sym in z_syms:
        dae['h_dict'][f"{z_sym}"] = z_sym
    for i in range(n):
        dae['h_dict'][f"x_hat_{i + 1}_kalman_{name}"] = x_hat_syms[i]
    for j in range(p):
        dae['h_dict'][f"y_hat_{j + 1}_kalman_{name}"] = y_hat_exprs[j]
        dae['h_dict'][f"y_tilde_{j + 1}_kalman_{name}"] = y_exprs[j] - y_hat_exprs[j]
    for i in range(n):
        for j in range(p):
            dae['h_dict'][f"K_{i + 1}{j + 1}_kalman_{name}"] = float(K_np[i, j])

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
def _build_kalman_model(name='1', A=None, B=None, C=None, D=None, Q=None, R=None,
                         u_0=None, y_0=None):
    """Assemble a standalone CasadiModel containing a single Kalman filter."""
    from pydae.core.builder.casadi_builder import MathBackend, CasadiBuilder
    from pydae.core.model.casadi_model import CasadiModel

    if A is None: A = [[-1.0]]
    if B is None: B = [[1.0]]
    if C is None: C = [[1.0]]
    if D is None: D = [[0.0]]
    if Q is None: Q = [[1.0]]
    if R is None: R = [[1.0]]
    if u_0 is None: u_0 = [0.0] * len(B[0])
    if y_0 is None: y_0 = [0.0] * len(C)

    dae = {'name': f'test_kalman_{name}',
           'f_list': [], 'x_list': [],
           'g_list': [], 'y_ini_list': [], 'y_run_list': [],
           'u_ini_dict': {}, 'u_run_dict': {},
           'params_dict': {},
           'h_dict': {},
           'xy_0_dict': {}}

    backend = MathBackend(use_casadi=True)
    dd = {'A': A, 'B': B, 'C': C, 'D': D, 'Q': Q, 'R': R}
    for k, val in enumerate(u_0):
        dd[f'u_{k + 1}'] = val
    for j, val in enumerate(y_0):
        dd[f'y_{j + 1}'] = val
    kalman(dae, name, backend, dd)

    model = CasadiModel(CasadiBuilder(dae).build())
    model.ini({}, dae['xy_0_dict'])

    return model


def test_siso_kalman_gain_matches_closed_form():
    """SISO A=-a, B=1, C=1, D=0, Q=q, R=r.
    Riccati: -2 a P - P^2 / r + q = 0  -> P = -a r + sqrt(a^2 r^2 + q r)
    K = P / r. For a=1, q=1, r=1: P = -1 + sqrt(2) ~ 0.4142, K ~ 0.4142.
    """
    import numpy as np
    name = '1'
    a, q, r = 1.0, 1.0, 1.0
    A = [[-a]]; B = [[1.0]]; C = [[1.0]]; D = [[0.0]]
    Q = [[q]];  R = [[r]]
    model = _build_kalman_model(name=name, A=A, B=B, C=C, D=D, Q=Q, R=R,
                                  u_0=[0.5], y_0=[0.5])
    expected_P = -a * r + np.sqrt(a * a * r * r + q * r)
    expected_K = expected_P / r
    K_built = model.get_value(f'K_11_kalman_{name}')
    assert abs(K_built - expected_K) < 1e-9, (K_built, expected_K)

    # Steady-state x_hat for u=0.5, y=0.5: dx_hat/dt = 0 ->
    # -a x_hat + u + K (y - x_hat) = 0  -> x_hat = (u + K y) / (a + K)
    expected_x = (0.5 + expected_K * 0.5) / (a + expected_K)
    assert abs(model.get_value(f'x_hat_1_kalman_{name}') - expected_x) < 1e-9


def test_observer_converges_to_plant_state():
    """Run an `lti` plant in parallel with a `kalman` filter and confirm
    the Kalman state estimate converges to the true plant state after
    a step input."""
    from pydae.core.builder.casadi_builder import MathBackend, CasadiBuilder
    from pydae.core.model.casadi_model import CasadiModel
    from pydae.control.continuos.lti import lti

    # Same plant as the luenberger end-to-end test: 2-state SIMO.
    A = [[-1.0, 1.0], [0.0, -2.0]]
    B = [[0.0], [1.0]]
    C = [[1.0, 0.0]]
    D = [[0.0]]
    # Process noise heavier than measurement noise -> faster, more
    # aggressive Kalman gain.
    Q = [[1.0, 0.0], [0.0, 1.0]]
    R = [[0.01]]

    dae = {'name': 'test_kf_plus_lti',
           'f_list': [], 'x_list': [],
           'g_list': [], 'y_ini_list': [], 'y_run_list': [],
           'u_ini_dict': {}, 'u_run_dict': {},
           'params_dict': {}, 'h_dict': {}, 'xy_0_dict': {}}

    backend = MathBackend(use_casadi=True)
    lti(dae, 'plant', backend, {'A': A, 'B': B, 'C': C, 'D': D, 'u_1': 0.0})
    kalman(dae, 'obs', backend,
           {'A': A, 'B': B, 'C': C, 'D': D, 'Q': Q, 'R': R},
           input=['u_1_lti_plant', 'z_1_lti_plant'])

    model = CasadiModel(CasadiBuilder(dae).build())
    model.ini({}, dae['xy_0_dict'])
    model.run(20.0, {'u_1_lti_plant': 1.0})
    model.post()

    # Plant SS: x_1 = x_2 = 0.5 (derived in the luenberger test).
    assert abs(model.get_value('x_hat_1_kalman_obs') - 0.5) < 1e-3
    assert abs(model.get_value('x_hat_2_kalman_obs') - 0.5) < 1e-3
    assert abs(model.get_value('y_tilde_1_kalman_obs')) < 1e-3


def test_gain_increases_with_lower_R():
    """Decreasing the measurement-noise covariance R should trust the
    measurement more and produce a *larger* Kalman gain. Verify the
    monotonic relationship numerically on the SISO plant from the
    first test."""
    import numpy as np
    A = [[-1.0]]; B = [[1.0]]; C = [[1.0]]; D = [[0.0]]; Q = [[1.0]]
    K_vals = []
    for r in (0.01, 0.1, 1.0, 10.0):
        model = _build_kalman_model(name='r' + str(r).replace('.', '_'),
                                      A=A, B=B, C=C, D=D, Q=Q, R=[[r]])
        K_vals.append(model.get_value(f'K_11_kalman_r{str(r).replace(".", "_")}'))
    # K should be monotonically decreasing as R increases.
    for prev, curr in zip(K_vals, K_vals[1:]):
        assert prev > curr, f"K should decrease with R: {K_vals}"


def test_Q_R_shape_validated():
    """Q must be n x n; R must be p x p."""
    from pydae.core.builder.casadi_builder import MathBackend
    A = [[-1.0, 0.0], [0.0, -2.0]]; B = [[1.0], [0.0]]
    C = [[1.0, 1.0]]; D = [[0.0]]
    for label, kwargs in [
        ('Q wrong shape', {'A': A, 'B': B, 'C': C, 'D': D,
                            'Q': [[1.0]], 'R': [[1.0]]}),
        ('R wrong shape', {'A': A, 'B': B, 'C': C, 'D': D,
                            'Q': [[1.0, 0.0], [0.0, 1.0]],
                            'R': [[1.0, 0.0], [0.0, 1.0]]}),
    ]:
        dae = {'h_dict': {}, 'g_list': [], 'y_ini_list': [], 'y_run_list': [],
               'u_ini_dict': {}, 'u_run_dict': {}, 'params_dict': {},
               'xy_0_dict': {}, 'x_list': [], 'f_list': []}
        try:
            kalman(dae, 'x', MathBackend(use_casadi=True), kwargs)
        except ValueError as e:
            assert ('Q' in str(e)) or ('R' in str(e)), (label, str(e))
            continue
        raise AssertionError(f"should have rejected: {label}")


if __name__ == '__main__':
    name = '1'
    model = _build_kalman_model(name=name, u_0=[0.5], y_0=[0.5])
    model.report_u()
    model.report_x()
    model.report_z()
    print(f"K = {model.get_value(f'K_11_kalman_{name}'):.6f}")
    print(f"x_hat = {model.get_value(f'x_hat_1_kalman_{name}'):.6f}")
