# -*- coding: utf-8 -*-
r"""
Created on Sun June 15 13:44:00 2026

@author: jmmauricio

Matrix gain block (MATRIX_GAIN).

A purely algebraic multi-input / multi-output static map

$$z = K\,u,
   \qquad z_i = \sum_{j=1}^{m} K_{ij}\,u_j$$

where ``K`` is a ``p x m`` matrix passed in ``data_dict`` (nested
list of floats). Dimensions are derived from ``K`` -- ``m`` is the
number of input channels (columns), ``p`` the number of output
channels (rows). The block has no dynamic state.

**Relation to other blocks.** This is the multi-channel
generalisation of :mod:`pydae.control.continuos.summation`
(``z = K_1 u_1 + K_2 u_2``, ``p = 1``, ``m = 2``); it is also the
pure-static special case of :mod:`pydae.control.continuos.lti`
(``A = 0``, ``B = 0``, ``C = 0``, ``D = K``). Use ``matrix_gain``
when you need a constant linear mapping between two signal vectors
-- for example, the state-feedback law ``u = -K x`` of an LQR
controller, or the measurement-mixing of an estimator's output
into a small set of derived signals.

Entries of ``K`` are baked into the symbolic equations at build
time. Changing the gain matrix requires a model rebuild; for a
single runtime-tunable gain see the scalar gain in the ``pi`` /
``lpf1`` parameter sets.
"""

import io


def descriptions():
    """
    Single source of truth for model parameters, inputs, states, and outputs.

    The matrix ``K`` is **structural** -- it fixes the block's
    dimensions at build time and is baked into the equations, not
    exposed as a runtime-tunable ``params_dict`` entry.
    """
    descriptions_list = []

    # Structural parameter (set at build time)
    descriptions_list += [{"type": "Parameter", "tex": "K", "data": "K", "model": "K", "default": [[1.0]], "description": "Gain matrix, p x m (nested list of floats)", "units": "-"}]

    # Inputs (m channels, u_1 named here for documentation)
    descriptions_list += [{"type": "Input", "tex": "u_k", "data": "u_k", "model": "u_k (k=1..m)", "default": 0.0, "description": "Input channel k", "units": "-"}]

    # Algebraic States (p outputs, z_1 named here for documentation)
    descriptions_list += [{"type": "Algebraic State", "tex": "z_j", "data": "", "model": "z_j (j=1..p)", "default": "", "description": "Output channel j (z = K*u, exposed in h_dict)", "units": "-"}]

    return descriptions_list


def _check_matrix(M, name):
    """Sanity-check a ``data_dict`` matrix and return it as a nested list
    plus its (rows, cols) shape. Used by ``matrix_gain`` only."""
    if M is None:
        raise ValueError(f"matrix_gain: matrix {name} is required")
    rows = [list(r) for r in M]
    if not rows:
        raise ValueError(f"matrix_gain: matrix {name} is empty")
    n_rows = len(rows)
    n_cols = len(rows[0])
    if any(len(r) != n_cols for r in rows):
        raise ValueError(f"matrix_gain: matrix {name} has rows of unequal length")
    if n_cols == 0:
        raise ValueError(f"matrix_gain: matrix {name} has zero-length rows")
    return rows, n_rows, n_cols


def matrix_gain(dae, name, backend, data_dict=None, input=None, output=None):
    """Append a static matrix gain block ``z = K*u`` to `dae`.

    Adds ``m`` global inputs ``u_1_matrix_gain_{name}`` ..
    ``u_m_matrix_gain_{name}`` (when ``input=None``), per-input
    bindings from ``K`` (no separate ``K_ij`` symbols -- the
    coefficients are baked into the algebraic equation), and ``p``
    algebraic outputs ``z_1_matrix_gain_{name}`` ..
    ``z_p_matrix_gain_{name}`` (when ``output=None``).

    Parameters
    ----------
    data_dict : dict
        Required key ``K`` as a nested list of floats (rows of the
        gain matrix, length ``m`` each, ``p`` rows). Optional
        ``u_k`` (``k = 1..m``) sets the steady-state value of input
        ``k`` used for the xy_0 hint.
    input : str, list[str], or None
        ``None`` -- create ``m`` fresh global inputs.
        ``str`` -- only valid for ``m == 1``; resolved via
        :func:`pydae.control.continuos._common.resolve_input`.
        ``list[str]`` -- must contain exactly ``m`` names; each is
        resolved independently and bound to the corresponding
        column of ``K``. Note that, unlike the PI block, the list
        **is not** combined as an error signal.
    output : str, list[str], or None
        Same convention as ``input`` but for the ``p`` output
        channels. Names that were previously global inputs are
        promoted to algebraic variables driven by this block.
    """
    import numpy as np
    from pydae.control.continuos._common import resolve_input

    data_dict = data_dict or {}

    # 1. Defaults (from descriptions())
    meta = descriptions()
    default_map = {item['data']: item['default'] for item in meta if item.get('data')}

    K_in = data_dict.get('K', default_map.get('K'))

    # 2. Validate K and derive dimensions p (rows) and m (cols).
    K, p, m = _check_matrix(K_in, 'K')
    val_u = [float(data_dict.get(f'u_{k + 1}', 0.0)) for k in range(m)]

    # 3. Input channels: fresh symbols, single alias (m == 1 only),
    #    or m aliased names.
    u_exprs = []
    if input is None:
        for k in range(m):
            u_sym = backend.symbols(f"u_{k + 1}_matrix_gain_{name}")
            dae['u_ini_dict'][f"{u_sym}"] = val_u[k]
            dae['u_run_dict'][f"{u_sym}"] = val_u[k]
            u_exprs.append(u_sym)
    elif isinstance(input, str):
        if m != 1:
            raise ValueError(
                f"matrix_gain: a single input name was given but K has m = {m} "
                f"columns; pass a list of {m} names"
            )
        u_exprs = [resolve_input(input, dae, backend, val_u[0])]
    else:
        names = list(input)
        if len(names) != m:
            raise ValueError(
                f"matrix_gain: input list has {len(names)} entries, K has "
                f"m = {m} columns"
            )
        for k, nm in enumerate(names):
            u_exprs.append(resolve_input(nm, dae, backend, val_u[k]))

    # 4. Output channels: fresh symbols, single alias (p == 1 only),
    #    or p aliased names.
    z_syms = []
    z_aliased_flags = []
    if output is None:
        for j in range(p):
            z_syms.append(backend.symbols(f"z_{j + 1}_matrix_gain_{name}"))
            z_aliased_flags.append(False)
    elif isinstance(output, str):
        if p != 1:
            raise ValueError(
                f"matrix_gain: a single output name was given but K has p = {p} "
                f"rows; pass a list of {p} names"
            )
        z_syms.append(backend.symbols(output))
        z_aliased_flags.append(True)
    else:
        names_out = list(output)
        if len(names_out) != p:
            raise ValueError(
                f"matrix_gain: output list has {len(names_out)} entries, K has "
                f"p = {p} rows"
            )
        for nm in names_out:
            z_syms.append(backend.symbols(nm))
            z_aliased_flags.append(True)

    # 5. Algebraic equations: z_j = sum_k K[j][k] * u_k.
    #    Zero coefficients are skipped from the symbolic sum.
    def _row_combination(coeffs, syms):
        terms = [c * s for c, s in zip(coeffs, syms) if c != 0]
        if not terms:
            return 0
        expr = terms[0]
        for t in terms[1:]:
            expr = expr + t
        return expr

    g_eqs = [z_syms[j] - _row_combination(K[j], u_exprs) for j in range(p)]

    # 6. Promote any aliased outputs that were previously global inputs.
    for z_sym, aliased in zip(z_syms, z_aliased_flags):
        if aliased:
            dae['u_ini_dict'].pop(str(z_sym), None)
            dae['u_run_dict'].pop(str(z_sym), None)

    # 7. Init hint: z_ss = K @ u_0 (purely algebraic).
    K_np = np.array(K, dtype=float)
    u_0 = np.array(val_u, dtype=float)
    z_ss = K_np @ u_0
    for z_sym, val in zip(z_syms, z_ss):
        dae['xy_0_dict'][f"{z_sym}"] = float(val)

    # 8. Assembly. No params_dict entries: coefficients are baked into
    #    the equations as numeric constants.
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
def _build_matrix_gain_model(name='1', K=None, u_0=None):
    """Assemble a standalone CasadiModel containing a matrix_gain block.

    ``matrix_gain`` is purely algebraic (no differential equations);
    the CasADi builder needs at least one ``f`` equation to introspect
    symbols, so a tiny dummy ``lpf1`` with a separate, untouched input
    is added in parallel just to give the dae a state. It does not
    interact with the matrix_gain output.
    """
    from pydae.core.builder.casadi_builder import MathBackend, CasadiBuilder
    from pydae.core.model.casadi_model import CasadiModel
    from pydae.control.continuos.lpf1 import lpf1

    K = K if K is not None else [[1.0]]
    if u_0 is None:
        u_0 = [0.0] * len(K[0])

    dae = {'name': f'test_matrix_gain_{name}',
           'f_list': [], 'x_list': [],
           'g_list': [], 'y_ini_list': [], 'y_run_list': [],
           'u_ini_dict': {}, 'u_run_dict': {},
           'params_dict': {},
           'h_dict': {},
           'xy_0_dict': {}}

    backend = MathBackend(use_casadi=True)
    lpf1(dae, '_dummy', backend, {'T': 1.0, 'u': 0.0})  # provides one state
    dd = {'K': K}
    for k, val in enumerate(u_0):
        dd[f'u_{k + 1}'] = val
    matrix_gain(dae, name, backend, dd)

    model = CasadiModel(CasadiBuilder(dae).build())
    model.ini({}, dae['xy_0_dict'])

    return model


def test_scalar_gain():
    """1 x 1 matrix gain: z = K * u."""
    name = '1'
    K, u_0 = [[2.5]], [0.4]
    model = _build_matrix_gain_model(name=name, K=K, u_0=u_0)
    assert abs(model.get_value(f'z_1_matrix_gain_{name}') - 2.5 * 0.4) < 1e-9


def test_general_matrix_gain():
    """3 x 2 matrix gain at steady state matches numpy z = K @ u."""
    import numpy as np
    name = '1'
    K = [[1.0, 2.0],
         [-1.5, 0.5],
         [0.0, 3.0]]
    u_0 = [0.7, -0.2]
    model = _build_matrix_gain_model(name=name, K=K, u_0=u_0)

    expected = np.array(K) @ np.array(u_0)
    for j, exp in enumerate(expected):
        got = model.get_value(f'z_{j + 1}_matrix_gain_{name}')
        assert abs(got - float(exp)) < 1e-9


def test_input_change_propagates_algebraically():
    """The block is static: a step input must produce an immediate jump
    in the output (no lag at all)."""
    import numpy as np
    name = '1'
    K = [[1.0, -1.0], [2.0, 0.5]]
    model = _build_matrix_gain_model(name=name, K=K, u_0=[0.0, 0.0])

    u_step = [0.6, 0.4]
    model.run(1.0, {f'u_1_matrix_gain_{name}': u_step[0],
                    f'u_2_matrix_gain_{name}': u_step[1]})
    model.post()

    expected = np.array(K) @ np.array(u_step)
    for j, exp in enumerate(expected):
        got = model.get_value(f'z_{j + 1}_matrix_gain_{name}')
        assert abs(got - float(exp)) < 1e-9


def test_input_aliasing_via_h_dict_list():
    """`input=` list with names referring to h_dict expressions binds
    each input column to the corresponding upstream signal."""
    from pydae.core.builder.casadi_builder import MathBackend, CasadiBuilder
    from pydae.core.model.casadi_model import CasadiModel
    from pydae.control.continuos.lpf1 import lpf1

    dae = {'name': 'test_matrix_gain_alias',
           'f_list': [], 'x_list': [],
           'g_list': [], 'y_ini_list': [], 'y_run_list': [],
           'u_ini_dict': {}, 'u_run_dict': {},
           'params_dict': {}, 'h_dict': {}, 'xy_0_dict': {}}
    backend = MathBackend(use_casadi=True)

    lpf1(dae, 'm1', backend, {'T': 0.05, 'u': 0.6})
    lpf1(dae, 'm2', backend, {'T': 0.05, 'u': 0.2})
    matrix_gain(dae, 'mix', backend,
                {'K': [[1.0, -1.0], [2.0, 1.0]],
                 'u_1': 0.6, 'u_2': 0.2},
                input=['z_lpf1_m1', 'z_lpf1_m2'])

    model = CasadiModel(CasadiBuilder(dae).build())
    model.ini({}, dae['xy_0_dict'])
    # Steady state: z = K @ [0.6, 0.2] = [0.4, 1.4]
    assert abs(model.get_value('z_1_matrix_gain_mix') - 0.4) < 1e-9
    assert abs(model.get_value('z_2_matrix_gain_mix') - 1.4) < 1e-9


def test_output_aliasing_promotes_global_input():
    """`output=` aliasing an existing u_run input promotes that input
    to an algebraic state driven by this block (state-feedback pattern:
    `u_ctrl = K * x_meas`)."""
    from pydae.core.builder.casadi_builder import MathBackend, CasadiBuilder
    from pydae.core.model.casadi_model import CasadiModel
    from pydae.control.continuos.lpf1 import lpf1

    dae = {'name': 'test_matrix_gain_out',
           'f_list': [], 'x_list': [],
           'g_list': [], 'y_ini_list': [], 'y_run_list': [],
           'u_ini_dict': {}, 'u_run_dict': {},
           'params_dict': {}, 'h_dict': {}, 'xy_0_dict': {}}
    backend = MathBackend(use_casadi=True)

    # Pre-existing global input that the matrix_gain will then drive.
    backend.symbols('u_ctrl')
    dae['u_ini_dict']['u_ctrl'] = 0.0
    dae['u_run_dict']['u_ctrl'] = 0.0
    # Need a state somewhere.
    lpf1(dae, '_dummy', backend, {'T': 1.0, 'u': 0.0})

    matrix_gain(dae, 'fb', backend,
                {'K': [[2.0, -1.0]], 'u_1': 0.5, 'u_2': 0.3},
                output='u_ctrl')

    assert 'u_ctrl' not in dae['u_run_dict']
    model = CasadiModel(CasadiBuilder(dae).build())
    model.ini({}, dae['xy_0_dict'])
    # u_ctrl = 2 * 0.5 + (-1) * 0.3 = 0.7
    assert abs(model.get_value('u_ctrl') - 0.7) < 1e-9


def test_validation_rejections():
    """Bad K / input / output shapes raise ValueError."""
    from pydae.core.builder.casadi_builder import MathBackend
    def fresh():
        return {'h_dict': {}, 'g_list': [], 'y_ini_list': [], 'y_run_list': [],
                'u_ini_dict': {}, 'u_run_dict': {}, 'params_dict': {},
                'xy_0_dict': {}, 'x_list': [], 'f_list': []}

    cases = [
        ('K rows unequal',     {'K': [[1.0, 2.0], [3.0]]}),
        ('K empty',            {'K': []}),
        ('K zero-length rows', {'K': [[]]}),
        ('input wrong length', {'K': [[1.0, 2.0]], 'input_arg': ['a']}),
        ('output wrong length',{'K': [[1.0, 2.0], [3.0, 4.0]], 'output_arg': ['only_one']}),
        ('single str when m>1',{'K': [[1.0, 2.0]], 'input_arg': 'just_one'}),
        ('single str when p>1',{'K': [[1.0], [2.0]], 'output_arg': 'just_one'}),
    ]
    for label, kwargs in cases:
        ia = kwargs.pop('input_arg', None)
        oa = kwargs.pop('output_arg', None)
        try:
            matrix_gain(fresh(), 't', MathBackend(use_casadi=True), kwargs,
                        input=ia, output=oa)
        except ValueError as e:
            continue
        raise AssertionError(f"should have rejected: {label}")


if __name__ == '__main__':
    name = '1'
    K = [[1.0, 2.0], [-1.5, 0.5], [0.0, 3.0]]
    u_0 = [0.7, -0.2]
    model = _build_matrix_gain_model(name=name, K=K, u_0=u_0)
    model.report_z()
    print(f"z = ({model.get_value('z_1_matrix_gain_1'):+.4f}, "
          f"{model.get_value('z_2_matrix_gain_1'):+.4f}, "
          f"{model.get_value('z_3_matrix_gain_1'):+.4f})")
