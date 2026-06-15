# -*- coding: utf-8 -*-
r"""
Created on Sat June 13 00:00:00 2026

@author: jmmauricio

Stabilizer filter chain with two washouts and three lead-lags
(STAB_1LPF_2WO_3LL).

A cascaded linear filter assembled entirely from the basic
``pydae.control.continuos`` blocks (``lpf1``, ``washout``,
``leadlag``), followed by an output saturation. This is the classic
PSS/POD stabilizer signal path: a low-pass measurement filter, two
washout (high-pass) stages that block any DC offset of the input, and
three lead-lag compensation stages, with the final output hard-limited
to $\pm Limit$.

**Transfer function** (unsaturated)

$$\frac{Z(s)}{U(s)} = K_{stab} \cdot \frac{1}{1+sT_{lpf}}
\cdot \frac{sT_{wo1}}{1+sT_{wo1}} \cdot \frac{sT_{wo2}}{1+sT_{wo2}}
\cdot \frac{1+sT_1}{1+sT_2} \cdot \frac{1+sT_3}{1+sT_4}
\cdot \frac{1+sT_5}{1+sT_6}$$

**Output saturation**

$$0 = z - K_{stab} \cdot \mathrm{clip}\left(z_{unsat}, -Limit/K_{stab}, Limit/K_{stab}\right)$$

(equivalent to $z = \mathrm{clip}(K_{stab} \cdot z_{unsat}, -Limit, Limit)$ for
$K_{stab} > 0$, but clips the $O(1)$ pre-gain signal -- numerically robust
even when $K_{stab}$ is a large unit-conversion factor, e.g. pu to W)

Because of the two washout stages, the steady-state ($s=0$) gain of
this block is always zero: any constant input $u$ produces $z=0$ at
steady state, and only transients pass through, shaped by the
low-pass filter and the three lead-lag stages.
"""

import io


def descriptions():
    """
    Single source of truth for model parameters, inputs, states, and outputs.
    """
    descriptions_list = []

    # Inputs
    descriptions_list += [{"type": "Input", "tex": "u", "data": "u", "model": "u", "default": 0.0, "description": "Stabilizer input signal", "units": "-"}]

    # Parameters
    descriptions_list += [{"type": "Parameter", "tex": "T_{lpf}", "data": "T_lpf", "model": "T_lpf", "default": 0.02, "description": "Input low-pass filter time constant", "units": "s"}]
    descriptions_list += [{"type": "Parameter", "tex": "T_{wo1}", "data": "T_wo1", "model": "T_wo1", "default": 10.0, "description": "Washout 1 time constant", "units": "s"}]
    descriptions_list += [{"type": "Parameter", "tex": "T_{wo2}", "data": "T_wo2", "model": "T_wo2", "default": 10.0, "description": "Washout 2 time constant", "units": "s"}]
    descriptions_list += [{"type": "Parameter", "tex": "T_1", "data": "T_1", "model": "T_1", "default": 0.1, "description": "Lead-lag 1 numerator time constant", "units": "s"}]
    descriptions_list += [{"type": "Parameter", "tex": "T_2", "data": "T_2", "model": "T_2", "default": 0.02, "description": "Lead-lag 1 denominator time constant", "units": "s"}]
    descriptions_list += [{"type": "Parameter", "tex": "T_3", "data": "T_3", "model": "T_3", "default": 0.1, "description": "Lead-lag 2 numerator time constant", "units": "s"}]
    descriptions_list += [{"type": "Parameter", "tex": "T_4", "data": "T_4", "model": "T_4", "default": 0.02, "description": "Lead-lag 2 denominator time constant", "units": "s"}]
    descriptions_list += [{"type": "Parameter", "tex": "T_5", "data": "T_5", "model": "T_5", "default": 0.1, "description": "Lead-lag 3 numerator time constant", "units": "s"}]
    descriptions_list += [{"type": "Parameter", "tex": "T_6", "data": "T_6", "model": "T_6", "default": 0.02, "description": "Lead-lag 3 denominator time constant", "units": "s"}]
    descriptions_list += [{"type": "Parameter", "tex": "K_{stab}", "data": "K_stab", "model": "K_stab", "default": 1.0, "description": "Overall stabilizer gain, applied before the output saturation", "units": "-"}]
    descriptions_list += [{"type": "Parameter", "tex": "Limit", "data": "Limit", "model": "Limit", "default": 0.1, "description": "Output saturation limit (symmetric, $\\pm Limit$)", "units": "-"}]

    # Dynamic States
    descriptions_list += [{"type": "Dynamic State", "tex": "x_{lpf}", "data": "", "model": "x_lpf", "default": "", "description": "Input low-pass filter state", "units": "-"}]
    descriptions_list += [{"type": "Dynamic State", "tex": "x_{wo1}", "data": "", "model": "x_wo1", "default": "", "description": "Washout 1 state", "units": "-"}]
    descriptions_list += [{"type": "Dynamic State", "tex": "x_{wo2}", "data": "", "model": "x_wo2", "default": "", "description": "Washout 2 state", "units": "-"}]
    descriptions_list += [{"type": "Dynamic State", "tex": "x_{ll1}", "data": "", "model": "x_ll1", "default": "", "description": "Lead-lag 1 state", "units": "-"}]
    descriptions_list += [{"type": "Dynamic State", "tex": "x_{ll2}", "data": "", "model": "x_ll2", "default": "", "description": "Lead-lag 2 state", "units": "-"}]
    descriptions_list += [{"type": "Dynamic State", "tex": "x_{ll3}", "data": "", "model": "x_ll3", "default": "", "description": "Lead-lag 3 state", "units": "-"}]

    # Algebraic States
    descriptions_list += [{"type": "Algebraic State", "tex": "z", "data": "", "model": "z", "default": "", "description": "Stabilizer output, saturated to $\\pm Limit$ (exposed in h_dict)", "units": "-"}]

    return descriptions_list


def stab_1lpf_2wo_3ll(dae, name, backend, data_dict=None, input=None, output=None):
    """Append a 2-washout / 3-lead-lag stabilizer filter chain to `dae`.

    Assembles, in series: ``lpf1`` (T_lpf) -> ``washout`` (T_wo1) ->
    ``washout`` (T_wo2) -> ``leadlag`` (T_1/T_2) -> ``leadlag``
    (T_3/T_4) -> ``leadlag`` (T_5/T_6) -> output saturation
    (``hard_limits``, ``+-Limit``).

    By default the composite input/output are new symbols exposed as
    ``u_stab_1lpf_2wo_3ll_{name}`` and ``z_stab_1lpf_2wo_3ll_{name}``
    (the latter through `h_dict`); every internal block keeps its own
    ``..._{name}_lpf`` / ``..._{name}_wo1`` / ``..._{name}_wo2`` /
    ``..._{name}_ll1`` / ``..._{name}_ll2`` / ``..._{name}_ll3`` names
    and remains individually observable through `h_dict`.

    ``input`` / ``output`` -- optional names of *existing* symbols in
    `dae` (e.g. a dynamic state such as ``omega_1``, or a power-flow
    setpoint such as ``P_5`` already declared in ``u_ini_dict`` /
    ``u_run_dict``) to wire the block directly to, without introducing
    the extra ``u_stab_.../z_stab_...`` symbols and connecting
    equations. If ``output`` names an existing ``u_ini``/``u_run``
    input, it is promoted to an algebraic variable driven by this
    block's saturated output.
    """
    from pydae.control.continuos.lpf1 import lpf1
    from pydae.control.continuos.washout import washout
    from pydae.control.continuos.leadlag import leadlag
    from pydae.control.continuos._common import combine_inputs

    data_dict = data_dict or {}

    # 1. Defaults (from descriptions())
    meta = descriptions()
    default_map = {item['model']: item['default'] for item in meta if item.get('default') != ""}

    # 2. Data-dict overrides (fall back to descriptions() defaults)
    val_u = data_dict.get('u', default_map.get('u', 0.0))
    val_T_lpf = data_dict.get('T_lpf', default_map.get('T_lpf', 0.02))
    val_T_wo1 = data_dict.get('T_wo1', default_map.get('T_wo1', 10.0))
    val_T_wo2 = data_dict.get('T_wo2', default_map.get('T_wo2', 10.0))
    val_T_1 = data_dict.get('T_1', default_map.get('T_1', 0.1))
    val_T_2 = data_dict.get('T_2', default_map.get('T_2', 0.02))
    val_T_3 = data_dict.get('T_3', default_map.get('T_3', 0.1))
    val_T_4 = data_dict.get('T_4', default_map.get('T_4', 0.02))
    val_T_5 = data_dict.get('T_5', default_map.get('T_5', 0.1))
    val_T_6 = data_dict.get('T_6', default_map.get('T_6', 0.02))
    val_K_stab = data_dict.get('K_stab', default_map.get('K_stab', 1.0))
    val_Limit = data_dict.get('Limit', default_map.get('Limit', 0.1))

    # 3. External input expression: fresh u_stab_..., aliased symbol,
    #    h_dict expression, or a list combined as input[0] - sum(input[1:])
    #    (see pydae.control.continuos._common.combine_inputs).
    if input is None:
        u = backend.symbols(f"u_stab_1lpf_2wo_3ll_{name}")
        dae['u_ini_dict'][f"{u}"] = val_u
        dae['u_run_dict'][f"{u}"] = val_u
    else:
        u = combine_inputs(input, dae, backend, val_u)

    # 4. External output symbol (fresh or aliased -- promoted below if it
    #    was a global input like a power-modulation setpoint).
    z = backend.symbols(output) if output is not None else backend.symbols(f"z_stab_1lpf_2wo_3ll_{name}")
    K_stab = backend.symbols(f"K_stab_{name}")
    Limit = backend.symbols(f"Limit_stab_1lpf_2wo_3ll_{name}")

    # 5. Sub-blocks: lpf -> washout -> washout -> leadlag -> leadlag -> leadlag
    #    Steady-state seeds: lpf and washout1 pass DC (-> val_u); the
    #    washout output and everything downstream of it is zero at
    #    steady state.
    lpf1(dae, f"{name}_lpf",   backend, {'T':     val_T_lpf, 'u': val_u})
    washout(dae, f"{name}_wo1", backend, {'T_wo': val_T_wo1, 'u': val_u})
    washout(dae, f"{name}_wo2", backend, {'T_wo': val_T_wo2, 'u': 0.0})
    leadlag(dae, f"{name}_ll1", backend, {'T_num': val_T_1, 'T_den': val_T_2, 'u': 0.0})
    leadlag(dae, f"{name}_ll2", backend, {'T_num': val_T_3, 'T_den': val_T_4, 'u': 0.0})
    leadlag(dae, f"{name}_ll3", backend, {'T_num': val_T_5, 'T_den': val_T_6, 'u': 0.0})

    # 6. Chain the blocks together: each downstream block's input was
    #    created as a fixed u_run input by its own builder; promote it
    #    to an algebraic variable driven by the upstream block's output.
    chain = [
        (backend.symbols(f"u_lpf1_{name}_lpf"),    u,                                       val_u),
        (backend.symbols(f"u_washout_{name}_wo1"), dae['h_dict'][f"z_lpf1_{name}_lpf"],      val_u),
        (backend.symbols(f"u_washout_{name}_wo2"), dae['h_dict'][f"z_washout_{name}_wo1"],   0.0),
        (backend.symbols(f"u_leadlag_{name}_ll1"), dae['h_dict'][f"z_washout_{name}_wo2"],   0.0),
        (backend.symbols(f"u_leadlag_{name}_ll2"), dae['h_dict'][f"z_leadlag_{name}_ll1"],   0.0),
        (backend.symbols(f"u_leadlag_{name}_ll3"), dae['h_dict'][f"z_leadlag_{name}_ll2"],   0.0),
    ]

    for downstream_u, upstream_z, val in chain:
        dae['u_ini_dict'].pop(str(downstream_u))
        dae['u_run_dict'].pop(str(downstream_u))
        dae['g_list'] += [downstream_u - upstream_z]
        dae['y_ini_list'] += [downstream_u]
        dae['y_run_list'] += [downstream_u]
        dae['xy_0_dict'].update({f"{downstream_u}": val})

    # 7. Output saturation: z = K_stab * clip(z_ll3, -Limit/K_stab, Limit/K_stab).
    #    Equivalent to clip(K_stab * z_ll3, -Limit, Limit) for K_stab > 0, but
    #    clips the O(1) pre-gain signal instead of the gain-amplified one --
    #    avoids huge intermediate values / flat Jacobian regions during Newton
    #    iterations when K_stab is large (e.g. a pu-to-W conversion factor).
    z_ll3 = dae['h_dict'][f"z_leadlag_{name}_ll3"]
    g_z = z - K_stab * backend.hard_limits(z_ll3, -Limit / K_stab, Limit / K_stab)

    # If `output` aliases an existing u_ini/u_run input (e.g. a
    # power-modulation setpoint), promote it to an algebraic variable
    # driven by this saturation.
    if output is not None:
        dae['u_ini_dict'].pop(str(z), None)
        dae['u_run_dict'].pop(str(z), None)

    dae['g_list'] += [g_z]
    dae['y_ini_list'] += [z]
    dae['y_run_list'] += [z]
    dae['h_dict'].update({f"{z}": z})
    dae['params_dict'].update({f"{K_stab}": val_K_stab, f"{Limit}": val_Limit})
    dae['xy_0_dict'].update({f"{z}": 0.0})

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
def _build_stab_1lpf_2wo_3ll_model(name='1', T_lpf=0.02, T_wo1=2.0, T_wo2=2.0,
                               T_1=0.3, T_2=0.05, T_3=0.3, T_4=0.05,
                               T_5=0.3, T_6=0.05, Limit=10.0, u_0=0.0):
    """Assemble a standalone CasadiModel containing a single STAB_1LPF_2WO_3LL block."""
    from pydae.core.builder.casadi_builder import MathBackend, CasadiBuilder
    from pydae.core.model.casadi_model import CasadiModel

    dae = {'name': f'test_stab_1lpf_2wo_3ll_{name}',
           'f_list': [], 'x_list': [],
           'g_list': [], 'y_ini_list': [], 'y_run_list': [],
           'u_ini_dict': {}, 'u_run_dict': {},
           'params_dict': {},
           'h_dict': {},
           'xy_0_dict': {}}

    backend = MathBackend(use_casadi=True)
    stab_1lpf_2wo_3ll(dae, name, backend, {'T_lpf': T_lpf, 'T_wo1': T_wo1, 'T_wo2': T_wo2,
                              'T_1': T_1, 'T_2': T_2, 'T_3': T_3, 'T_4': T_4,
                              'T_5': T_5, 'T_6': T_6, 'Limit': Limit,
                              'u': u_0})

    model = CasadiModel(CasadiBuilder(dae).build())
    model.ini({}, dae['xy_0_dict'])

    return model


def test_steady_state_output_is_zero():
    """At steady state (du/dt = 0), the two washout stages drive the
    output z to zero, regardless of the operating-point value of u."""
    model = _build_stab_1lpf_2wo_3ll_model(name='1', u_0=0.7)

    assert abs(model.get_value('z_stab_1lpf_2wo_3ll_1') - 0.0) < 1e-6


def test_step_response_transient_then_decays_to_zero():
    """A step in u excites a transient (shaped by the low-pass filter
    and the three lead-lag stages) which then decays back to zero as
    the washout stages remove the new DC component.
    """
    name = '1'
    T_wo1, T_wo2 = 2.0, 2.0
    model = _build_stab_1lpf_2wo_3ll_model(name=name, T_wo1=T_wo1, T_wo2=T_wo2, u_0=0.0)

    model.run(10 * max(T_wo1, T_wo2), {f'u_stab_1lpf_2wo_3ll_{name}': 1.0})
    model.post()

    z = model.get_values(f'z_stab_1lpf_2wo_3ll_{name}')

    assert max(abs(z)) > 0.1     # transient excitation
    assert abs(z[-1] - 0.0) < 0.01  # washed out at steady state


def test_saturation_limits_output():
    """A large step in u drives the unsaturated chain output beyond
    +-Limit; the output z stays clipped at the saturation limit during
    the transient.
    """
    name = '1'
    Limit = 0.05
    model = _build_stab_1lpf_2wo_3ll_model(name=name, Limit=Limit, u_0=0.0)

    model.run(0.5, {f'u_stab_1lpf_2wo_3ll_{name}': 1.0})
    model.post()

    z = model.get_values(f'z_stab_1lpf_2wo_3ll_{name}')

    assert abs(max(abs(z)) - Limit) < 1e-3
    assert (abs(z) <= Limit + 1e-6).all()


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    name = '1'
    T_wo1, T_wo2 = 2.0, 2.0
    model = _build_stab_1lpf_2wo_3ll_model(name=name, T_wo1=T_wo1, T_wo2=T_wo2, u_0=0.0)

    model.report_u()
    model.report_x()
    model.report_z()
    model.report_params()

    model.run(10 * max(T_wo1, T_wo2), {f'u_stab_1lpf_2wo_3ll_{name}': 1.0})
    model.post()

    fig, ax = plt.subplots()
    ax.plot(model.Time, model.get_values(f'z_stab_1lpf_2wo_3ll_{name}'), label='$z$ (stabilizer output)')
    ax.axhline(1.0, color='gray', ls=':', lw=0.8, label='$u$ (step input)')
    ax.axhline(0.0, color='k', ls='--', lw=0.8, label='steady state')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Output')
    ax.legend()
    fig.savefig('stab_1lpf_2wo_3ll_step_response.svg')
    print("Test completed. Plot saved as 'stab_1lpf_2wo_3ll_step_response.svg'.")
