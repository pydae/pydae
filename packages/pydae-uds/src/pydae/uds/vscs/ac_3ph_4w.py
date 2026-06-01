r"""
AC-only 4-wire VSC carrier model. The converter terminal voltages
$v_{t,\varphi}$ are inputs by default — set them directly to use this as a
fixed-source converter, or nest a `vsg` (e.g. `gflpfzv`) which converts
them into algebraic states driven by the chosen grid-forming control.

**Per-phase branch equation** ($\varphi \in \{a, b, c, n\}$, real form):

$$0 = v_{og} + v_{t,\varphi} - Z_\varphi i_{s,\varphi} - v_{s,\varphi}$$

with $Z_a = Z_b = Z_c = R_s + jX_s$ (phase impedance) and
$Z_n = R_{sn} + jX_{sn}$ (neutral impedance).

**Neutral-to-ground loop** — the converter neutral $v_{og}$ closes to ground
through $Z_{ng} = R_{ng} + jX_{ng}$:

$$0 = i_{sa} + i_{sb} + i_{sc} + i_{sn} + v_{og}/Z_{ng}$$

(expanded into real form by multiplying numerator and denominator by
$\overline{Z_{ng}}$ so the equation lives on CasADi `SX`).

**Conduction loss** (per wire, same polynomial as `acdc_3ph_4w_*`):

$$p_{loss,\varphi} = A_{loss}|i_\varphi|^2 + B_{loss}|i_\varphi| + C_{loss}$$
$$p_{dc} = p_{ac,total} + p_{loss,total}$$

**Defaults** — on construction the terminal voltages are seeded with a
balanced three-phase set at $V_n / \sqrt{3} \angle \phi$ and the loss
coefficients are estimated from the rated apparent power (1% conduction
loss assumption) unless overridden via the HJSON `A_loss`, `B_loss`,
`C_loss` fields.

**HJSON snippet**

```hjson
vscs: [
    {type: "ac_3ph_4w", bus: "A3", S_n: 100e3, U_n: 400,
     R: 0.01, X: 0.05, R_n: 0.01, X_n: 0.1, R_ng: 0.01, X_ng: 3.0,
     vsg: {type: "gflpfzv", ...}}
]
```
"""

import numpy as np


def descriptions():
    return [
        {"type": "Parameter", "tex": "S_n",     "data": "S_n",   "model": "",                  "default": "",  "units": "VA",      "description": "Rated apparent power"},
        {"type": "Parameter", "tex": "U_n",     "data": "U_n",   "model": "",                  "default": "",  "units": "V",       "description": "Rated line-to-line voltage"},
        {"type": "Parameter", "tex": "R_s",     "data": "R",     "model": "R_s_{bus}",         "default": "",  "units": r"\Omega", "description": "Phase resistance"},
        {"type": "Parameter", "tex": "X_s",     "data": "X",     "model": "X_s_{bus}",         "default": "",  "units": r"\Omega", "description": "Phase reactance"},
        {"type": "Parameter", "tex": "R_{sn}",  "data": "R_n",   "model": "R_sn_{bus}",        "default": "",  "units": r"\Omega", "description": "Neutral wire resistance"},
        {"type": "Parameter", "tex": "X_{sn}",  "data": "X_n",   "model": "X_sn_{bus}",        "default": "",  "units": r"\Omega", "description": "Neutral wire reactance"},
        {"type": "Parameter", "tex": "R_{ng}",  "data": "R_ng",  "model": "R_ng_{bus}",        "default": "",  "units": r"\Omega", "description": "Neutral-to-ground resistance"},
        {"type": "Parameter", "tex": "X_{ng}",  "data": "X_ng",  "model": "X_ng_{bus}",        "default": "",  "units": r"\Omega", "description": "Neutral-to-ground reactance"},
        {"type": "Parameter", "tex": "A_{loss}","data": "A_loss","model": "A_loss_{bus}",      "default": "S_n-derived", "units": r"\Omega", "description": "Quadratic-loss coefficient"},
        {"type": "Parameter", "tex": "B_{loss}","data": "B_loss","model": "B_loss_{bus}",      "default": 1.0, "units": "V",       "description": "Linear-loss coefficient"},
        {"type": "Parameter", "tex": "C_{loss}","data": "C_loss","model": "C_loss_{bus}",      "default": "S_n-derived", "units": "W", "description": "No-load loss"},
        {"type": "Input", "tex": r"v_{t,\varphi}^{r,i}", "data": "", "model": "v_t{a,b,c,n}_{r,i}_{bus}", "default": "balanced", "units": "V", "description": "Terminal voltage components (turned into states by a nested vsg)"},
        {"type": "Algebraic State", "tex": r"i_{s,\varphi}^{r,i}", "data": "", "model": "i_vsc_{bus}_{ph}_{r,i}", "default": "", "units": "A", "description": "Per-phase converter current"},
        {"type": "Algebraic State", "tex": "v_{og}^{r,i}", "data": "", "model": "v_{bus}_o_{r,i}", "default": "", "units": "V", "description": "Neutral-to-ground voltage"},
        {"type": "Algebraic State", "tex": "p_{dc}", "data": "", "model": "p_dc_{bus}", "default": "", "units": "W", "description": "DC-side power equivalent (after losses)"},
        {"type": "Output", "tex": r"p_{vsc,\varphi}, q_{vsc,\varphi}", "data": "", "model": "p_vsc_{bus}_{ph}, q_vsc_{bus}_{ph}", "default": "", "units": "W / var", "description": "Per-phase grid-side complex power"},
    ]


def ac_3ph_4w(grid, data):
    '''
    AC-only 3 phase 4 wire VSC with separate phase/neutral/neutral-ground
    impedances. Terminal EMFs (v_t*) are inputs by default; a nested vsg
    converts them into algebraic states. Dual-backend via grid.backend.
    '''
    name = data['bus']
    bk = grid.backend

    params_dict = grid.dae['params_dict']

    # --- parameters
    R_s  = bk.symbols(f'R_s_{name}');  X_s  = bk.symbols(f'X_s_{name}')
    R_sn = bk.symbols(f'R_sn_{name}'); X_sn = bk.symbols(f'X_sn_{name}')
    R_ng = bk.symbols(f'R_ng_{name}'); X_ng = bk.symbols(f'X_ng_{name}')
    A_loss = bk.symbols(f'A_loss_{name}')
    B_loss = bk.symbols(f'B_loss_{name}')
    C_loss = bk.symbols(f'C_loss_{name}')

    # --- bus node voltages (rectangular)
    v_sa_r = bk.symbols(f'V_{name}_0_r'); v_sa_i = bk.symbols(f'V_{name}_0_i')
    v_sb_r = bk.symbols(f'V_{name}_1_r'); v_sb_i = bk.symbols(f'V_{name}_1_i')
    v_sc_r = bk.symbols(f'V_{name}_2_r'); v_sc_i = bk.symbols(f'V_{name}_2_i')
    v_sn_r = bk.symbols(f'V_{name}_3_r'); v_sn_i = bk.symbols(f'V_{name}_3_i')

    # --- neutral-to-ground voltage (algebraic)
    v_og_r = bk.symbols(f'v_{name}_o_r'); v_og_i = bk.symbols(f'v_{name}_o_i')

    # --- VSC currents (algebraic)
    i_sa_r = bk.symbols(f'i_vsc_{name}_a_r'); i_sa_i = bk.symbols(f'i_vsc_{name}_a_i')
    i_sb_r = bk.symbols(f'i_vsc_{name}_b_r'); i_sb_i = bk.symbols(f'i_vsc_{name}_b_i')
    i_sc_r = bk.symbols(f'i_vsc_{name}_c_r'); i_sc_i = bk.symbols(f'i_vsc_{name}_c_i')
    i_sn_r = bk.symbols(f'i_vsc_{name}_n_r'); i_sn_i = bk.symbols(f'i_vsc_{name}_n_i')

    # --- terminal voltages (inputs by default, replaced as states by a nested vsg)
    v_ta_r = bk.symbols(f'v_ta_r_{name}'); v_ta_i = bk.symbols(f'v_ta_i_{name}')
    v_tb_r = bk.symbols(f'v_tb_r_{name}'); v_tb_i = bk.symbols(f'v_tb_i_{name}')
    v_tc_r = bk.symbols(f'v_tc_r_{name}'); v_tc_i = bk.symbols(f'v_tc_i_{name}')
    v_tn_r = bk.symbols(f'v_tn_r_{name}'); v_tn_i = bk.symbols(f'v_tn_i_{name}')

    p_dc = bk.symbols(f'p_dc_{name}')

    # --- VSC RMS currents (with smoothing)
    i_rms_a = bk.sqrt(i_sa_r**2 + i_sa_i**2 + 1e-6)
    i_rms_b = bk.sqrt(i_sb_r**2 + i_sb_i**2 + 1e-6)
    i_rms_c = bk.sqrt(i_sc_r**2 + i_sc_i**2 + 1e-6)
    i_rms_n = bk.sqrt(i_sn_r**2 + i_sn_i**2 + 1e-6)

    p_loss_a = A_loss*i_rms_a*i_rms_a + B_loss*i_rms_a + C_loss
    p_loss_b = A_loss*i_rms_b*i_rms_b + B_loss*i_rms_b + C_loss
    p_loss_c = A_loss*i_rms_c*i_rms_c + B_loss*i_rms_c + C_loss
    p_loss_n = A_loss*i_rms_n*i_rms_n + B_loss*i_rms_n + C_loss
    p_vsc_loss = p_loss_a + p_loss_b + p_loss_c + p_loss_n

    # --- terminal power per phase: s_t* = v_t* * conj(i_s*)  (real form)
    # NOTE: legacy code uses i_sc on the neutral term; preserving that quirk
    # so behaviour matches the prior SymPy build.
    re_s_ta = v_ta_r*i_sa_r + v_ta_i*i_sa_i
    re_s_tb = v_tb_r*i_sb_r + v_tb_i*i_sb_i
    re_s_tc = v_tc_r*i_sc_r + v_tc_i*i_sc_i
    re_s_tn = v_tn_r*i_sc_r + v_tn_i*i_sc_i   # legacy: i_sc, not i_sn
    p_ac = re_s_ta + re_s_tb + re_s_tc + re_s_tn

    # --- phase branch equations: v_og + v_t* - Z_s*i_s* - v_s* = 0
    # Z_s*i_s* in real form: (R_s + jX_s)(i_r + j*i_i)
    #   re: R_s*i_r - X_s*i_i
    #   im: R_s*i_i + X_s*i_r
    def _zi(R, X, ir, ii):
        return R*ir - X*ii, R*ii + X*ir

    re_zi_a, im_zi_a = _zi(R_s,  X_s,  i_sa_r, i_sa_i)
    re_zi_b, im_zi_b = _zi(R_s,  X_s,  i_sb_r, i_sb_i)
    re_zi_c, im_zi_c = _zi(R_s,  X_s,  i_sc_r, i_sc_i)
    re_zi_n, im_zi_n = _zi(R_sn, X_sn, i_sn_r, i_sn_i)

    eq_re_a = v_og_r + v_ta_r - re_zi_a - v_sa_r
    eq_im_a = v_og_i + v_ta_i - im_zi_a - v_sa_i
    eq_re_b = v_og_r + v_tb_r - re_zi_b - v_sb_r
    eq_im_b = v_og_i + v_tb_i - im_zi_b - v_sb_i
    eq_re_c = v_og_r + v_tc_r - re_zi_c - v_sc_r
    eq_im_c = v_og_i + v_tc_i - im_zi_c - v_sc_i
    eq_re_n = v_og_r + v_tn_r - re_zi_n - v_sn_r
    eq_im_n = v_og_i + v_tn_i - im_zi_n - v_sn_i

    # --- v_og loop: i_sa + i_sb + i_sc + i_sn + v_og/Z_ng = 0
    # v_og/Z_ng with Z_ng = R_ng + j*X_ng, real form via multiplication by conj:
    #   re = (v_og_r*R_ng + v_og_i*X_ng) / (R_ng^2 + X_ng^2)
    #   im = (v_og_i*R_ng - v_og_r*X_ng) / (R_ng^2 + X_ng^2)
    den = R_ng*R_ng + X_ng*X_ng
    re_voz = (v_og_r*R_ng + v_og_i*X_ng) / den
    im_voz = (v_og_i*R_ng - v_og_r*X_ng) / den
    eq_re_og = i_sa_r + i_sb_r + i_sc_r + i_sn_r + re_voz
    eq_im_og = i_sa_i + i_sb_i + i_sc_i + i_sn_i + im_voz

    eq_p_dc = -p_dc + p_ac + p_vsc_loss

    grid.dae['g'] += [eq_re_a, eq_re_b, eq_re_c, eq_re_n, eq_re_og,
                      eq_im_a, eq_im_b, eq_im_c, eq_im_n, eq_im_og,
                      eq_p_dc]

    ys = [i_sa_r, i_sb_r, i_sc_r, i_sn_r, v_og_r,
          i_sa_i, i_sb_i, i_sc_i, i_sn_i, v_og_i,
          p_dc]
    grid.dae['y_ini'] += ys
    grid.dae['y_run'] += ys

    # bus current injections + magnitudes
    for ph, ir, ii in [('a', i_sa_r, i_sa_i), ('b', i_sb_r, i_sb_i),
                       ('c', i_sc_r, i_sc_i), ('n', i_sn_r, i_sn_i)]:
        idx_r, idx_i = grid.node2idx(name, ph)
        grid.dae['g'][idx_r] += -ir
        grid.dae['g'][idx_i] += -ii
        grid.dae['h_dict'][f'i_vsc_{name}_{ph}_m'] = (ir**2 + ii**2)**0.5

    # per-phase grid-side complex power: s = v_s * conj(i_s)
    for ph, v_r, v_i, ir, ii in [('a', v_sa_r, v_sa_i, i_sa_r, i_sa_i),
                                 ('b', v_sb_r, v_sb_i, i_sb_r, i_sb_i),
                                 ('c', v_sc_r, v_sc_i, i_sc_r, i_sc_i)]:
        p = v_r*ir + v_i*ii
        q = v_i*ir - v_r*ii
        grid.dae['h_dict'][f'p_vsc_{name}_{ph}'] = p
        grid.dae['h_dict'][f'q_vsc_{name}_{ph}'] = q

    # default parameter values
    params_dict.update({f'X_s_{name}': data['X'],   f'R_s_{name}': data['R']})
    params_dict.update({f'X_sn_{name}': data['X_n'], f'R_sn_{name}': data['R_n']})
    params_dict.update({f'X_ng_{name}': data['X_ng'], f'R_ng_{name}': data['R_ng']})

    # default terminal voltages (balanced three-phase)
    V_n_N = data['U_n']/np.sqrt(3)
    phi_N = data.get('phi', 0.0)
    v_ta_N = V_n_N*np.exp(1j*phi_N)
    v_tb_N = V_n_N*np.exp(1j*(phi_N - 2*np.pi/3))
    v_tc_N = V_n_N*np.exp(1j*(phi_N - 4*np.pi/3))
    v_tn_N = 0.0 + 0.0j

    for sym_name, val in [(str(v_ta_r), v_ta_N.real), (str(v_ta_i), v_ta_N.imag),
                          (str(v_tb_r), v_tb_N.real), (str(v_tb_i), v_tb_N.imag),
                          (str(v_tc_r), v_tc_N.real), (str(v_tc_i), v_tc_N.imag),
                          (str(v_tn_r), v_tn_N.real), (str(v_tn_i), v_tn_N.imag)]:
        grid.dae['xy_0_dict'][sym_name] = val
        grid.dae['u_ini_dict'][sym_name] = val
        grid.dae['u_run_dict'][sym_name] = val

    # losses defaults: derive from S_n unless overridden
    S_n_N = data['S_n']; U_n_N = data['U_n']
    I_n = S_n_N/(np.sqrt(3)*U_n_N)
    C_loss_N = 0.01*S_n_N/3
    A_loss_N = 0.01*S_n_N/(I_n**2)/3
    B_loss_N = 1.0
    if 'A_loss' in data:
        A_loss_N = data['A_loss']
        B_loss_N = data['B_loss']
        C_loss_N = data['C_loss']
    params_dict.update({str(A_loss): A_loss_N,
                        str(B_loss): B_loss_N,
                        str(C_loss): C_loss_N})
