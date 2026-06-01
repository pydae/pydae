r"""
Grid-following + Primary-Frequency-Response VSG with LPF and per-phase
Q-PI control.

Hosts inside an `ac_3ph_4w` VSC entry (see HJSON below). The VSG generates
the converter terminal EMFs $e_{ao}, e_{bo}, e_{co}$ from a rotating frame
$\phi(t)$ plus a per-phase magnitude that's slowly tracked by an outer-loop
PI on the **positive-sequence** reactive power.

**Symmetrical-component decomposition** of the grid-side voltage and
current ($\alpha = e^{j2\pi/3}$, in real form for backend portability):

$$\begin{bmatrix}v_0 \\ v_+ \\ v_-\end{bmatrix} =
\frac{1}{3}\begin{bmatrix}
 1 & 1 & 1 \\
 1 & \alpha & \alpha^2 \\
 1 & \alpha^2 & \alpha
\end{bmatrix}
\begin{bmatrix}v_a \\ v_b \\ v_c\end{bmatrix}$$

$$p_+ = 3\,\Re(v_+ \overline{i_+}), \qquad
  q_+ = 3\,\Im(v_+ \overline{i_+})$$

**Dynamic states** — swing equation with droop + power LPF:

$$\dot{\phi} = \Omega_b (\omega - \omega_{coi}) - K_\delta \phi$$
$$\dot{\omega} = \frac{1}{2H}\bigl(p_m - p_{ef} - D(\omega - 1)\bigr)$$
$$\dot{p_{ef}} = \frac{1}{T_e}\!\left(\frac{p_+}{S_n} - p_{ef}\right)$$
$$\dot{p_{cf}} = \frac{1}{T_c}\!\bigl(p_c - p_{cf}\bigr)$$
$$\dot{p_{pfr}} = \frac{1}{T_{pfr}}\!\left(\frac{1}{R}(\omega - \omega_{ref}) - p_{pfr}\right)$$

**Reactive PI with saturation + conditional-integration anti-windup**:

$$\varepsilon_q = q_{ref} - q_+/S_n$$
$$\Delta_{q,\text{nosat}} = K_{qp}\varepsilon_q + K_{qi}\xi_q$$
$$\Delta_q = \text{clip}(\Delta_{q,\text{nosat}}, -0.05\,U_n, +0.05\,U_n)$$
$$\dot{\xi_q} = K_{qaw}\varepsilon_q - (1 - K_{qaw})\xi_q$$

where $K_{qaw} \in \{0, 1\}$ is the anti-windup indicator (0 once saturated).
The saturation uses `bk.hard_limits` (so it codegens cleanly on both
backends); the indicator stays on `bk.Piecewise` because it's a true step.

**Per-phase EMF amplitude** — each phase has its own $\Delta e_{*o,m}$
state tracking $v_{*r} + \Delta_q$ with first-order lag $T_v$:

$$\dot{\Delta e_{ao,m}} = \frac{1}{T_v}\bigl(v_{ra} + \Delta_q - \Delta e_{ao,m}\bigr)$$

**Virtual-impedance terminal voltage** (writes the EMFs back into the
host VSC's $v_{t*}$ algebraic vars):

$$0 = e_{*o} - (R_v + jX_v) i_{s\varphi} - v_{t\varphi}$$

**HJSON snippet** (nested under an `ac_3ph_4w` VSC):

```hjson
vsg: {type: "gflpfzv", bus: "A3", S_n: 100e3, U_n: 400,
      R_v: 0, X_v: 0.1,
      H: 5.0, D: 0.1, T_e: 0.1, T_c: 0.1, T_v: 0.1, T_pfr: 0.2,
      Droop: 0.05, K_qp: 0.01, K_qi: 0.01,
      K_agc: 0.0, K_delta: 0.0, K_sec: 0.0}
```
"""

import numpy as np


def descriptions():
    return [
        # parameters
        {"type": "Parameter", "tex": "S_n",   "data": "S_n",  "model": "S_n_{name}",   "default": "", "units": "VA",  "description": "Nominal apparent power"},
        {"type": "Parameter", "tex": "U_n",   "data": "U_n",  "model": "U_n_{name}",   "default": "", "units": "V",   "description": "Nominal line-to-line voltage"},
        {"type": "Parameter", "tex": "H",     "data": "H",    "model": "H_{name}",     "default": "", "units": "s",   "description": "Virtual inertia constant"},
        {"type": "Parameter", "tex": "D",     "data": "D",    "model": "D_{name}",     "default": "", "units": "-",   "description": "Damping coefficient"},
        {"type": "Parameter", "tex": "T_e",   "data": "T_e",  "model": "T_e_{name}",   "default": "", "units": "s",   "description": "Power-measurement LPF time constant"},
        {"type": "Parameter", "tex": "T_c",   "data": "T_c",  "model": "T_c_{name}",   "default": "", "units": "s",   "description": "Command LPF time constant"},
        {"type": "Parameter", "tex": "T_v",   "data": "T_v",  "model": "T_v_{name}",   "default": "", "units": "s",   "description": "Voltage-amplitude LPF time constant"},
        {"type": "Parameter", "tex": "T_{pfr}","data": "T_pfr","model": "T_pfr_{name}","default": "", "units": "s",   "description": "PFR LPF time constant"},
        {"type": "Parameter", "tex": "R",     "data": "Droop","model": "Droop_{name}", "default": "", "units": "-",   "description": "PFR droop (pu/pu)"},
        {"type": "Parameter", "tex": "K_{qp}","data": "K_qp", "model": "K_qp_{name}",  "default": "", "units": "-",   "description": "Reactive-PI proportional gain"},
        {"type": "Parameter", "tex": "K_{qi}","data": "K_qi", "model": "K_qi_{name}",  "default": "", "units": "1/s", "description": "Reactive-PI integral gain"},
        {"type": "Parameter", "tex": "K_{agc}","data": "K_agc","model": "K_agc_{name}","default": "", "units": "-",   "description": "AGC participation factor"},
        {"type": "Parameter", "tex": "K_\\delta","data": "K_delta","model": "K_delta_{name}","default": "", "units": "1/s", "description": "Angle reference-pull gain"},
        {"type": "Parameter", "tex": "R_v",   "data": "R_v",  "model": "R_v_{name}",   "default": "", "units": "pu",  "description": "Virtual resistance"},
        {"type": "Parameter", "tex": "X_v",   "data": "X_v",  "model": "X_v_{name}",   "default": "", "units": "pu",  "description": "Virtual reactance"},
        # inputs
        {"type": "Input", "tex": "p_c",        "data": "", "model": "p_c_{name}",        "default": 0.0, "units": "pu", "description": "Active-power command"},
        {"type": "Input", "tex": "q_{ref}",    "data": "", "model": "q_ref_{name}",      "default": 0.0, "units": "pu", "description": "Reactive-power reference"},
        {"type": "Input", "tex": "\\omega_{ref}","data": "","model": "omega_{name}_ref","default": 1.0, "units": "pu", "description": "Frequency reference for PFR"},
        {"type": "Input", "tex": "e_{\\varphi o}^m", "data": "", "model": "e_{a,b,c}o_m_{name}", "default": r"V_n/\sqrt{3}", "units": "V", "description": "Per-phase EMF magnitude (set-point)"},
        {"type": "Input", "tex": "\\phi_\\varphi","data": "", "model": "phi_{a,b,c,n}_{name}", "default": 0.0, "units": "rad", "description": "Per-phase angle offset"},
        {"type": "Input", "tex": "v_{r\\varphi}","data": "", "model": "v_r{a,b,c,n}_{name}", "default": 0.0, "units": "V", "description": "Per-phase voltage residual injection"},
        # dynamic states
        {"type": "Dynamic State", "tex": "\\phi",   "data": "", "model": "phi_{name}",   "default": "", "units": "rad", "description": "Internal swing angle"},
        {"type": "Dynamic State", "tex": "\\omega", "data": "", "model": "omega_{name}", "default": 1.0, "units": "pu", "description": "Rotor (virtual) speed"},
        {"type": "Dynamic State", "tex": "\\xi_q",  "data": "", "model": "xi_q_{name}",  "default": "", "units": "pu·s", "description": "Reactive-PI integrator state"},
        {"type": "Dynamic State", "tex": "p_{ef}",  "data": "", "model": "p_ef_{name}",  "default": "", "units": "pu", "description": "Filtered positive-sequence active power"},
        {"type": "Dynamic State", "tex": "p_{cf}",  "data": "", "model": "p_cf_{name}",  "default": "", "units": "pu", "description": "Filtered active-power command"},
        {"type": "Dynamic State", "tex": "p_{pfr}", "data": "", "model": "p_pfr_{name}", "default": "", "units": "pu", "description": "Filtered PFR output"},
        {"type": "Dynamic State", "tex": "\\Delta e_{\\varphi o,m}", "data": "", "model": "De_{a,b,c,n}o_m_{name}", "default": "", "units": "V", "description": "Per-phase EMF amplitude correction"},
        # algebraic states
        {"type": "Algebraic State", "tex": "p_m", "data": "", "model": "p_m_{name}", "default": "", "units": "pu", "description": "Mechanical power balance variable"},
        {"type": "Algebraic State", "tex": r"v_{t\varphi}^{r,i}", "data": "", "model": "v_t{a,b,c}_{r,i}_{name}", "default": "", "units": "V", "description": "Per-phase terminal voltage (host VSC's input turned into state)"},
        # outputs
        {"type": "Output", "tex": "p_+",  "data": "", "model": "p_pos_{name}", "default": "", "units": "W",   "description": "Positive-sequence active power"},
        {"type": "Output", "tex": "q_+",  "data": "", "model": "q_pos_{name}", "default": "", "units": "var", "description": "Positive-sequence reactive power"},
        {"type": "Output", "tex": "p_-",  "data": "", "model": "p_neg_{name}", "default": "", "units": "W",   "description": "Negative-sequence active power"},
        {"type": "Output", "tex": "p_0",  "data": "", "model": "p_zer_{name}", "default": "", "units": "W",   "description": "Zero-sequence active power"},
    ]


# Sequence-transform constants (alpha = e^{j*2π/3})
_A_R = -0.5
_A_I = np.sin(2*np.pi/3)        # +√3/2
_A2_R = -0.5
_A2_I = -np.sin(2*np.pi/3)      # -√3/2


def _cmul(a_r, a_i, b_r, b_i):
    """complex * complex in real form -> (re, im)"""
    return a_r*b_r - a_i*b_i, a_r*b_i + a_i*b_r


def gflpfzv(grid, data, name, bus_name):
    bk = grid.backend

    # --- inputs / parameters (algebraic + parametric)
    xi_freq   = bk.symbols('xi_freq')
    omega_coi = bk.symbols('omega_coi')

    S_n  = bk.symbols(f'S_n_{name}')
    U_n  = bk.symbols(f'U_n_{name}')
    K_agc   = bk.symbols(f'K_agc_{name}')
    T_e     = bk.symbols(f'T_e_{name}')
    H       = bk.symbols(f'H_{name}')
    D       = bk.symbols(f'D_{name}')
    T_c     = bk.symbols(f'T_c_{name}')
    T_v     = bk.symbols(f'T_v_{name}')
    K_qp    = bk.symbols(f'K_qp_{name}')
    K_qi    = bk.symbols(f'K_qi_{name}')
    Droop   = bk.symbols(f'Droop_{name}')
    R_v     = bk.symbols(f'R_v_{name}')
    X_v     = bk.symbols(f'X_v_{name}')
    K_delta = bk.symbols(f'K_delta_{name}')
    K_soc   = bk.symbols(f'K_soc_{name}')
    T_pfr   = bk.symbols(f'T_pfr_{name}')

    omega_ref = bk.symbols(f'omega_{name}_ref')
    p_c       = bk.symbols(f'p_c_{name}')
    q_ref     = bk.symbols(f'q_ref_{name}')

    e_ao_m = bk.symbols(f'e_ao_m_{name}')
    e_bo_m = bk.symbols(f'e_bo_m_{name}')
    e_co_m = bk.symbols(f'e_co_m_{name}')

    phi_a = bk.symbols(f'phi_a_{name}'); phi_b = bk.symbols(f'phi_b_{name}')
    phi_c = bk.symbols(f'phi_c_{name}'); phi_n = bk.symbols(f'phi_n_{name}')

    v_ra = bk.symbols(f'v_ra_{name}'); v_rb = bk.symbols(f'v_rb_{name}')
    v_rc = bk.symbols(f'v_rc_{name}'); v_rn = bk.symbols(f'v_rn_{name}')

    # --- bus node voltages & VSC currents (rectangular)
    v_sa_r = bk.symbols(f'V_{bus_name}_0_r'); v_sa_i = bk.symbols(f'V_{bus_name}_0_i')
    v_sb_r = bk.symbols(f'V_{bus_name}_1_r'); v_sb_i = bk.symbols(f'V_{bus_name}_1_i')
    v_sc_r = bk.symbols(f'V_{bus_name}_2_r'); v_sc_i = bk.symbols(f'V_{bus_name}_2_i')

    i_sa_r = bk.symbols(f'i_vsc_{name}_a_r'); i_sa_i = bk.symbols(f'i_vsc_{name}_a_i')
    i_sb_r = bk.symbols(f'i_vsc_{name}_b_r'); i_sb_i = bk.symbols(f'i_vsc_{name}_b_i')
    i_sc_r = bk.symbols(f'i_vsc_{name}_c_r'); i_sc_i = bk.symbols(f'i_vsc_{name}_c_i')

    v_ta_r = bk.symbols(f'v_ta_r_{name}'); v_ta_i = bk.symbols(f'v_ta_i_{name}')
    v_tb_r = bk.symbols(f'v_tb_r_{name}'); v_tb_i = bk.symbols(f'v_tb_i_{name}')
    v_tc_r = bk.symbols(f'v_tc_r_{name}'); v_tc_i = bk.symbols(f'v_tc_i_{name}')

    p_soc = bk.symbols(f'p_soc_{name}')

    # --- dynamical states
    p_cf  = bk.symbols(f'p_cf_{name}')
    phi   = bk.symbols(f'phi_{name}')
    xi_q  = bk.symbols(f'xi_q_{name}')
    p_ef  = bk.symbols(f'p_ef_{name}')
    p_pfr = bk.symbols(f'p_pfr_{name}')
    De_ao_m = bk.symbols(f'De_ao_m_{name}'); De_bo_m = bk.symbols(f'De_bo_m_{name}')
    De_co_m = bk.symbols(f'De_co_m_{name}'); De_no_m = bk.symbols(f'De_no_m_{name}')
    omega = bk.symbols(f'omega_{name}')

    p_m = bk.symbols(f'p_m_{name}')

    # --- positive-sequence components of v_s, i_s in real form
    # v_pos = (1/3) * (v_sa + alpha*v_sb + alpha^2*v_sc),  alpha = e^{j2π/3}
    avb_r, avb_i = _cmul(_A_R,  _A_I,  v_sb_r, v_sb_i)
    a2vc_r, a2vc_i = _cmul(_A2_R, _A2_I, v_sc_r, v_sc_i)
    v_pos_r = (v_sa_r + avb_r + a2vc_r) / 3
    v_pos_i = (v_sa_i + avb_i + a2vc_i) / 3

    aib_r, aib_i = _cmul(_A_R,  _A_I,  i_sb_r, i_sb_i)
    a2ic_r, a2ic_i = _cmul(_A2_R, _A2_I, i_sc_r, i_sc_i)
    i_pos_r = (i_sa_r + aib_r + a2ic_r) / 3
    i_pos_i = (i_sa_i + aib_i + a2ic_i) / 3

    # negative-sequence (alpha and alpha^2 swapped)
    a2vb_r, a2vb_i = _cmul(_A2_R, _A2_I, v_sb_r, v_sb_i)
    avc_r,  avc_i  = _cmul(_A_R,  _A_I,  v_sc_r, v_sc_i)
    v_neg_r = (v_sa_r + a2vb_r + avc_r) / 3
    v_neg_i = (v_sa_i + a2vb_i + avc_i) / 3

    a2ib_r, a2ib_i = _cmul(_A2_R, _A2_I, i_sb_r, i_sb_i)
    aic_r,  aic_i  = _cmul(_A_R,  _A_I,  i_sc_r, i_sc_i)
    i_neg_r = (i_sa_r + a2ib_r + aic_r) / 3
    i_neg_i = (i_sa_i + a2ib_i + aic_i) / 3

    # zero-sequence
    v_zer_r = (v_sa_r + v_sb_r + v_sc_r) / 3
    v_zer_i = (v_sa_i + v_sb_i + v_sc_i) / 3
    i_zer_r = (i_sa_r + i_sb_r + i_sc_r) / 3
    i_zer_i = (i_sa_i + i_sb_i + i_sc_i) / 3

    # s_seq = 3 * v_seq * conj(i_seq)   (re, im in real form)
    p_pos = 3*(v_pos_r*i_pos_r + v_pos_i*i_pos_i)
    q_pos = 3*(v_pos_i*i_pos_r - v_pos_r*i_pos_i)
    p_neg = 3*(v_neg_r*i_neg_r + v_neg_i*i_neg_i)
    p_zer = 3*(v_zer_r*i_zer_r + v_zer_i*i_zer_i)
    # legacy file emits only re(s_pos), re(s_neg), re(s_zer), im(s_pos)

    p_r = K_agc*xi_freq
    epsilon_q = q_ref - q_pos/S_n

    # reactive PI with saturation + conditional-integration anti-windup
    De_q_nosat = K_qp*epsilon_q + K_qi*xi_q
    De_min = -U_n*0.05
    De_max =  U_n*0.05
    De_q  = bk.hard_limits(De_q_nosat, De_min, De_max)
    # anti-windup indicator: 1 inside the limits, 0 once saturated (this is a
    # step function, not a saturation, so it stays on Piecewise)
    K_qaw = bk.Piecewise((0, De_q_nosat < De_min),
                         (0, De_q_nosat > De_max),
                         (1, True))

    # phase EMFs (scalar trig - bk.cos / bk.sin)
    e_ao_r = (e_ao_m + De_ao_m)*bk.cos(phi_a + phi)
    e_ao_i = (e_ao_m + De_ao_m)*bk.sin(phi_a + phi)
    e_bo_r = (e_bo_m + De_bo_m)*bk.cos(phi_b + phi - 2/3*np.pi)
    e_bo_i = (e_bo_m + De_bo_m)*bk.sin(phi_b + phi - 2/3*np.pi)
    e_co_r = (e_co_m + De_co_m)*bk.cos(phi_c + phi - 4/3*np.pi)
    e_co_i = (e_co_m + De_co_m)*bk.sin(phi_c + phi - 4/3*np.pi)

    # differential equations
    dphi   = 2*np.pi*50*(omega - omega_coi) - K_delta*phi
    domega = 1/(2*H)*(p_m - p_ef - D*(omega - 1.0))
    dxi_q  = K_qaw*epsilon_q - (1 - K_qaw)*xi_q - 1e-6*xi_q
    dp_ef  = 1/T_e*(p_pos/S_n - p_ef)
    dp_cf  = 1/T_c*(p_c - p_cf)
    dp_pfr = 1/T_pfr*(1/Droop*(omega - omega_ref) - p_pfr)
    dDe_ao_m = 1/T_v*(v_ra + De_q - De_ao_m)
    dDe_bo_m = 1/T_v*(v_rb + De_q - De_bo_m)
    dDe_co_m = 1/T_v*(v_rc + De_q - De_co_m)
    dDe_no_m = 1/T_v*(v_rn - De_no_m)

    grid.dae['f'] += [dphi, domega, dxi_q, dp_ef, dp_cf, dp_pfr,
                      dDe_ao_m, dDe_bo_m, dDe_co_m, dDe_no_m]
    grid.dae['x'] += [phi, omega, xi_q, p_ef, p_cf, p_pfr,
                      De_ao_m, De_bo_m, De_co_m, De_no_m]

    # algebraic equations: p_m balance + terminal-voltage virtual impedance
    g_p_m = -p_m + p_cf + p_r - p_pfr + K_soc*p_soc

    # eq_v_t* = e_*o - i_s* * Z_v - v_t* = 0   in real form
    def _zi(R, X, ir, ii):
        return R*ir - X*ii, R*ii + X*ir

    re_zi_a, im_zi_a = _zi(R_v, X_v, i_sa_r, i_sa_i)
    re_zi_b, im_zi_b = _zi(R_v, X_v, i_sb_r, i_sb_i)
    re_zi_c, im_zi_c = _zi(R_v, X_v, i_sc_r, i_sc_i)

    eq_re_ta = e_ao_r - re_zi_a - v_ta_r
    eq_re_tb = e_bo_r - re_zi_b - v_tb_r
    eq_re_tc = e_co_r - re_zi_c - v_tc_r
    eq_im_ta = e_ao_i - im_zi_a - v_ta_i
    eq_im_tb = e_bo_i - im_zi_b - v_tb_i
    eq_im_tc = e_co_i - im_zi_c - v_tc_i

    grid.dae['g'] += [g_p_m,
                      eq_re_ta, eq_re_tb, eq_re_tc,
                      eq_im_ta, eq_im_tb, eq_im_tc]
    ys = [p_m, v_ta_r, v_tb_r, v_tc_r, v_ta_i, v_tb_i, v_tc_i]
    grid.dae['y_ini'] += ys
    grid.dae['y_run'] += ys

    V_1 = 400/np.sqrt(3)

    grid.dae['u_ini_dict'].update({
        f'e_ao_m_{name}': V_1, f'e_bo_m_{name}': V_1, f'e_co_m_{name}': V_1, f'e_no_m_{name}': 0.0,
        f'v_ra_{name}': 0.0, f'v_rb_{name}': 0.0, f'v_rc_{name}': 0.0, f'v_rn_{name}': 0.0,
        f'phi_a_{name}': 0.0, f'phi_b_{name}': 0.0, f'phi_c_{name}': 0.0, f'phi_n_{name}': 0.0,
        f'p_c_{name}': 0.0, f'omega_{name}_ref': 1.0, f'q_ref_{name}': 0.0,
        f'p_soc_{name}': 0.0, f'v_dc_{name}': 800.0,
    })
    grid.dae['u_run_dict'].update({
        f'e_ao_m_{name}': V_1, f'e_bo_m_{name}': V_1, f'e_co_m_{name}': V_1, f'e_no_m_{name}': 0.0,
        f'v_ra_{name}': 0.0, f'v_rb_{name}': 0.0, f'v_rc_{name}': 0.0, f'v_rn_{name}': 0.0,
        f'phi_a_{name}': 0.0, f'phi_b_{name}': 0.0, f'phi_c_{name}': 0.0, f'phi_n_{name}': 0.0,
        f'p_c_{name}': 0.0, f'omega_{name}_ref': 1.0, f'q_ref_{name}': 0.0,
        f'p_soc_{name}': 0.0, f'v_dc_{name}': 800.0,
    })

    grid.dae['params_dict'].update({
        f'X_v_{name}': data['X_v'], f'R_v_{name}': data['R_v'],
        f'S_n_{name}': data['S_n'], f'U_n_{name}': data['U_n'],
        f'T_e_{name}': data['T_e'], f'T_c_{name}': data['T_c'],
        f'T_v_{name}': data['T_v'], f'Droop_{name}': data['Droop'],
        f'H_{name}': data['H'], f'D_{name}': data['D'],
        f'K_agc_{name}': data['K_agc'], f'K_delta_{name}': data['K_delta'],
        f'K_qp_{name}': data['K_qp'], f'K_qi_{name}': data['K_qi'],
        f'K_soc_{name}': 1.0, f'T_pfr_{name}': data['T_pfr'],
    })

    # outputs
    grid.dae['h_dict'].update({
        f'p_pos_{name}': p_pos, f'p_neg_{name}': p_neg, f'p_zer_{name}': p_zer,
        f'q_pos_{name}': q_pos,
        str(e_ao_m): e_ao_m, str(e_bo_m): e_bo_m, str(e_co_m): e_co_m,
        str(v_ra): v_ra, str(v_rb): v_rb, str(v_rc): v_rc,
        str(p_c): p_c, str(omega_ref): omega_ref,
        str(phi): phi,
    })

    # the host ac_3ph_4w VSC seeds v_t* as inputs; the VSG turns them into states
    for s in (v_ta_r, v_tb_r, v_tc_r, v_ta_i, v_tb_i, v_tc_i):
        grid.dae['u_ini_dict'].pop(str(s), None)
        grid.dae['u_run_dict'].pop(str(s), None)

    grid.omega_coi_numerator   += S_n*omega
    grid.omega_coi_denominator += S_n
