
import numpy as np


def ac3ph3w_ideal(grid, vsc_data):
    '''
    VSC with 3 phase and 3 wire working in open loop as a grid former.
    Dual-backend (SymPy / CasADi) via grid.backend; nodal algebra in real form.
    '''
    bk = grid.backend
    params_dict = grid.dae['params_dict']
    g_list      = grid.dae['g']
    y_ini_list  = grid.dae['y_ini']
    u_ini_dict  = grid.dae['u_ini_dict']
    y_run_list  = grid.dae['y_run']
    u_run_dict  = grid.dae['u_run_dict']
    h_dict      = grid.dae['h_dict']

    name = vsc_data['bus']

    # inputs
    e_ao_m = bk.symbols(f'e_ao_m_{name}')
    e_bo_m = bk.symbols(f'e_bo_m_{name}')
    e_co_m = bk.symbols(f'e_co_m_{name}')
    v_pu   = bk.symbols(f'v_pu_{name}')
    phi    = bk.symbols(f'phi_{name}')

    # bus node voltages (rectangular)
    v_sa_r = bk.symbols(f'V_{name}_0_r'); v_sa_i = bk.symbols(f'V_{name}_0_i')
    v_sb_r = bk.symbols(f'V_{name}_1_r'); v_sb_i = bk.symbols(f'V_{name}_1_i')
    v_sc_r = bk.symbols(f'V_{name}_2_r'); v_sc_i = bk.symbols(f'V_{name}_2_i')

    # parameters
    R_s = bk.symbols(f'R_{name}_s')
    X_s = bk.symbols(f'X_{name}_s')

    # algebraic source currents (rectangular)
    i_sa_r = bk.symbols(f'i_vsc_{name}_a_r'); i_sa_i = bk.symbols(f'i_vsc_{name}_a_i')
    i_sb_r = bk.symbols(f'i_vsc_{name}_b_r'); i_sb_i = bk.symbols(f'i_vsc_{name}_b_i')
    i_sc_r = bk.symbols(f'i_vsc_{name}_c_r'); i_sc_i = bk.symbols(f'i_vsc_{name}_c_i')

    # EMF (real form): e_ph = v_pu*|e_pho_m|*exp(j*(phi - shift))
    e_a_r = (v_pu*e_ao_m)*bk.cos(phi)
    e_a_i = (v_pu*e_ao_m)*bk.sin(phi)
    e_b_r = (v_pu*e_bo_m)*bk.cos(phi - 2/3*np.pi)
    e_b_i = (v_pu*e_bo_m)*bk.sin(phi - 2/3*np.pi)
    e_c_r = (v_pu*e_co_m)*bk.cos(phi - 4/3*np.pi)
    e_c_i = (v_pu*e_co_m)*bk.sin(phi - 4/3*np.pi)

    # series-impedance drop in real form: Z*i = (R+jX)(i_r+j*i_i)
    #   re(Z*i) = R*i_r - X*i_i
    #   im(Z*i) = R*i_i + X*i_r
    eq_a_r = e_a_r - (R_s*i_sa_r - X_s*i_sa_i) - v_sa_r
    eq_a_i = e_a_i - (R_s*i_sa_i + X_s*i_sa_r) - v_sa_i
    eq_b_r = e_b_r - (R_s*i_sb_r - X_s*i_sb_i) - v_sb_r
    eq_b_i = e_b_i - (R_s*i_sb_i + X_s*i_sb_r) - v_sb_i
    eq_c_r = e_c_r - (R_s*i_sc_r - X_s*i_sc_i) - v_sc_r
    eq_c_i = e_c_i - (R_s*i_sc_i + X_s*i_sc_r) - v_sc_i

    g_list += [eq_a_r, eq_b_r, eq_c_r, eq_a_i, eq_b_i, eq_c_i]

    y_ini_list += [i_sa_r, i_sb_r, i_sc_r, i_sa_i, i_sb_i, i_sc_i]
    y_run_list += [i_sa_r, i_sb_r, i_sc_r, i_sa_i, i_sb_i, i_sc_i]

    # current injection into the nodal equations + magnitude monitors
    for ph, i_r, i_i in [('a', i_sa_r, i_sa_i), ('b', i_sb_r, i_sb_i), ('c', i_sc_r, i_sc_i)]:
        idx_r, idx_i = grid.node2idx(name, ph)
        grid.dae['g'][idx_r] += -i_r
        grid.dae['g'][idx_i] += -i_i
        h_dict[f'i_vsc_{name}_{ph}_m'] = (i_r**2 + i_i**2)**0.5

    V_1 = vsc_data['U_n']/np.sqrt(3)
    u_ini_dict.update({f'e_ao_m_{name}': V_1, f'e_bo_m_{name}': V_1, f'e_co_m_{name}': V_1})
    u_run_dict.update({f'e_ao_m_{name}': V_1, f'e_bo_m_{name}': V_1, f'e_co_m_{name}': V_1})

    v_pu_n = vsc_data.get('v_pu', 1.0)
    u_ini_dict.update({f'v_pu_{name}': v_pu_n})
    u_run_dict.update({f'v_pu_{name}': v_pu_n})
    u_ini_dict.update({f'phi_{name}': 0.0})
    u_run_dict.update({f'phi_{name}': 0.0})

    params_dict.update({f'X_{name}_s': vsc_data['X'], f'R_{name}_s': vsc_data['R']})

    # per-phase apparent power (real form): s = v*conj(i)
    #   p = v_r*i_r + v_i*i_i
    #   q = v_i*i_r - v_r*i_i
    p_total = (v_sa_r*i_sa_r + v_sa_i*i_sa_i +
               v_sb_r*i_sb_r + v_sb_i*i_sb_i +
               v_sc_r*i_sc_r + v_sc_i*i_sc_i)
    q_total = (v_sa_i*i_sa_r - v_sa_r*i_sa_i +
               v_sb_i*i_sb_r - v_sb_r*i_sb_i +
               v_sc_i*i_sc_r - v_sc_r*i_sc_i)
    h_dict[f'p_{name}'] = p_total
    h_dict[f'q_{name}'] = q_total

    grid.dae['xy_0_dict'].update({'omega': 1.0})

    # COI contribution: ideal source carries infinite inertia
    grid.omega_coi_numerator   += 1.0e9
    grid.omega_coi_denominator += 1.0e9
