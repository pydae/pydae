# -*- coding: utf-8 -*-
"""
Created on Thu August 10 23:52:55 2022

@author: jmmauricio
"""

import numpy as np
import sympy as sym


def pv_dq_vrt(grid, name, bus_name, data_dict):
    '''
    Simplified VSC model in Stationary Real/Imaginary (x-y) coordinates.
    Core dynamics are extremely simple, but internal variables (DC link, 
    filter voltages, MPPT) are analytically reconstructed as optional outputs.
    '''
    sin = sym.sin
    cos = sym.cos

    ## Common Bus Variables
    V_s = sym.Symbol(f"V_{bus_name}", real=True)
    theta_s = sym.Symbol(f"theta_{bus_name}", real=True)

    ## Inputs (u)
    p_s_ppc = sym.Symbol(f"p_s_ppc_{name}", real=True)
    q_s_ppc = sym.Symbol(f"q_s_ppc_{name}", real=True)
    lvrt_ext = sym.Symbol(f"lvrt_ext_{name}", real=True)
    i_sa_ref = sym.Symbol(f"i_sa_ref_{name}", real=True)  
    i_sr_ref = sym.Symbol(f"i_sr_ref_{name}", real=True)  

    ## Dynamic States (x)
    lvrt_ext_ramp = sym.Symbol(f"lvrt_ext_ramp_{name}", real=True) 

    ## Algebraic States (y)
    p_s = sym.Symbol(f"p_s_{name}", real=True)
    q_s = sym.Symbol(f"q_s_{name}", real=True)

    ## Parameters
    S_n = sym.Symbol(f"S_n_{name}", real=True)
    I_max = sym.Symbol(f"I_max_{name}", real=True)
    T_lvrt = sym.Symbol(f"T_lvrt_{name}", real=True)
    Epsilon = sym.Symbol(f"Epsilon_{name}", real=True)

    ## 1. Dynamic Equation for the LVRT Ramp
    f_lvrt_ext_ramp = (lvrt_ext - lvrt_ext_ramp) / T_lvrt

    ## 2. Grid Voltages in Real/Imaginary (x-y) coordinates
    v_sr = V_s * cos(theta_s) # Equivalent to v_x
    v_si = V_s * sin(theta_s) # Equivalent to v_y
    v_sq_mag = v_sr**2 + v_si**2 + 1e-8
    v_m = sym.sqrt(v_sq_mag)

    ## 3. Calculate Required Currents (Stationary Frame)
    i_sr_pq = (p_s_ppc * v_sr + q_s_ppc * v_si) / v_sq_mag
    i_si_pq = (p_s_ppc * v_si - q_s_ppc * v_sr) / v_sq_mag

    i_sr_ar = (i_sa_ref * v_sr + i_sr_ref * v_si) / v_m
    i_si_ar = (i_sa_ref * v_si - i_sr_ref * v_sr) / v_m

    ## 4. Blend Modes & Soft Saturation
    i_sr_nosat = (1.0 - lvrt_ext_ramp) * i_sr_pq + lvrt_ext_ramp * i_sr_ar
    i_si_nosat = (1.0 - lvrt_ext_ramp) * i_si_pq + lvrt_ext_ramp * i_si_ar

    i_mod = sym.sqrt(i_sr_nosat**2 + i_si_nosat**2 + Epsilon)
    i_mod_sat = 0.5 * (i_mod + I_max - sym.sqrt((i_mod - I_max)**2 + Epsilon))
    
    ratio = i_mod_sat / i_mod
    i_sr = i_sr_nosat * ratio
    i_si = i_si_nosat * ratio

    ## 5. Calculate Final Injected Powers
    p_s_expr = v_sr * i_sr + v_si * i_si
    q_s_expr = v_si * i_sr - v_sr * i_si
    
    g_p_s = p_s_expr - p_s  
    g_q_s = q_s_expr - q_s 

    ### Core DAE Formulation
    grid.dae['f'] += [f_lvrt_ext_ramp]
    grid.dae['x'] += [lvrt_ext_ramp]
    grid.dae['g'] += [g_p_s, g_q_s]
    grid.dae['y_ini'] += [p_s, q_s]  
    grid.dae['y_run'] += [p_s, q_s]  

    ### Core Initialization
    grid.dae['u_ini_dict'].update({f'lvrt_ext_{name}': 0.0, 
                                   f'p_s_ppc_{name}': 1.0, 
                                   f'q_s_ppc_{name}': 0.0, 
                                   f'i_sa_ref_{name}': 0.0, 
                                   f'i_sr_ref_{name}': 0.0})
    grid.dae['u_run_dict'].update({f'lvrt_ext_{name}': 0.0, 
                                   f'p_s_ppc_{name}': 1.0, 
                                   f'q_s_ppc_{name}': 0.0,
                                   f'i_sa_ref_{name}': 0.0, 
                                   f'i_sr_ref_{name}': 0.0})
    
    grid.dae['params_dict'].update({
        f'S_n_{name}': data_dict['S_n'],
        f'I_max_{name}': data_dict.get('I_max', 1.2),
        f'T_lvrt_{name}': data_dict.get('T_lvrt', 0.02),
        f'Epsilon_{name}': data_dict.get('Epsilon', 1e-8) # Default to a sharper 1e-8
    })

    grid.dae['xy_0_dict'].update({f"lvrt_ext_ramp_{name}": 0.0, f"p_s_{name}": 1.0, f"q_s_{name}": 0.0})

    # Output Signals
    grid.dae['h_dict'].update({
        f"p_s_{name}": p_s, f"q_s_{name}": q_s
    })
    
    # =====================================================================
    # OPTIONAL INTERNAL VARIABLES RECONSTRUCTION (Analytical)
    # =====================================================================
    if data_dict.get('monitor', False):
        # 1. AC Filter Voltage Reconstruction
        R_s = sym.Symbol(f"R_s_{name}", real=True)
        X_s = sym.Symbol(f"X_s_{name}", real=True)
        U_n = sym.Symbol(f"U_n_{name}", real=True)
        V_dc_b = U_n * np.sqrt(2)
        
        v_tr = v_sr + R_s * i_sr - X_s * i_si
        v_ti = v_si + R_s * i_si + X_s * i_sr
        v_t_m = sym.sqrt(v_tr**2 + v_ti**2)

        # 2. PV Array Parameters
        K_vt, K_it = sym.symbols(f"K_vt_{name}, K_it_{name}", real=True)
        V_oc, V_mp, I_sc, I_mp = sym.symbols(f"V_oc_{name}, V_mp_{name}, I_sc_{name}, I_mp_{name}", real=True)
        temp_deg, irrad = sym.symbols(f"temp_deg_{name}, irrad_{name}", real=True)
        N_pv_s, N_pv_p = sym.symbols(f"N_pv_s_{name}, N_pv_p_{name}", real=True)
        
        T_stc_deg = 25.0
        V_oc_t = N_pv_s * V_oc * (1 + K_vt/100.0 * (temp_deg - T_stc_deg))
        V_mp_t = N_pv_s * V_mp * (1 + K_vt/100.0 * (temp_deg - T_stc_deg))
        I_sc_t = N_pv_p * I_sc * (1 + K_it/100.0 * (temp_deg - T_stc_deg))
        I_mp_t = N_pv_p * I_mp * (1 + K_it/100.0 * (temp_deg - T_stc_deg))
        I_mp_i = I_mp_t * irrad / 1000.0

        v_1, i_1 = V_mp_t, I_mp_i
        v_2, i_2 = V_oc_t, 0

        # 3. Explicit Analytical Solution for DC Voltage
        # The original algebraic equation was quadratic. We solve it explicitly using (-B + sqrt(B^2 - 4AC)) / 2A
        m_pv = (v_1 - v_2) / (i_1 - i_2)
        # Equation: v_dc_v^2 - (v_1 - i_1*m_pv)*v_dc_v - (p_s*S_n*m_pv) = 0
        B_coef = -(v_1 - i_1 * m_pv)
        C_coef = -(p_s * S_n * m_pv)
        
        v_dc_v_expr = (-B_coef + sym.sqrt(B_coef**2 - 4 * C_coef)) / 2.0
        i_pv_expr = (p_s * S_n) / v_dc_v_expr
        v_dc_expr = v_dc_v_expr / V_dc_b
        m_ref_expr = v_t_m / v_dc_expr

        # Register Parameters for Output Reconstruction
        grid.dae['u_ini_dict'].update({f'{str(irrad)}': 1000.0, f'{str(temp_deg)}': 25.0})
        grid.dae['u_run_dict'].update({f'{str(irrad)}': 1000.0, f'{str(temp_deg)}': 25.0})
        
        grid.dae['params_dict'].update({
            f"U_n_{name}": data_dict['U_n'], f"R_s_{name}": data_dict['R_s'], f"X_s_{name}": data_dict['X_s'],
            f"I_sc_{name}": data_dict['I_sc'], f"I_mp_{name}": data_dict['I_mp'], f"V_mp_{name}": data_dict['V_mp'], 
            f"V_oc_{name}": data_dict['V_oc'], f"N_pv_s_{name}": data_dict['N_pv_s'], f"N_pv_p_{name}": data_dict['N_pv_p'], 
            f"K_vt_{name}": data_dict['K_vt'], f"K_it_{name}": data_dict['K_it']
        })

        # Output Signals
        grid.dae['h_dict'].update({
            f"v_m_{name}": v_m, f"i_mod_{name}": i_mod,
            f"i_sr_{name}": i_sr, f"i_si_{name}": i_si, f"v_tr_{name}": v_tr, f"v_ti_{name}": v_ti,
            f"v_dc_v_{name}": v_dc_v_expr, f"v_dc_{name}": v_dc_expr, f"i_pv_{name}": i_pv_expr,
            f"m_ref_{name}": m_ref_expr, f"lvrt_ext_ramp_{name}": lvrt_ext_ramp, f"i_mod_sat_{name}": i_mod_sat
        })

    p_W   = p_s * S_n
    q_var = q_s * S_n

    return p_W, q_var

def sym2model():
    v_sd = sym.Symbol(f"v_sd", real=True)
    v_sq = sym.Symbol(f"v_sq", real=True)
    p_s_ref = sym.Symbol(f"p_s_ref", real=True)
    q_s_ref = sym.Symbol(f"q_s_ref", real=True)
    i_sa_ref = sym.Symbol(f"i_sa_ref", real=True)
    i_sr_ref = sym.Symbol(f"i_sr_ref", real=True)
    i_sd_pq_ref,i_sq_pq_ref = sym.symbols(f'i_sd_pq_ref,i_sq_pq_ref', real=True)
    i_sd_ar_ref,i_sq_ar_ref = sym.symbols(f'i_sd_ar_ref,i_sq_ar_ref', real=True)


    g_i_sd_pq_ref  = i_sd_pq_ref*v_sd + i_sq_pq_ref*v_sq - p_s_ref  
    g_i_sq_pq_ref  =-i_sq_pq_ref*v_sd + i_sd_pq_ref*v_sq - q_s_ref 

    sol = sym.solve([g_i_sd_pq_ref,g_i_sq_pq_ref],[i_sd_pq_ref,i_sq_pq_ref])

    for item in sol:
        print(f"{item} = {sol[item]}")

    v_m = sym.sqrt(v_sd**2 + v_sq**2)
    g_i_sd_ar_ref  = i_sd_ar_ref*v_sd/v_m + i_sq_ar_ref*v_sq/v_m - i_sa_ref  
    g_i_sq_ar_ref  =-i_sq_ar_ref*v_sd/v_m + i_sd_ar_ref*v_sq/v_m - i_sr_ref 

    sol = sym.solve([g_i_sd_ar_ref,g_i_sq_ar_ref],[i_sd_ar_ref,i_sq_ar_ref])

    for item in sol:
        print(f"{item} = {sol[item]}")

    # coupling filter:
    v_tr,v_ti = sym.symbols(f"v_tr,v_ti", real=True)
    i_sr,i_si = sym.symbols(f"i_sr,i_si", real=True)
    v_sr,v_si = sym.symbols(f"v_sr,v_si", real=True)
    R_s,X_s =   sym.symbols(f"R_s,X_s", real=True)

    g_i_si = v_ti - R_s*i_si + X_s*i_sr - v_si  
    g_i_sr = v_tr - R_s*i_sr - X_s*i_si - v_sr 
    sol = sym.solve([g_i_sr,g_i_si],[i_sr,i_si])

    for item in sol:
        print(f"{item} = {sol[item]}")

    # simplified PV module




#sym2model()
if __name__ == "__main__":

    from pydae.bmapu import bmapu_builder
    
    data = {
    "system":{"name":"test_model","S_base":100e6, "K_p_agc":0.01,"K_i_agc":0.01,"K_xif":0.0},       
    "buses":[{"name":"1", "P_W":0.0,"Q_var":0.0,"U_kV":20.0},
             {"name":"2", "P_W":0.0,"Q_var":0.0,"U_kV":20.0}
            ],
    "lines":[{"bus_j":"1", "bus_k":"2", "X_pu":0.05,"R_pu":0.0,"Bs_pu":0.0,"S_mva":10000.0}],
    "syns":[{
      "bus": "2",
      "type": "milano2ord",
            "S_n": 200e6,
      "F_n": 50.0,
      "X1d": 0.3,   
      "X1q": 0.55, 
      "R_a": 0.01, 
      "H": 5.0, 
      "D": 0.0,
      "K_delta": 0.01,
      "K_sec": 0.01
    }],
    #"genapes":[{"bus":"2","S_n":10000e6,"F_n":50.0,"X_v":0.001,"R_v":0.0,"K_delta":0.001,"K_alpha":1e-6}]
    }


    grid = bmapu_builder.bmapu(data)
    #grid.checker()
    grid.uz_jacs = True
    grid.verbose = False
    grid.construct('test_model')
    grid.construct('temp_m2')
    bld = Builder(grid.sys_dict, target='ctypes')
    bld.build()
 
    # bus_name = "1"
    # data_dict = {"bus":bus_name,"type":"pv_pq",
    #              "S_n":1e6,"U_n":400.0,"F_n":50.0,
    #              "X_s":0.1,"R_s":0.01,"monitor":True,
    #              "I_sc":8,"V_oc":42.1,"I_mp":3.56,"V_mp":33.7,
    #              "K_vt":-0.160,"K_it":0.065,
    #              "N_pv_s":25,"N_pv_p":250}

    # p_W,q_var = pv_dq(grid,'1','1',data_dict)

    # # grid power injection
    # idx_bus = [bus['name'] for bus in grid.data['buses']].index(bus_name) # get the number of the bus where the syn is connected

    # S_base = sym.Symbol('S_base', real = True)
    # grid.dae['g'][idx_bus*2]   += -p_W/S_base
    # grid.dae['g'][idx_bus*2+1] += -q_var/S_base
    
    # grid.compile()   

    # import test_model

    # xy_0 = {
    # "V_1": 1.0,
    # "theta_1": 0.0,
    # "V_2": 1.0,
    # "theta_2": 0.0,
    # "omega_coi": 1.0,
    # "omega_2": 1.0,
    # "v_dc_v_1":800
    # }
    # model = test_model.model()
    # model.ini({'p_s_ppc_1':0.9,'q_s_ppc_1':0.2,'irrad_1':100,'temp_deg_1':75,
    #            'v_ref_2':1.0},xy_0)

    # #model.report_y()
    # model.report_z()
    # #model.report_u()
