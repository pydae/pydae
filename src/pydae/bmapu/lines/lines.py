# -*- coding: utf-8 -*-
"""
Created on Thu August 10 23:52:55 2022

@author: jmmauricio
"""

import numpy as np

def change_line(model,bus_j,bus_k, *args,**kwagrs):
    line = kwagrs
    S_base = model.get_value('S_base')
    
    line_name = f"{bus_j}_{bus_k}"
    if 'X_pu' in line:
        if 'S_mva' in line: S_line = 1e6*line['S_mva']
        R = line['R_pu']*S_base/S_line  # in pu of the model base
        X = line['X_pu']*S_base/S_line  # in pu of the model base
    if 'X' in line:
        U_base = model.get_value(f'U_{bus_j}_n') 
        Z_base = U_base**2/S_base
        R = line['R']/Z_base  # in pu of the model base
        X = line['X']/Z_base  # in pu of the model base
    if 'X_km' in line:
        U_base = model.get_value(f'U_{bus_j}_n')
        Z_base = U_base**2/S_base
        R = line['R_km']*line['km']/Z_base  # in pu of the model base
        X = line['X_km']*line['km']/Z_base  # in pu of the model base
    if 'Bs_km' in line:
        U_base = model.get_value(f'U_{bus_j}_n')
        Z_base = U_base**2/S_base
        Y_base = 1.0/Z_base
        Bs = line['Bs_km']*line['km']/Y_base  # in pu of the model base
        bs = Bs
        model.set_value(f'bs_{line_name}',bs)

    G =  R/(R**2+X**2)
    B = -X/(R**2+X**2)
    model.set_value(f"g_{line_name}",G)
    model.set_value(f"b_{line_name}",B)

def get_line_current(model,bus_j,bus_k, units='A'):
    V_j_m,theta_j = model.get_mvalue([f'V_{bus_j}',f'theta_{bus_j}'])
    V_k_m,theta_k = model.get_mvalue([f'V_{bus_k}',f'theta_{bus_k}'])
    name = f'b_{bus_j}_{bus_k}'
    if name in model.params_list:
        B_j_k,G_j_k = model.get_mvalue([f'b_{bus_j}_{bus_k}',f'g_{bus_j}_{bus_k}'])
        Bs_j_k = model.get_value(f'bs_{bus_j}_{bus_k}')
    elif f'b_{bus_k}_{bus_j}' in model.params_list:
        B_j_k,G_j_k = model.get_mvalue([f'b_{bus_k}_{bus_j}',f'g_{bus_k}_{bus_j}'])
        Bs_j_k = model.get_value(f'bs_{bus_k}_{bus_j}')

    
    Y_j_k = G_j_k + 1j*B_j_k 
    V_j = V_j_m*np.exp(1j*theta_j)
    V_k = V_k_m*np.exp(1j*theta_k)
    I_j_k_pu = (V_j - V_k)*Y_j_k + 1j*V_j*Bs_j_k/2

    if units == 'A':
        S_b = model.get_value('S_base')
        U_b = model.get_value(f'U_{bus_j}_n')
        I_b = S_b/(np.sqrt(3)*U_b)
        I_j_k = I_j_k_pu*I_b

    return I_j_k