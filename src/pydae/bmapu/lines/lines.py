# -*- coding: utf-8 -*-
"""
Created on Thu August 10 23:52:55 2022

@author: jmmauricio
"""


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
        print('U_base',U_base,'Z_base',Z_base)
        Y_base = 1.0/Z_base
        Bs = line['Bs_km']*line['km']/Y_base  # in pu of the model base
        bs = Bs
        model.set_value(f'bs_{line_name}',bs)
        print(bs)
    G =  R/(R**2+X**2)
    B = -X/(R**2+X**2)
    model.set_value(f"g_{line_name}",G)
    model.set_value(f"b_{line_name}",B)