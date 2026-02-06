# -*- coding: utf-8 -*-
"""
Created on Thu August 10 23:52:55 2022

@author: jmmauricio
"""

import numpy as np
import sympy as sym

from pydae.bmapu.lines.line_dtr import add_line_dtr


def add_lines(self):
    sys = self.system

    for line in self.lines:

        if not 'dtr' in line: line.update({'dtr':False})

        if line['dtr']:
            add_line_dtr(self, line)
        else:
            add_line(self, line)


def add_line(self, line):
        
    sys = self.system

    bus_j = line['bus_j']
    bus_k = line['bus_k']

    idx_j = self.buses_list.index(bus_j)
    idx_k = self.buses_list.index(bus_k)    

    self.A[self.it,idx_j] = 1
    self.A[self.it,idx_k] =-1   
    self.A[self.it+1,idx_j] = 1
    self.A[self.it+2,idx_k] = 1   
    
    line_name = f"{bus_j}_{bus_k}"
    g_jk = sym.Symbol(f"g_{line_name}", real=True) 
    b_jk = sym.Symbol(f"b_{line_name}", real=True) 
    bs_jk = sym.Symbol(f"bs_{line_name}", real=True) 
    self.G_primitive[self.it,self.it] = g_jk
    self.B_primitive[self.it,self.it] = b_jk
    self.B_primitive[self.it+1,self.it+1] = bs_jk/2
    self.B_primitive[self.it+2,self.it+2] = bs_jk/2

    if not 'thermal' in line:
        line.update({'thermal':False})      

    if 'X_pu' in line:
        if 'S_mva' in line: S_line = 1e6*line['S_mva']
        R = line['R_pu']*sys['S_base']/S_line  # in pu of the system base
        X = line['X_pu']*sys['S_base']/S_line  # in pu of the system base
        G =  R/(R**2+X**2)
        B = -X/(R**2+X**2)
        self.dae['params_dict'].update({f"g_{line_name}":G})
        self.dae['params_dict'].update({f'b_{line_name}':B})

    if 'X' in line:
        bus_idx = buses_list.index(line['bus_j'])
        U_base = self.buses[bus_idx]['U_kV']*1000
        Z_base = U_base**2/sys['S_base']
        R = line['R']/Z_base  # in pu of the system base
        X = line['X']/Z_base  # in pu of the system base
        G =  R/(R**2+X**2)
        B = -X/(R**2+X**2)
        self.dae['params_dict'].update({f"g_{line_name}":G})
        self.dae['params_dict'].update({f'b_{line_name}':B})

    if 'X_km' in line:
        bus_idx = self.buses_list.index(line['bus_j'])
        U_base = self.buses[bus_idx]['U_kV']*1000
        Z_base = U_base**2/sys['S_base']
        R = line['R_km']*line['km']/Z_base  # in pu of the system base

        X = line['X_km']*line['km']/Z_base  # in pu of the system base
        G =  R/(R**2+X**2)
        B = -X/(R**2+X**2)
        self.dae['params_dict'].update({f"g_{line_name}":G})
        self.dae['params_dict'].update({f'b_{line_name}':B})        

    self.dae['params_dict'].update({f'bs_{line_name}':0.0})
    if 'Bs_pu' in line:
        if 'S_mva' in line: S_line = 1e6*line['S_mva']
        Bs = line['Bs_pu']*S_line/sys['S_base']  # in pu of the system base
        bs = Bs
        self.dae['params_dict'][f'bs_{line_name}'] = bs

    if 'Bs_km' in line:
        bus_idx = self.buses_list.index(line['bus_j'])
        U_base = self.buses[bus_idx]['U_kV']*1000
        Z_base = U_base**2/sys['S_base']
        Y_base = 1.0/Z_base
        Bs = line['Bs_km']*line['km']/Y_base # in pu of the system base
        bs = Bs 
        self.dae['params_dict'][f'bs_{line_name}'] = bs
        
    self.it += 3


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


def test_line_pu_build():
    
    from pydae.bmapu import bmapu_builder
    from pydae.build_v2 import build_mkl,build_numba

    data = {
        "system":{"name":"temp","S_base":100e6, "K_p_agc":0.01,"K_i_agc":0.01, "K_xif":0.01},       
        "buses":[{"name":"1", "P_W":0.0,"Q_var":0.0,"U_kV":20.0},
                 {"name":"2", "P_W":0.0,"Q_var":0.0,"U_kV":20.0}],
        "lines":[{"bus_j":"1", "bus_k":"2", "X_pu":0.15,"R_pu":0.0,"Bs_pu":0.5, "S_mva":900.0, "monitor":True}],
        "sources":[{"bus":"1","type":"vsource","V_mag_pu":1.0,"V_ang_rad":0.0,"K_delta":0.1}]
        }

    grid = bmapu_builder.bmapu(data)
    grid.uz_jacs = False
    grid.verbose = True
    grid.construct(f'temp')

    
    build_numba(grid.sys_dict)

def test_line_pu_ini():
    
    import temp

    model = temp.model()
    model.ini({'P_2':0e6, "K_xif":0.01},'xy_0.json')
    print(f"V_1 = {model.get_value('V_1'):2.2f}, V_2 = {model.get_value('V_2'):2.2f}")

    model.report_y()

def test_line_km_build():
    
    from pydae.bmapu import bmapu_builder
    from pydae.build_v2 import build_mkl,build_numba

    R_km = 0.0268  # (Ω/km)
    X_km = 0.2766  # (Ω/km)
    B_km = 4.59e-6 # (℧-1/Km)

    N_parallel = 44.0

    Lenght_km = 22000.0/N_parallel

    data = {
        "system":{"name":"temp","S_base":100e6, "K_p_agc":0.01,"K_i_agc":0.01, "K_xif":0.01},       
        "buses":[{"name":"1", "P_W":0.0,"Q_var":0.0,"U_kV":400.0},
                 {"name":"2", "P_W":5000e6,"Q_var":0.0,"U_kV":400.0}],
        "lines":[{"bus_j":"1", "bus_k":"2", 'X_km':X_km/44, 'R_km':R_km/44, 'Bs_km':B_km*44, "km":Lenght_km, "monitor":True}],
        "sources":[{"bus":"1","type":"vsource","V_mag_pu":1.0,"V_ang_rad":0.0,"K_delta":0.1}]
        }

    grid = bmapu_builder.bmapu(data)
    grid.uz_jacs = False
    grid.verbose = True
    grid.construct(f'temp')

    print(grid.sys_dict)

   
    build_numba(grid.sys_dict)

def test_line_km_ini():
    
    import temp

    model = temp.model()
    model.ini({'P_2':5000e6,'Q_2':0, "K_p_agc":0.0,"K_i_agc":0.0, "K_xif":0.01},'xy_0.json')
    print(f"V_1 = {model.get_value('V_1'):2.2f}, V_2 = {model.get_value('V_2'):2.2f}")

    print("report_y:")
    model.report_y()
    print("-----------------------------------------------------------------------")
    print("report_z:")
    model.report_z()

    U_1 = 400e3
    U_2 = 400e3 
    Bs = 4.59e-6*22000
    Q = U_1**2*Bs/2 + U_2**2*Bs/2
    print(f"Q = {Q/1e6:2.2f} Mvar")


if __name__ == "__main__":

    #test_line_pu_build()
    #test_line_pu_ini()

    test_line_km_build()
    test_line_km_ini()


