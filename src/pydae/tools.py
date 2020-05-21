#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2020

@author: jmmauricio
"""

import numpy as np

def eval_A(system):
    
    Fx = system.struct[0].Fx
    Fy = system.struct[0].Fy
    Gx = system.struct[0].Gx
    Gy = system.struct[0].Gy
    
    A = Fx - Fy @ np.linalg.solve(Gy,Gx)
    
    system.A = A
    
    return A

def eval_A_ini(system):
    
    Fx = system.struct[0].Fx_ini
    Fy = system.struct[0].Fy_ini
    Gx = system.struct[0].Gx_ini
    Gy = system.struct[0].Gy_ini
    
    A = Fx - Fy @ np.linalg.solve(Gy,Gx)
    
    system.A = A

    return A


def damp_report(system):
    eig,eigv = np.linalg.eig(system.A)
    omegas = eig.imag
    sigmas = eig.real

    freqs = np.abs(omegas/(2*np.pi))
    zetas = -sigmas/np.sqrt(sigmas**2+omegas**2)
    
    string = ''
    
    string += f' Real'.ljust(10, ' ') 
    string += f' Imag'.ljust(10, ' ') 
    string += f' Freq.'.ljust(10, ' ') 
    string += f' Damp'.ljust(10, ' ') 
    string += '\n'

    N_x = len(eig)
    for it in range(N_x):
        r = eig[it].real
        i = eig[it].imag
        string += f'{r:5.4f}  '.rjust(10, ' ') 
        string += f'{i:5.4f}j'.rjust(10, ' ') 
        string += f'{freqs[it]:5.4f}'.rjust(10, ' ') 
        string += f'{zetas[it]:5.4f}'.rjust(10, ' ') 

        string += '\n'
        #'\t{i:0.4f}\t{freqs[it]:0.3f}\t{zetas[it]:0.4f}\n'
        
    return string
    
def get_v(syst,bus_name,v_type='rms_phph'):
    X = syst.X
    Y = syst.Y

    v_d_name = f'v_{bus_name}_d'
    v_q_name = f'v_{bus_name}_q'

    if v_d_name in syst.y_list:
        v_d = Y[-1,syst.y_list.index(v_d_name)]
        v_q = Y[-1,syst.y_list.index(v_q_name)]
    elif  v_d_name in syst.x_list:
        v_d = X[-1,syst.x_list.index(v_d_name)]
        v_q = X[-1,syst.x_list.index(v_q_name)]
    elif v_d_name in syst.struct[0].dtype.names:
        v_d = syst.struct[0][v_d_name]
        v_q = syst.struct[0][v_q_name]
    else:
        print(f'Voltage in bus {bus_name} is not a state.')
        v_d = 0.0
        v_q = 0.0

    v_m = 0.0
    if v_type == 'rms_phph':
        v_m = np.abs(v_d+1j*v_q)*np.sqrt(3/2)
        return v_m

    if v_type == 'dq_cplx':
        return v_q+1j*v_d


def get_i(syst,bus_name,i_type='rms_phph'):
    X = syst.X
    Y = syst.Y

    i_d_name = f'i_{bus_name}_d'
    i_q_name = f'i_{bus_name}_q'
    if i_d_name in syst.y_list:
        i_d = Y[-1,syst.y_list.index(i_d_name)]
        i_q = Y[-1,syst.y_list.index(i_q_name)]
    elif  i_d_name in syst.x_list:
        i_d = X[-1,syst.x_list.index(i_d_name)]
        i_q = X[-1,syst.x_list.index(i_q_name)]
    elif i_d_name in syst.struct[0].dtype.names:
        i_d = syst.struct[0][i_d_name]
        i_q = syst.struct[0][i_q_name]
    else:
        print(f'Voltage in bus {bus_name} is not a state.')
        i_d = 0.0
        i_q = 0.0

    i_m = 0.0
    if i_type == 'rms_phph':
        i_m = np.abs(i_d+1j*i_q)*np.sqrt(3/2)
        return i_m

    if i_type == 'dq_cplx':
        return i_q+1j*i_d

def get_s(syst,bus_name,s_type='cplx'):
    v_dq = get_v(syst,bus_name,v_type='dq_cplx')
    i_dq = get_i(syst,bus_name,i_type='dq_cplx')

    if s_type == 'cplx':
        s = 3/2*v_dq*np.conj(i_dq)
        return s


if __name__ == "__main__":
    
    class sys_class():
        
        def __init__(self): 
     
            self.A = np.array([[ 0.00000000e+00,  3.14159265e+02,  0.00000000e+00,
                 0.00000000e+00,  0.00000000e+00],
               [-4.43027144e-01, -1.42857143e-01, -1.44867645e-01,
                 1.69999669e-01,  0.00000000e+00],
               [-1.80455968e-01,  0.00000000e+00, -6.64265905e-01,
                -2.31113959e-03, -2.50000000e+01],
               [ 1.32845727e+00,  0.00000000e+00,  1.35913375e-02,
                -2.58565604e+00,  0.00000000e+00],
               [-3.90435895e-01,  0.00000000e+00,  2.69427382e+00,
                 4.81098835e-01, -2.00000000e+01]])
            
    sys = sys_class()
    print(damp_report(sys))
    
    

    