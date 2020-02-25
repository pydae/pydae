#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  5 20:43:46 2018

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
        string += f'{r:0.4f}  '.rjust(10, ' ') 
        string += f'{i:0.4f}j'.rjust(10, ' ') 
        string += f'{freqs[it]:0.4f}'.rjust(10, ' ') 
        string += f'{zetas[it]:0.4f}'.rjust(10, ' ') 

        string += '\n'
        #'\t{i:0.4f}\t{freqs[it]:0.3f}\t{zetas[it]:0.4f}\n'
        
    return string
    
    
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
    
    

    