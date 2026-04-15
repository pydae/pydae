# -*- coding: utf-8 -*-
"""
Created on Thu August 10 23:52:55 2022

@author: jmmauricio
"""

import numpy as np
import sympy as sym



def load_dc(grid,data):

    buses = grid.data['buses']

    self = grid

    name = data['bus']
    v_p,v_n = sym.symbols(f'V_{name}_0_r,V_{name}_1_r', real=True)

    i_p,i_n = sym.symbols(f'i_load_{name}_p_r,i_load_{name}_n_r', real=True)
    p = sym.Symbol(f'p_load_{name}', real=True)

    
    self.dae['g'] += [i_p*(v_p - v_n) - p]

    self.dae['y_ini'] += [i_p]
    self.dae['y_run'] += [i_p]

    i_n = -i_p

    idx_r,idx_i = self.node2idx(name,'a')
    self.dae['g'] [idx_r] += i_p
    self.dae['g'] [idx_i] += 0.0

    idx_r,idx_i = self.node2idx(name,'b')
    self.dae['g'] [idx_r] += i_n
    self.dae['g'] [idx_i] += 0.0

    self.dae['u_ini_dict'].update({f'p_load_{name}':data['kW']*1e3})
    self.dae['u_run_dict'].update({f'p_load_{name}':data['kW']*1e3}) 