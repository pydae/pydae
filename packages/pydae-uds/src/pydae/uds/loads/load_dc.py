# -*- coding: utf-8 -*-
"""
Created on Thu August 10 23:52:55 2022

@author: jmmauricio
"""

import numpy as np


def load_dc(grid,data):

    self = grid
    bk = grid.backend

    name = data['bus']
    v_p = bk.symbols(f'V_{name}_0_r')
    v_n = bk.symbols(f'V_{name}_1_r')

    i_p = bk.symbols(f'i_load_{name}_p_r')
    p   = bk.symbols(f'p_load_{name}')

    
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