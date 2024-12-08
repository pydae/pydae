# -*- coding: utf-8 -*-
"""
Created on Thu August 10 23:52:55 2022

@author: jmmauricio
"""

import numpy as np
import sympy as sym

from pydae.bmapu.pvs.pv_1 import pv_1
from pydae.bmapu.pvs.pv_dq import pv_dq
from pydae.bmapu.pvs.pv_dq_d import pv_dq_d
from pydae.bmapu.pvs.pv_dq_ss import pv_dq_ss

def add_pvs(grid):

    buses = grid.data['buses']
    buses_list = [bus['name'] for bus in buses]

    for item in grid.data['pvs']:

        data_dict = item
        
        bus_name = item['bus']
        
        if 'name' in item:
            name = item['name']
        else:
            name = bus_name
            
        for gen_id in range(100):
            if name not in grid.generators_id_list:
                grid.generators_id_list += [name]
                break
            else:
                name = name + f'_{gen_id}'
                
        item['name'] = name

        if item['type'] == 'pv_1':                    
            p_W, q_var = pv_1(grid,name,bus_name,data_dict)

        if item['type'] == 'pv_dq':                    
            p_W, q_var = pv_dq(grid,name,bus_name,data_dict)

        if item['type'] == 'pv_dq_d':                    
            p_W, q_var = pv_dq_d(grid,name,bus_name,data_dict)

        if item['type'] == 'pv_dq_ss':                    
            p_W, q_var = pv_dq_ss(grid,name,bus_name,data_dict)


        # grid power injection
        idx_bus = buses_list.index(bus_name) # get the number of the bus where the syn is connected
        if not 'idx_powers' in buses[idx_bus]: buses[idx_bus].update({'idx_powers':0})
        buses[idx_bus]['idx_powers'] += 1

        S_base = sym.Symbol('S_base', real = True)
        grid.dae['g'][idx_bus*2]   += -p_W/S_base
        grid.dae['g'][idx_bus*2+1] += -q_var/S_base


        ## Add controlers and models
        m = f'm_{name}'
        theta_t = f'theta_t_{name}'
        p_s_ref = f'p_s_ref_{name}'
        
