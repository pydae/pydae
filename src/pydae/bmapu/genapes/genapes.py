# -*- coding: utf-8 -*-
"""
Created on Thu August 10 23:52:55 2022

@author: jmmauricio
"""

import numpy as np
import sympy as sym
from pydae.bmapu.genapes.genape_inf import genape_inf

def add_genapes(grid):

    buses = grid.data['buses']
    buses_list = [bus['name'] for bus in buses]

    for item in grid.data['genapes']:

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
                            

        p_W, q_var = genape_inf(grid,name,bus_name,data_dict)

        # grid power injection
        idx_bus = buses_list.index(bus_name) # get the number of the bus where the syn is connected
        if not 'idx_powers' in buses[idx_bus]: buses[idx_bus].update({'idx_powers':0})
        buses[idx_bus]['idx_powers'] += 1

        S_base = sym.Symbol('S_base', real = True)
        grid.dae['g'][idx_bus*2]   += -p_W/S_base
        grid.dae['g'][idx_bus*2+1] += -q_var/S_base
        