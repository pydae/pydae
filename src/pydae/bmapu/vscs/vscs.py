# -*- coding: utf-8 -*-
"""
Created on Thu August 10 23:52:55 2022

@author: jmmauricio
"""

import numpy as np
import sympy as sym
from pydae.bmapu.vscs.vsc_pq import vsc_pq
from pydae.bmapu.vscs.vsc_l  import vsc_l
from pydae.bmapu.vscs.vsc_lcl_uc  import vsc_lcl_uc

from pydae.bmapu.vsc_ctrls.vsc_ctrls import add_ctrl
from pydae.bmapu.vsc_models.vsc_models import add_model

def add_vscs(grid):

    buses = grid.data['buses']
    buses_list = [bus['name'] for bus in buses]

    for item in grid.data['vscs']:

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

        if item['type'] == 'vsc_pq':                    
            p_W, q_var = vsc_pq(grid,name,bus_name,data_dict)
        if item['type'] == 'vsc_l':                    
            p_W, q_var = vsc_l(grid,name,bus_name,data_dict)
        if item['type'] == 'vsc_lcl_uc':                    
            p_W, q_var = vsc_lcl_uc(grid,name,bus_name,data_dict)




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
        
        if 'ctrl' in item:
            add_ctrl(grid,name,bus_name,data_dict)
            grid.dae['u_ini_dict'].pop(str(m))
            grid.dae['u_run_dict'].pop(str(m))
            grid.dae['u_ini_dict'].pop(str(theta_t))
            grid.dae['u_run_dict'].pop(str(theta_t))
            grid.dae['xy_0_dict'].update({str(m):0.8})
            grid.dae['xy_0_dict'].update({str(theta_t):0.0})


        if 'models' in item:
            add_model(grid,name,bus_name,data_dict)
            grid.dae['u_ini_dict'].pop(str(v_f))
            grid.dae['u_run_dict'].pop(str(v_f))
            grid.dae['xy_0_dict'].update({str(v_f):1.5})