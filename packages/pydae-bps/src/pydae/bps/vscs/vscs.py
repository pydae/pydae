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
from pydae.bmapu.vscs.bess_pq  import bess_pq
from pydae.bmapu.vscs.bess_pq_ss import bess_pq_ss

from pydae.bmapu.vsc_ctrls.vsc_ctrls import add_ctrl
from pydae.bmapu.vsc_models.vsc_models import add_model

from pydae.bmapu.vscs.vsc_acgf_dcp import vsc_acgf_dcp
from pydae.bmapu.vscs.vsc_acq_dcgf import vsc_acq_dcgf
from pydae.bmapu.vscs.vsc_pq_qv_pfr import vsc_pq_qv_pfr

from pydae.bmapu.miscellaneous.pll import add_pll




def add_vscs(grid):

    buses = grid.data['buses']
    buses_list = [bus['name'] for bus in buses]

    for item in grid.data['vscs']:

        data_dict = item
        
        if 'bus' in data_dict:   # single bus VSC
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
            if item['type'] == 'bess_pq':                    
                p_W, q_var = bess_pq(grid,name,bus_name,data_dict)
            if item['type'] == 'pq_qv_pfr':                    
                p_W, q_var = vsc_pq_qv_pfr(grid,name,bus_name,data_dict)
                if 'pll' in item:
                    add_pll(grid,item['pll'])              
            if item['type'] == 'bess_pq_ss':                    
                p_W, q_var = bess_pq_ss(grid,name,bus_name,data_dict)

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

        if 'bus_ac' in data_dict:  # more than one bus VSC

            bus_dc = item['bus_dc']
            bus_ac = item['bus_ac']
            
            if 'name' in item:
                name = item['name']
            else:
                name = f'{bus_dc}_{bus_ac}'
                    
            item['name'] = name

            if item['type'] == 'vsc_acgf_dcp':    # AC side Grid-Former               
                p_W_j, q_var_j,p_W_k, q_var_k = vsc_acgf_dcp(grid,name,bus_dc,bus_ac,data_dict)
            if item['type'] == 'vsc_acq_dcgf':    # DC side Grid-Former               
                p_W_j, q_var_j,p_W_k, q_var_k = vsc_acq_dcgf(grid,name,bus_dc,bus_ac,data_dict)

            # grid power injection
            idx_bus_dc = buses_list.index(bus_dc) # get the number of the bus j
            S_base = sym.Symbol('S_base', real = True)
            grid.dae['g'][idx_bus_dc*2]   += -p_W_j/S_base
            grid.dae['g'][idx_bus_dc*2+1] += -q_var_j/S_base

            idx_bus_ac = buses_list.index(bus_ac) # get the number of the bus k
            S_base = sym.Symbol('S_base', real = True)
            grid.dae['g'][idx_bus_ac*2]   += -p_W_k/S_base
            grid.dae['g'][idx_bus_ac*2+1] += -q_var_k/S_base