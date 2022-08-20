# -*- coding: utf-8 -*-
"""
Created on Thu August 10 23:52:55 2022

@author: jmmauricio
"""

import numpy as np
import sympy as sym

from pydae.bmapu.vsgs.leon_gvsg import leon_gvsg
from pydae.bmapu.vsgs.leon_vsg_ll import leon_vsg_ll
from pydae.bmapu.vsgs.uvsg_mid import uvsg_mid
from pydae.bmapu.vsgs.uvsg_high import uvsg_high
from pydae.bmapu.vsgs.olives_vsg import olives_vsg


def add_vsgs(grid):

    for item in grid.data['vsgs']:

        data_dict = item

        bus_name = item['bus']                
        name = create_name(grid, data_dict)
        item['name'] = name
                            
        #if item['type'] == 'wo_vsg': p_W, q_var = wo_vsg(grid,name,bus_name,data_dict)
        #if item['type'] == 'pi_vsg': p_W, q_var = pi_vsg(grid,name,bus_name,data_dict)
        if item['type'] == 'uvsg': p_W, q_var = uvsg_mid(grid,name,bus_name,data_dict)
        if item['type'] == 'uvsg_high': p_W, q_var = uvsg_high(grid,name,bus_name,data_dict)
        if item['type'] == 'vsg_co': p_W, q_var = olives_vsg(grid,name,bus_name,data_dict)
        if item['type'] == 'leon_gvsg': p_W, q_var = leon_gvsg(grid,name,bus_name,data_dict)
        if item['type'] == 'vsg_ll': p_W, q_var = leon_vsg_ll(grid,name,bus_name,data_dict)      

        # grid power injection
        idx_bus = grid.buses_list.index(data_dict['bus']) # get the number of the bus where the syn is connected
        if not 'idx_powers' in grid.buses[idx_bus]: grid.buses[idx_bus].update({'idx_powers':0})
        grid.buses[idx_bus]['idx_powers'] += 1

        S_base = sym.Symbol('S_base', real = True)
        grid.dae['g'][idx_bus*2]   += -p_W/S_base
        grid.dae['g'][idx_bus*2+1] += -q_var/S_base


def create_name(grid,data_dict):

    bus_name = data_dict['bus']
        
    if 'name' in data_dict:
        name = data_dict['name']
    else:
        name = bus_name
        
    for gen_id in range(100):
        if name not in grid.generators_id_list:
            grid.generators_id_list += [name]
            break
        else:
            name = name + f'_{gen_id}'

    return name
        

if __name__ == "__main__":

    import pydae.build_cffi as db
    from pydae.bmapu import bmapu_builder
    from pydae.bmapu.vsgs.vsgs import add_vsgs
    import pydae.build_cffi as db
    import sympy as sym

    data = {
        "sys":{"name":"sys2buses","S_base":100e6, "K_p_agc":0.01,"K_i_agc":0.01},       
        "buses":[{"name":"1", "P_W":0.0,"Q_var":0.0,"U_kV":20.0},
                {"name":"2", "P_W":0.0,"Q_var":0.0,"U_kV":20.0}],
        "lines":[{"bus_j":"1", "bus_k":"2", "X_pu":0.15,"R_pu":0.0, "S_mva":900.0}],
        "vsgs":[
            {"bus":"1","type":"vsg_ll",'S_n':10e6,'F_n':50,'K_delta':0.0,
            'R_v':0.01,'X_v':0.1,'K_p':1.0,'K_i':0.1,'K_g':0.0,'K_q':20.0,
            'T_q':0.1,'K_p_v':1e-6,'K_i_v':1e-6}],
        "genapes":[{"bus":"2","S_n":100e6,"F_n":50.0,"R_v":0.0,"X_v":0.1,"K_delta":0.001,"K_alpha":1.0}],
        }

    grid = bmapu_builder.bmapu(data)

    add_vsgs(grid)