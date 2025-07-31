# -*- coding: utf-8 -*-
"""
Created on Thu August 10 23:52:55 2022

@author: jmmauricio
"""

import numpy as np
import sympy as sym

def add_banks(grid):

    buses = grid.data['buses']
    buses_list = [bus['name'] for bus in buses]

    for item in grid.data['banks']:

        data_dict = item
        
        bus_name = item['bus']
        
        if 'name' in item:
            name = item['name']
        else:
            name = bus_name
            

        q_cap,B_cap,V,S_n_cap,S_base = sym.symbols(f'q_cap_{name},B_cap_{name},V_{bus_name},S_n_cap_{name},S_base', real=True)
        B_cap_ref,T_cap = sym.symbols(f'B_cap_ref_{name},T_cap_{name}', real=True)


        dB_cap = 1/T_cap*(B_cap_ref - B_cap)

        q_cap = B_cap*(V**2) # pu-m

        grid.dae['f'] += [dB_cap]
        grid.dae['x'] += [ B_cap]

        grid.dae['params_dict'].update({str(T_cap):item['T_cap']})
        grid.dae['params_dict'].update({str(S_n_cap):item['S_mva']*1e6})

        grid.dae['u_ini_dict'].update({str(B_cap_ref):item['B']})
        grid.dae['u_run_dict'].update({str(B_cap_ref):item['B']})



        # grid power injection
        idx_bus = buses_list.index(bus_name) # get the number of the bus where the syn is connected
        if not 'idx_powers' in buses[idx_bus]: buses[idx_bus].update({'idx_powers':0})
        buses[idx_bus]['idx_powers'] += 1

        S_base = sym.Symbol('S_base', real = True)
        grid.dae['g'][idx_bus*2]   += -0/S_base
        grid.dae['g'][idx_bus*2+1] += -q_cap/S_base*S_n_cap

def test_build():
    import sympy as sym
    from pydae.bmapu import bmapu_builder
    import matplotlib.pyplot as plt
    grid = bmapu_builder.bmapu('banks.hjson')
    grid.build('temp')

def test_run():
    import temp

    model = temp.model()
    model.ini({},'xy_0.json')


    model.report_x()
    model.report_y()
    model.report_z()

if __name__=='__main__':

    test_build()
    test_run()

        