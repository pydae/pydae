# -*- coding: utf-8 -*-
"""
Created on Thu August 10 23:52:55 2022

@author: jmmauricio
"""

import numpy as np
import sympy as sym

def add_fault(grid,data):
    """
    # auxiliar
    
    """

    bus_name = data['bus']
    name = bus_name

    fault_b = sym.Symbol(f'fault_b_{name}', real=True)
    fault_g = sym.Symbol(f'fault_g_{name}', real=True)
    fault_g_ref= sym.Symbol(f'fault_g_ref_{name}', real=True)
    RampDown = sym.Symbol(f'RampDown_{name}', real=True)
    RampUp = sym.Symbol(f'RampUp_{name}', real=True)
    K_fault = sym.Symbol(f'K_fault_{name}', real=True)

    V = sym.Symbol(f'V_{name}', real=True)


    epsilon_g = fault_g_ref - fault_g
    dfault_g_nosat = K_fault*(fault_g_ref - fault_g)
    dfault_g_sat = sym.Piecewise((RampDown,dfault_g_nosat<RampDown),
                                (RampUp,dfault_g_nosat>RampUp),
                                (dfault_g_nosat,True))
    grid.dae['f'] += [dfault_g_sat]
    grid.dae['x'] += [fault_g]

    p_pu = fault_g*V**2
    q_pu = fault_b*V**2

    grid.dae['u_ini_dict'].update({f'{fault_b}':0.0})
    grid.dae['u_run_dict'].update({f'{fault_b}':0.0})
    grid.dae['u_ini_dict'].update({f'{fault_g_ref}':0.0})
    grid.dae['u_run_dict'].update({f'{fault_g_ref}':0.0})

    grid.dae['params_dict'].update({f'{str(RampDown)}':-20000})
    grid.dae['params_dict'].update({f'{str(RampUp)}':10000})
    grid.dae['params_dict'].update({f'{str(K_fault)}':10000})

    # grid power injection
    buses = grid.data['buses']
    buses_list = [bus['name'] for bus in buses]

    idx_bus = buses_list.index(bus_name) # get the number of the bus where the syn is connected
    if not 'idx_powers' in buses[idx_bus]: buses[idx_bus].update({'idx_powers':0})
    buses[idx_bus]['idx_powers'] += 1

    S_base = sym.Symbol('S_base', real = True)
    grid.dae['g'][idx_bus*2]   += -p_pu
    grid.dae['g'][idx_bus*2+1] += -q_pu


def test():
    import numpy as np
    import sympy as sym
    import hjson
    from pydae.bmapu.bmapu_builder import bmapu
    import pydae.build_cffi as db
    import pytest
    import matplotlib.pyplot as plt

    grid = bmapu('fault.hjson')
    grid.checker()
    grid.uz_jacs = True
    grid.build('temp')


    import temp

    model = temp.model()
    model.Dt = 0.001
    model.decimation = 1

    v_ref_1 = 1.0
    model.ini({'p_m_1':0.5,'v_ref_1':v_ref_1,'K_a_1':200,
            'RampDown_1':-45e3,'RampUp_1':30e3, 'K_fault_1':1e2},'xy_0.json')

    # assert model.get_value('V_1') == pytest.approx(v_ref_1, rel=0.001)

    model.run(1.0,{})
    model.run(1.1,{'fault_g_ref_1':20000})
    model.run(1.4,{})
    model.run(5,{'fault_g_ref_1':0})

    # model.run(1.11,{'fault_g_ref_1':0})
    # model.run(2,{'fault_g_ref_1':0})

    model.post()
    # # assert model.get_value('q_A2') == pytest.approx(-q_ref, rel=0.05)

    # model.ini({'p_m_1':0.5,'v_ref_1':1.0},'xy_0.json')
    # model.run(1.0,{})
    # model.run(15.0,{'v_ref_1':1.05})
    # model.post()


    fig,axes = plt.subplots()
    axes.plot(model.Time,model.get_values('omega_1'))
    #axes.plot(model.Time,model.Y)
    axes.grid()
    fig.savefig('fault.svg')

if __name__ == '__main__':

    #development()
    test()