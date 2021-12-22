# -*- coding: utf-8 -*-
"""
Created on Wed Dec 15 23:38:58 2021

@author: jmmau
"""

import pytest
import numpy as np
import sympy as sym
import pydae.build_cffi as db
 
 
    
def test_pendulum_builder():
    params_dict = {'L':5.21,'G':9.81,'M':10.0,'K_d':1e-6}


    u_ini_dict = {'theta':np.deg2rad(5.0)}  # for the initialization problem
    u_run_dict = {'f_x':10}  # for the running problem (here initialization and running problem are the same)
    
    #u_ini_dict = {'theta':10}  # for the initialization problem
    u_run_dict = {'f_x':10}  # for the running problem (here initialization and running problem are the same)
    
    
    
    x_list = ['pos_x','pos_y','v_x','v_y']    # [inductor current, PI integrator]
    y_ini_list = ['lam','f_x'] # for the initialization problem
    y_run_list = ['lam','theta'] # for the running problem (here initialization and running problem are the same)
    
    sys_vars = {'params':params_dict,
                'u_list':u_run_dict,
                'x_list':x_list,
                'y_list':y_run_list}
    
    exec(db.sym_gen_str())  # exec to generate the required symbolic varables and constants
    
    dpos_x = v_x
    dpos_y = v_y
    dv_x = (-2*pos_x*lam + f_x - K_d*v_x)/M
    dv_y = (-M*G - 2*pos_y*lam - K_d*v_y)/M   
    
    g_1 = pos_x**2 + pos_y**2 - L**2 -lam*1e-6
    g_2 = -theta + sym.atan2(pos_x,-pos_y)

    sys = {'name':'pendulum',
           'params_dict':params_dict,
           'f_list':[dpos_x,dpos_y,dv_x,dv_y],
           'g_list':[g_1,g_2],
           'x_list':x_list,
           'y_ini_list':y_ini_list,
           'y_run_list':y_run_list,
           'u_run_dict':u_run_dict,
           'u_ini_dict':u_ini_dict,
           'h_dict':{'g_1':g_1,'PE':M*G*pos_y,'KE':0.5*M*(v_x**2+v_y**2),'theta':theta}}
    
    sys = db.build(sys)


def test_pendulum_ini():
    
    import pendulum
    
    pend = pendulum.pendulum_class()
    M = 30.0
    L = 5.21
    pend.ini({'f_x':0,'M':M,'L':L,'theta':np.deg2rad(-5)},-5)

    f_x = pend.get_value('f_x')
    
    del pendulum 
    print(f_x)
    assert np.abs(f_x-(-25.7479))<0.01


if __name__ == "__main__":
    
    #test_pendulum_builder()
    test_pendulum_ini()