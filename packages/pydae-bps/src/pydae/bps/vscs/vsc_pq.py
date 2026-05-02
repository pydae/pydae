# -*- coding: utf-8 -*-
"""
Created on Thu August 10 23:52:55 2022

@author: jmmauricio
"""

import numpy as np

def vsc_pq(grid,name,bus_name,data_dict):
    """
    # auxiliar
    
    """

    backend = grid.backend

    p_in = backend.symbols(f"p_in_{name}")
    Dp_r = backend.symbols(f"Dp_r_{name}")
    Dq_r = backend.symbols(f"Dq_r_{name}")

        
    # dynamic states
    p_out = backend.symbols(f"p_out_{name}")
    q_out = backend.symbols(f"q_out_{name}")

    # algebraic states

    # parameters
    S_n = backend.symbols(f"S_n_{name}")
    
    # auxiliar
    
               
    # dynamic equations            

    # algebraic equations   
    p_out_sat = backend.Piecewise((0.0,p_in + Dp_r<0.0),(p_in,p_in + Dp_r>p_in),(p_in + Dp_r,True))         
    q_out_max = (1**2 - p_out_sat**2)**0.5
    q_out_sat = backend.Piecewise((-q_out_max,Dq_r<-q_out_max),(q_out_max,Dq_r>q_out_max),(Dq_r,True))     
    g_p_out = -p_out + p_out_sat
    g_q_out = -q_out + q_out_sat

    # dae 
    grid.dae['f'] += []
    grid.dae['x'] += []
    grid.dae['g'] += [g_p_out,g_q_out]
    grid.dae['y_ini'] += [p_out,q_out]  
    grid.dae['y_run'] += [p_out,q_out]  
    
    grid.dae['u_ini_dict'].update({f'{p_in}':data_dict['p_in']})
    grid.dae['u_run_dict'].update({f'{p_in}':data_dict['p_in']})
    grid.dae['u_ini_dict'].update({f'{Dp_r}':0.0})
    grid.dae['u_run_dict'].update({f'{Dp_r}':0.0})
    grid.dae['u_ini_dict'].update({f'{Dq_r}':0.0})
    grid.dae['u_run_dict'].update({f'{Dq_r}':0.0})            
               
    # outputs
    grid.dae['h_dict'].update({f"{p_in}":p_in})
    grid.dae['h_dict'].update({f"{Dp_r}":Dp_r})
    grid.dae['h_dict'].update({f"{Dq_r}":Dq_r})
    
    # parameters            
    grid.dae['params_dict'].update({f"{S_n}":data_dict['S_n']}) 

    p_W   = p_out*S_n
    q_var = q_out*S_n

    return p_W,q_var