# -*- coding: utf-8 -*-
"""
Created on Thu August 10 23:52:55 2022

@author: jmmauricio
"""

import numpy as np
import sympy as sym
from pydae.bmapu.vsc_ctrls.ctrl_pq import ctrl_pq
from pydae.bmapu.vsc_ctrls.leon_vsg_ll import leon_vsg_ll

def add_ctrl(dae,name,bus_name,data_dict):
    
    if data_dict['ctrl']['type'] == 'pq':
        ctrl_pq(dae,name,bus_name,data_dict)
    if data_dict['ctrl']['type'] == 'leon_vsg_ll':
        leon_vsg_ll(dae,name,bus_name,data_dict)        
