# -*- coding: utf-8 -*-
"""
Created on Thu August 10 23:52:55 2022

@author: jmmauricio
"""

import numpy as np
import sympy as sym
from pydae.urisi.vsc_ctrls.ctrl_3ph_4w_pq import ctrl_3ph_4w_pq

def add_vsc_ctrl(grid,item,name,bus_name):

    buses = grid.data['buses']
    buses_list = [bus['name'] for bus in buses]
    if item['type'] == 'ctrl_3ph_4w_pq':
        ctrl_3ph_4w_pq(grid,item,name,bus_name)            