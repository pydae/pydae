# -*- coding: utf-8 -*-
"""
Created on Thu August 10 23:52:55 2022

@author: jmmauricio
"""

import numpy as np
import sympy as sym
from pydae.urisi.vscs.ac3ph4wgfpi2 import ac3ph4wgfpi2
from pydae.urisi.vscs.ac_3ph_4w_l import ac_3ph_4w_l

def add_vscs(grid):

    buses = grid.data['buses']
    buses_list = [bus['name'] for bus in buses]

    for item in grid.data['vscs']:
        if item['type'] == 'ac_3ph_4w_l':
            ac_3ph_4w_l(grid,item)
        if item['type'] == 'ac3ph4wgfpi2':
            ac3ph4wgfpi2(grid,item)