# -*- coding: utf-8 -*-
"""
Created on Thu August 10 23:52:55 2022

@author: jmmauricio
"""

import numpy as np
import sympy as sym
from pydae.urisi.vscs.ac3ph4wgfpi2 import ac3ph4wgfpi2
from pydae.urisi.genapes.ac3ph4w_ideal import ac3ph4w_ideal
from pydae.urisi.genapes.ac3ph3w_ideal import ac3ph3w_ideal
from pydae.urisi.genapes.dc_ideal import dc_ideal

def add_genapes(grid):

    buses = grid.data['buses']
    buses_list = [bus['name'] for bus in buses]

    for item in grid.data['genapes']:
        if item['type'] == 'ac3ph4w_ideal':
            ac3ph4w_ideal(grid,item)
        if item['type'] == 'ac3ph3w_ideal':
            ac3ph3w_ideal(grid,item)       
        if item['type'] == 'dc_ideal':
            dc_ideal(grid,item)       