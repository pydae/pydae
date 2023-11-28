# -*- coding: utf-8 -*-
"""
Created on Thu August 10 23:52:55 2022

@author: jmmauricio
"""

import numpy as np
import sympy as sym
from pydae.urisi.sources.vdc_src import vdc_src
from pydae.urisi.sources.ac3ph4w_ideal import ac3ph4w_ideal
from pydae.urisi.sources.ac3ph3w_ideal import ac3ph3w_ideal
def add_sources(grid):

    buses = grid.data['buses']
    buses_list = [bus['name'] for bus in buses]

    for item in grid.data['sources']:
        if item['type'] == 'vdc_src':
            vdc_src(grid,item)
        if item['type'] == 'ac3ph4w_ideal':
            ac3ph4w_ideal(grid,item)
        if item['type'] == 'ac3ph3w_ideal':
            ac3ph3w_ideal(grid,item)