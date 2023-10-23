# -*- coding: utf-8 -*-
"""
Created on Thu August 10 23:52:55 2022

@author: jmmauricio
"""

import numpy as np
import sympy as sym
from pydae.urisi.vsgs.vsg_lpf import vsg_lpf
from pydae.urisi.vsgs.gfpizv import gfpizv
from pydae.urisi.vsgs.gflpfzv import gflpfzv

def add_vsg(grid,item,name,bus_name):

    buses = grid.data['buses']
    buses_list = [bus['name'] for bus in buses]
    if item['type'] == 'vsg_lpf':
        vsg_lpf(grid,item,name,bus_name)
    if item['type'] == 'gfpizv':
        gfpizv(grid,item,name,bus_name)
    if item['type'] == 'gflpfzv':
        gflpfzv(grid,item,name,bus_name)

            