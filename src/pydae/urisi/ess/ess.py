# -*- coding: utf-8 -*-
"""
Created on Thu August 10 23:52:55 2022

@author: jmmauricio
"""

import numpy as np
import sympy as sym

from pydae.urisi.ess.bess_dcdc import bess_dcdc
from pydae.urisi.ess.bess_dcac import bess_dcac
from pydae.urisi.ess.bess_r import bess_r

def add_ess(grid,item,name,bus_name):

    buses = grid.data['buses']
    buses_list = [bus['name'] for bus in buses]

    if item['type'] == 'bess_dcdc':
        bess_dcdc(grid,item,name,bus_name)
    if item['type'] == 'bess_dcac':
        bess_dcac(grid,item,name,bus_name)       
    if item['type'] == 'bess_r':
        bess_r(grid,item,name,bus_name)    