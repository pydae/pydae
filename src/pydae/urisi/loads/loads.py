# -*- coding: utf-8 -*-
"""
Created on Thu August 10 23:52:55 2022

@author: jmmauricio
"""

import numpy as np
import sympy as sym
from pydae.urisi.loads.load_ac import load_ac
from pydae.urisi.loads.load_ac_3w import load_ac_3w
from pydae.urisi.loads.load_dc import load_dc


def add_loads(grid):

    buses = grid.data['buses']
    buses_list = [bus['name'] for bus in buses]

    self = grid


    # loads
    for load in self.data['loads']:
        if load['type'] == '3P+N' and  load["model"] == 'ZIP':
            load_ac(grid,load)
        if load['type'] == '3P' and  load["model"] == 'ZIP':
            load_ac_3w(grid,load)
        if load['type'] == 'DC' and  load["model"] == 'ZIP':
            load_dc(grid,load)