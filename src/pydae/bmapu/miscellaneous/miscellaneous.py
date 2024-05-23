# -*- coding: utf-8 -*-
"""
Created on Thu August 10 23:52:55 2022

@author: jmmauricio
"""

import numpy as np
import sympy as sym
from pydae.bmapu.miscellaneous.fault import add_fault
from pydae.bmapu.miscellaneous.pll import add_pll


def add_miscellaneous(grid):

    if 'faults' in grid.data:
        for item in grid.data['faults']:
            add_fault(grid,item)

    if 'plls' in grid.data:
        for item in grid.data['plls']:
            if 'type' in item:
                if item['type'] == 'pll_2wo_3ll':
                    add_pll_2wo_3ll(grid,item)
            else:
                add_pll(grid,item)

