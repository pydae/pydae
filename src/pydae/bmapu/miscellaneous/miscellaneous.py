# -*- coding: utf-8 -*-
"""
Created on Thu August 10 23:52:55 2022

@author: jmmauricio
"""

import numpy as np
import sympy as sym
from pydae.bmapu.miscellaneous.fault import add_fault


def add_miscellaneous(grid):

    if 'faults' in grid.data:
        for item in grid.data['faults']:
            add_fault(grid,item)
