# -*- coding: utf-8 -*-
"""
Created on Thu August 10 23:52:55 2022

@author: jmmauricio
"""

import numpy as np
import sympy as sym
from pydae.urisi.pvs.pv_mpt_dcdc import pv_mpt_dcdc
from pydae.urisi.pvs.pv_mpt_dcac import pv_mpt_dcac
from pydae.urisi.pvs.pv_mpt import pv_mpt

def add_pv(grid,item,name,bus_name):

    if item['type'] == 'pv_mpt_dcac':
        pv_mpt_dcac(grid,item,name,bus_name)
    if item['type'] == 'pv_mpt_dcdc':
        pv_mpt_dcdc(grid,item,name,bus_name)
    if item['type'] == 'pv_mpt':
        pv_mpt(grid,item,name,bus_name)
                