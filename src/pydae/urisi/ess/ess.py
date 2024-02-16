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
from pydae.urisi.ess.bess_dcdc_gf import bess_dcdc_gf

def add_ess(grid,data):

    bus_name = data['bus']
    name = bus_name

    if data['type'] == 'bess_dcdc':
        bess_dcdc(grid,data,name,bus_name)
    if data['type'] == 'bess_dcac':
        bess_dcac(grid,data,name,bus_name)       
    if data['type'] == 'bess_r':
        bess_r(grid,data,name,bus_name)    
    if data['type'] == 'bess_dcdc_gf':
        bess_dcdc_gf(grid,data,name,bus_name)