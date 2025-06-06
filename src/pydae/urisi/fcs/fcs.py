# -*- coding: utf-8 -*-
"""
Created on Thu August 10 23:52:55 2022

@author: jmmauricio
"""

import numpy as np
import sympy as sym

from pydae.urisi.fcs.sofc_dcdc_gf import sofc_dcdc_gf
from pydae.urisi.fcs.pemfc_dcdc_gf import pemfc_dcdc_gf
from pydae.urisi.fcs.sofc_dcdcac_gf import sofc_dcdcac_gf

def add_fcs(grid,data):

    bus_name = data['bus']
    name = bus_name

    if data['type'] == 'sofc_dcdc_gf':
        sofc_dcdc_gf(grid,data,name,bus_name)

    if data['type'] == 'sofc_dcdcac_gf':
        sofc_dcdcac_gf(grid,data,name,bus_name)

    if data['type'] == 'pemfc_dcdc_gf':
        pemfc_dcdc_gf(grid,data,name,bus_name)