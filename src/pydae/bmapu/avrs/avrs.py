# -*- coding: utf-8 -*-
"""
Created on Thu August 10 23:52:55 2022

@author: jmmauricio
"""

import numpy as np
import sympy as sym
from pydae.bmapu.avrs.sexs import sexs

def add_avr(dae,syn_data,name):
    
    if syn_data['avr']['type'] == 'sexs':
        sexs(dae,syn_data,name)
