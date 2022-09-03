# -*- coding: utf-8 -*-
"""
Created on Thu August 10 23:52:55 2022

@author: jmmauricio
"""

import numpy as np
import sympy as sym
from pydae.bmapu.psss.pss_kundur import pss_kundur

def add_pss(dae,syn_data,name):
    
    if syn_data['pss']['type'] == 'pss_kundur':
        pss_kundur(dae,syn_data,name)
