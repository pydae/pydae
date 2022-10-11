# -*- coding: utf-8 -*-
"""
Created on Thu August 10 23:52:55 2022

@author: jmmauricio
"""

import numpy as np
import sympy as sym
from pydae.bmapu.psss.pss_kundur import pss_kundur
from pydae.bmapu.psss.pss_pss2 import pss_pss2

def add_pss(dae,syn_data,name):
    
    if syn_data['pss']['type'] == 'pss_kundur':
        pss_kundur(dae,syn_data,name)
    elif syn_data['pss']['type'] == 'pss2':
        pss_pss2(dae,syn_data,name)
    else:
        print(f"PSS type {syn_data['pss']['type']} not found.")
