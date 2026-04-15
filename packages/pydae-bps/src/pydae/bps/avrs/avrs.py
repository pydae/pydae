# -*- coding: utf-8 -*-
"""
Created on Thu August 10 23:52:55 2022

@author: jmmauricio
"""

import numpy as np
import sympy as sym
from pydae.bmapu.avrs.sexs import sexs
from pydae.bmapu.avrs.sexs import sexsq
from pydae.bmapu.avrs.kundur import kundur
from pydae.bmapu.avrs.ntsst4 import ntsst4
from pydae.bmapu.avrs.ntsst1 import ntsst1
from pydae.bmapu.avrs.kundur_tgr import kundur_tgr

def add_avr(dae,syn_data,name,bus_name):
    
    if syn_data['avr']['type'] == 'sexs':
        sexs(dae,syn_data,name,bus_name)

    if syn_data['avr']['type'] == 'sexsq':
        sexsq(dae,syn_data,name)

    if syn_data['avr']['type'] == 'kundur':
        kundur(dae,syn_data,name,bus_name)

    if syn_data['avr']['type'] == 'avr_kundurq':
        avr_kundurq(dae,syn_data,name,bus_name)

    if syn_data['avr']['type'] == 'ntsst4':
        ntsst4(dae,syn_data,name,bus_name)

    if syn_data['avr']['type'] == 'ntsst1':
        ntsst1(dae,syn_data,name,bus_name)

    if syn_data['avr']['type'] == 'kundur_tgr':
        kundur_tgr(dae,syn_data,name,bus_name)
        
