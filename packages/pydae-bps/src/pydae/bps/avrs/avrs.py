# -*- coding: utf-8 -*-
"""
Created on Thu August 10 23:52:55 2022

@author: jmmauricio
"""

import numpy as np
import sympy as sym
from pydae.bps.avrs.sexs import sexs
from pydae.bps.avrs.kundur import kundur
from pydae.bps.avrs.ntsst4 import ntsst4
from pydae.bps.avrs.ntsst1 import ntsst1
from pydae.bps.avrs.kundur_tgr import kundur_tgr
from pydae.bps.avrs.avr_1 import avr_1

def add_avr(dae,syn_data,name,bus_name):
    
    if syn_data['avr']['type'] == 'sexs':
        sexs(dae,syn_data,name,bus_name)

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

    if syn_data['avr']['type'] == 'avr_1':
        avr_1(dae,syn_data,name,bus_name)


