# -*- coding: utf-8 -*-
"""
Created on Thu August 10 23:52:55 2022

@author: jmmauricio
"""

from pydae.bmapu.govs.tgov1 import tgov1
from pydae.bmapu.govs.agov1 import agov1
from pydae.bmapu.govs.ntsieeeg1 import ntsieeeg1

def add_gov(dae,syn_data,name,bus_name):
    
    if syn_data['gov']['type'] == 'tgov1':
        tgov1(dae,syn_data,name,bus_name)
    if syn_data['gov']['type'] == 'agov1':
        agov1(dae,syn_data,name,bus_name)
    if syn_data['gov']['type'] == 'ntsieeeg1':
        ntsieeeg1(dae,syn_data,name,bus_name)