# -*- coding: utf-8 -*-
"""
Created on Thu August 10 23:52:55 2022

@author: jmmauricio
"""

from pydae.bps.govs.tgov1 import tgov1
from pydae.bps.govs.tgov2 import tgov2
from pydae.bps.govs.hygov import hygov
from pydae.bps.govs.ggov1 import ggov1
from pydae.bps.govs.agov1 import agov1
from pydae.bps.govs.ntsieeeg1 import ntsieeeg1
from pydae.bps.govs.ieeeg1 import ieeeg1
from pydae.bps.govs.dgov import dgov

def add_gov(dae,syn_data,name,bus_name):

    if syn_data['gov']['type'] == 'tgov1':
        tgov1(dae,syn_data,name,bus_name)
    if syn_data['gov']['type'] == 'tgov2':
        tgov2(dae,syn_data,name,bus_name)
    if syn_data['gov']['type'] == 'hygov':
        hygov(dae,syn_data,name,bus_name)
    if syn_data['gov']['type'] == 'ggov1':
        ggov1(dae,syn_data,name,bus_name)
    if syn_data['gov']['type'] == 'agov1':
        agov1(dae,syn_data,name,bus_name)
    if syn_data['gov']['type'] == 'ntsieeeg1':
        ntsieeeg1(dae,syn_data,name,bus_name)
    if syn_data['gov']['type'] == 'ieeeg1':
        ieeeg1(dae,syn_data,name,bus_name)
    if syn_data['gov']['type'] == 'dgov':
        dgov(dae,syn_data,name,bus_name)