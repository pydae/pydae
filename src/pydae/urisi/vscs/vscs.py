# -*- coding: utf-8 -*-
"""
Created on Thu August 10 23:52:55 2022

@author: jmmauricio
"""

import numpy as np
import sympy as sym
from pydae.urisi.vscs.ac3ph4wgfpi2 import ac3ph4wgfpi2
from pydae.urisi.vscs.ac3ph4wgfpi3 import ac3ph4wgfpi3
from pydae.urisi.vscs.ac_3ph_4w_l import ac_3ph_4w_l
from pydae.urisi.vscs.acdc_3ph_4w_vdc_q import acdc_3ph_4w_vdc_q
from pydae.urisi.vscs.acdc_3ph_4w_pq import acdc_3ph_4w_pq
from pydae.urisi.vscs.dcdc_gfl import dcdc_gfl
from pydae.urisi.vscs.dcdc_gfh import dcdc_gfh
from pydae.urisi.vscs.dcdc_ph import dcdc_ph
from pydae.urisi.vscs.ac_3ph_4w_pq import ac_3ph_4w_pq
from pydae.urisi.vscs.ac_3ph_4w import ac_3ph_4w
from pydae.urisi.vscs.breaker import breaker
from pydae.urisi.vsgs.vsgs import add_vsg
from pydae.urisi.ess.ess import add_ess
from pydae.urisi.pvs.pvs import add_pv
from pydae.urisi.vsc_ctrls.vsc_ctrls import  add_vsc_ctrl


def add_vscs(grid):

    buses = grid.data['buses']
    buses_list = [bus['name'] for bus in buses]

    for item in grid.data['vscs']:

        name = item['bus']
        bus_name = item['bus']

        if item['type'] == 'ac_3ph_4w_l':
            ac_3ph_4w_l(grid,item)
        if item['type'] == 'ac3ph4wgfpi2':
            ac3ph4wgfpi2(grid,item)
        if item['type'] == 'ac3ph4wgfpi3':
            ac3ph4wgfpi3(grid,item)
        if item['type'] == 'acdc_3ph_4w_vdc_q': 
            acdc_3ph_4w_vdc_q(grid,item)
        if item['type'] == 'acdc_3ph_4w_pq': 
            acdc_3ph_4w_pq(grid,item)
        if item['type'] == 'dcdc_gfl': 
            dcdc_gfl(grid,item)
        if item['type'] == 'dcdc_gfh': 
            dcdc_gfh(grid,item)
        if item['type'] == 'dcdc_ph': 
            dcdc_ph(grid,item)
        if item['type'] == 'ac_3ph_4w_pq': 
            ac_3ph_4w_pq(grid,item)
        if item['type'] == 'ac_3ph_4w': 
            ac_3ph_4w(grid,item)
        if item['type'] == 'breaker': 
            breaker(grid,item)

        if 'vsg' in item:         
            add_vsg(grid,item['vsg'],name,bus_name)  
        if 'ess' in item:         
            add_ess(grid,item['ess'],name,bus_name)  
        if 'pv' in item:         
            add_pv(grid,item['pv'],name,bus_name)  
        if 'vsc_ctrl' in item:  
            print('vsc_ctrl')       
            add_vsc_ctrl(grid,item['vsc_ctrl'],name,bus_name)  

    if 'vsgs' in grid.data:     
        for item in grid.data['vsgs']:    
            add_vsg(grid,item)
