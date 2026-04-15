# -*- coding: utf-8 -*-
"""
Created on Thu August 10 23:52:55 2022

@author: jmmauricio
"""

import numpy as np
import sympy as sym


def add_shunts(self):

    for shunt in self.shunts:
        node_j_str = str(shunt['bus_nodes'][0])
        node_j = '{:s}.{:s}'.format(shunt['bus'], node_j_str)
        col = self.nodes_list.index(node_j)           
        row_j = self.it_branch
        self.A[row_j,col] = 1
        
        #node_k_str = str(shunt['bus_nodes'][1])
        #if not node_k_str == '0': # when connected to ground
        #    node_k = '{:s}.{:s}'.format(shunt['bus'], str(shunt['bus_nodes'][1]))
        #    row_k = self.nodes_list.index(node_k)            
        #    self.A[row_k,col] = -1
        shunt_name = f"shunt_{shunt['bus']}_{node_j_str}"
        g_jk = sym.Symbol(f"g_{shunt_name}", real=True) 
        b_jk = sym.Symbol(f"b_{shunt_name}", real=True) 
        self.G_primitive[self.it_branch,self.it_branch] = g_jk
        self.B_primitive[self.it_branch,self.it_branch] = b_jk

        Z = shunt['R'] + 1j*shunt['X']
        Y = 1/Z
        self.dae['params_dict'].update({str(g_jk):Y.real})
        self.dae['params_dict'].update({str(b_jk):Y.imag})
       
        self.it_branch += 1



def shunts_preprocess(self):

    for shunt in self.shunts:
        
        self.N_branches += 1

        