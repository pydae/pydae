# -*- coding: utf-8 -*-
"""
Created on Thu August 10 23:52:55 2022

@author: jmmauricio
"""

import numpy as np
import sympy as sym


def add_lines(self):

    for line in self.lines:

        for ibranch in range(line['N_branches']):

            bus_j = line['bus_j']
            bus_k = line['bus_k']

            if 'bus_j_nodes' in line:
                node_j = line['bus_j_nodes'][ibranch]
            else:
                node_j = ibranch

            if 'bus_k_nodes' in line:
                node_k = line['bus_k_nodes'][ibranch]
            else:
                node_k = ibranch

            idx_j = self.nodes_list.index(f"{bus_j}.{node_j}")
            idx_k = self.nodes_list.index(f"{bus_k}.{node_k}")    

            self.A[self.it_branch+ibranch,idx_j] = 1
            self.A[self.it_branch+ibranch,idx_k] =-1   
            #A[it+1,idx_j] = 1
            #A[it+2,idx_k] = 1   
            
            for icol in range(line['N_branches']):
                if line['sym']:
                    if np.abs(line['Y_primitive'][ibranch,icol]) != 0.0:
                        line_name = f"{bus_j}_{node_j}_{bus_k}_{node_k}_{icol}"
                        g_jk = sym.Symbol(f"g_{line_name}", real=True) 
                        b_jk = sym.Symbol(f"b_{line_name}", real=True) 
                        bs_jk = sym.Symbol(f"bs_{line_name}", real=True) 
                        self.G_primitive[self.it_branch+ibranch,self.it_branch+icol] = g_jk
                        self.B_primitive[self.it_branch+ibranch,self.it_branch+icol] = b_jk
                        self.dae['params_dict'].update({str(g_jk):line['Y_primitive'][ibranch,icol].real})
                        self.dae['params_dict'].update({str(b_jk):line['Y_primitive'][ibranch,icol].imag})

                else:
                    if np.abs(line['Y_primitive'][ibranch,icol]) != 0.0:
                        line_name = f"{bus_j}_{node_j}_{bus_k}_{node_k}_{icol}"
                        g_jk = line['Y_primitive'][ibranch,icol].real 
                        b_jk = line['Y_primitive'][ibranch,icol].imag 
                        bs_jk = sym.Symbol(f"bs_{line_name}", real=True) 
                        self.G_primitive[self.it_branch+ibranch,self.it_branch+icol] = g_jk
                        self.B_primitive[self.it_branch+ibranch,self.it_branch+icol] = b_jk


            #B_primitive[it,ibranch+it] = b_jk
            #B_primitive[it+1,it+1] = bs_jk/2
            #B_primitive[it+2,it+2] = bs_jk/2

        self.it_branch += line['N_branches']



def lines_preprocess(self):

    for line in self.lines:
        
        if 'X' in self.data['line_codes'][line['code']]:
            R_primitive_matrix = np.array(self.data['line_codes'][line['code']]['R'])
            X_primitive_matrix = np.array(self.data['line_codes'][line['code']]['X'])
            Z_primitive_matrix = R_primitive_matrix + 1j*X_primitive_matrix
            Y_primitive_matrix = np.linalg.inv(Z_primitive_matrix*line['m']/1000)
            line.update({'Y_primitive':Y_primitive_matrix}) 
            
            N_branches = Y_primitive_matrix.shape[0]
            line.update({'N_branches':N_branches}) 

        if not 'sym' in line:
            line.update({'sym':False})



        self.N_branches += N_branches

        
def add_line_monitors(self):


    for line in self.lines:

        if 'monitor' in line:
            if line['monitor']:
                            
                bus_j_name = line['bus_j']
                bus_k_name = line['bus_k']
  
                for it in range(line['N_branches']):

                    if 'bus_j_nodes' in line:
                        node_j = line['bus_j_nodes'][it]
                    else:
                        node_j = it

                    if 'bus_k_nodes' in line:
                        node_k = line['bus_k_nodes'][it]
                    else:
                        node_k = it

                    self.dae['h_dict'].update({f"i_l_{bus_j_name}_{node_j}_{bus_k_name}_{node_k}_r" : sym.re(self.I_lines[self.it_branch ++it,0])})
                    self.dae['h_dict'].update({f"i_l_{bus_j_name}_{node_j}_{bus_k_name}_{node_k}_i" : sym.im(self.I_lines[self.it_branch ++it,0])})

        self.it_branch += line['N_branches']
