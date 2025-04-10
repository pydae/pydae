# -*- coding: utf-8 -*-
"""
Created on Thu August 10 23:52:55 2022

@author: jmmauricio
"""

import numpy as np
import sympy as sym
import numba


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
        
        if 'code' in line:
            if 'X' in self.data['line_codes'][line['code']]:
                R_primitive_matrix = np.array(self.data['line_codes'][line['code']]['R'])
                X_primitive_matrix = np.array(self.data['line_codes'][line['code']]['X'])
                Z_primitive_matrix = R_primitive_matrix + 1j*X_primitive_matrix
                Y_primitive_matrix = np.linalg.inv(Z_primitive_matrix*line['m']/1000)
                line.update({'Y_primitive':Y_primitive_matrix}) 
                
                N_branches = Y_primitive_matrix.shape[0]
                line.update({'N_branches':N_branches}) 
        else:
            if 'X' in line:
                N_branches = 4
                if 'N_branches' in line: N_branches = line['N_branches']
                R_primitive_matrix = np.eye(N_branches)*line['R']
                X_primitive_matrix = np.eye(N_branches)*line['X']
                Z_primitive_matrix = R_primitive_matrix + 1j*X_primitive_matrix
                Y_primitive_matrix = np.linalg.inv(Z_primitive_matrix)
                line.update({'Y_primitive':Y_primitive_matrix}) 
                line.update({'N_branches':N_branches})                 
            if 'X_km' in line:
                N_branches = 4
                if 'km' in line:
                    line.update({'m':line['km']*1000})
                if 'N_branches' in line: N_branches = line['N_branches']
                R_primitive_matrix = np.eye(N_branches)*line['R_km']*line['m']/1000
                X_primitive_matrix = np.eye(N_branches)*line['X_km']*line['m']/1000
                Z_primitive_matrix = R_primitive_matrix + 1j*X_primitive_matrix
                Y_primitive_matrix = np.linalg.inv(Z_primitive_matrix)
                line.update({'Y_primitive':Y_primitive_matrix}) 
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


def change_line(model,bus_j,bus_k,data_line_code,length,N_branches=4):
    """
    Change line parameters.

    Parameters
    ----------
    model : pydae object
            pydae model of the grid where the line will be modified
    bus_j : str
            name of the from bus of the line
    bus_k : str
            name of the to bus of the line
    data_line_code : dict 
            dictionary with 2 keys 'R' and 'X' and the respective matrices as list of lists (see example below)

            
    Returns
    -------
    None


    Note
    ----

    Line must be considered *symbolic* ("sym":true) when building the model.

    "lines":[{"bus_j": "A2",  "bus_k": "A3",  "code": "UG1", "m": 100.0,"monitor":true,"sym":true}],
       
                 
    Example
    -------

    data_line_code = {"R":[[ 0.211,  0.049,  0.049,  0.049],
                           [ 0.049,  0.211,  0.049,  0.049],
                           [ 0.049,  0.049,  0.211,  0.049],
                           [ 0.049,  0.049,  0.049,  0.211]],
                      "X":[[ 0.747,  0.673,  0.651,  0.673],
                           [ 0.673,  0.747,  0.673,  0.651],
                           [ 0.651,  0.673,  0.747,  0.673],
                           [ 0.673,  0.651,  0.673,  0.747]], "I_max":430.0}
    
    length = 10
    N_branches = 4
    change_line(model,'A2','A3',data_line_code,length,N_branches)
    
    """
    

    R_primitive_matrix = np.array(data_line_code['R'])
    X_primitive_matrix = np.array(data_line_code['X'])
    Z_primitive_matrix = R_primitive_matrix + 1j*X_primitive_matrix
    Y_primitive_matrix = np.linalg.inv(Z_primitive_matrix*length/1000)

    for ibranch in range(N_branches):

        node_j = ibranch
        node_k = ibranch    

        for icol in range(N_branches):
            line_name = f"{bus_j}_{node_j}_{bus_k}_{node_k}_{icol}"
            g_jk  = f"g_{line_name}" 
            b_jk  = f"b_{line_name}" 
            bs_jk = f"bs_{line_name}"
                           
            model.set_value(g_jk,Y_primitive_matrix[ibranch,icol].real)
            model.set_value(b_jk,Y_primitive_matrix[ibranch,icol].imag)

    return None


def codes_matrix(model,data):
    """
    Generates two numpy arrays, code_g_primitives and code_b_primitives, with N rows each. 
    Each row of code_g_primitives has the flatten real part of primitive admitance matrix while code_b_primitives has the imaginary parts.

    Input
    =====

    Dictionary with line codes and their R and X primitive matrices.

    Returns
    =======

    - code_g_primitives: flatten real part of primitive admitance matrix
    - code_b_primitives: flatten imaginary part of primitive admitance matrix  

    """

    code_y_primitives = [] 

    line_codes = data['line_codes']
    for line_code in line_codes:

        R_primitive_matrix = np.array(line_codes[line_code]['R'])
        X_primitive_matrix = np.array(line_codes[line_code]['X'])
        Z_primitive_matrix = R_primitive_matrix + 1j*X_primitive_matrix
        Y_primitive_matrix = np.linalg.inv(Z_primitive_matrix)
        code_y_primitives += [Y_primitive_matrix.flatten()]
    y_primitives = np.array(code_y_primitives)
    codes_g_primitives = y_primitives.real
    codes_b_primitives =  y_primitives.imag
        
    model.codes_g_primitives = codes_g_primitives
    model.codes_b_primitives = codes_b_primitives

    return codes_g_primitives,codes_b_primitives
    
def lines2p_idx(model,data):
    """

    """

    lines2p_indices_list = []
    for line in data['lines']:
        bus_j = line['bus_j']
        bus_k = line['bus_k']
        line_name = f"{bus_j}_{0}_{bus_k}_{0}_{0}"
        idx = model.params_list.index(f"g_{line_name}")
        N_branches = 4
        lines2p_indices_list += [[idx,N_branches]]
        lines2p_indices = np.array(lines2p_indices_list)
    
    model.lines2p_indices = lines2p_indices

    return lines2p_indices


@numba.njit()
def change_line_fast(p,lines_idxs,line_codes_idx,lengths_km,codes_g_primitives,codes_b_primitives,lines2p_indices):
    for it in range(len(lines_idxs)):
        length_km = lengths_km[it]
        line_code_idx = line_codes_idx[it]
        ini_idx = lines2p_indices[lines_idxs[it],0] 
        end_idx = ini_idx + lines2p_indices[lines_idxs[it],1]**2*2

        p[ini_idx:  end_idx:2] = codes_g_primitives[line_code_idx]/length_km
        p[ini_idx+1:end_idx:2] = codes_b_primitives[line_code_idx]/length_km


def Z1Z0_to_primitive_matrix(R1, X1, R0, X0):
    # Impedance values as complex numbers
    Z1 = R1 + 1j * X1  # Positive sequence impedance
    Z0 = R0 + 1j * X0  # Zero sequence impedance
    
    # Mutual impedance approximation (assuming equal for all wires)
    Z_mut = (Z0 - Z1) / 3  # Mutual impedance between phases

    # Create the 3x3 primitive matrix
    Z = np.array([
        [Z1, Z_mut, Z_mut],    # Phase A
        [Z_mut, Z1, Z_mut],    # Phase B
        [Z_mut, Z_mut, Z1]     # Phase C
    ])

    return Z






if __name__ == '__main__':

    data  = {
    "system":{"S_base":1e6, "K_p_agc":0.01, "K_i_agc":0.01, "K_xif":0.01},
    "buses": [{"name":"01","U_kV":20.0, "N_nodes":3},{"name":"02","U_kV":20.0, "N_nodes":3},{"name":"03","U_kV":20.0, "N_nodes":3},
            {"name":"04","U_kV":20.0, "N_nodes":3},{"name":"05","U_kV":20.0, "N_nodes":3},
            {"name":"06","U_kV":20.0, "N_nodes":3},{"name":"07","U_kV":20.0, "N_nodes":3},{"name":"08","U_kV":20.0, "N_nodes":3},
            {"name":"09","U_kV":20.0, "N_nodes":3},{"name":"10","U_kV":20.0, "N_nodes":3},{"name":"11","U_kV":20.0, "N_nodes":3}],
    "lines":     [  
                    {"bus_j":"01", "bus_k":"02", "R_km":0.501, "X_km":0.716, "B_muS_km":47.493, "km":2.82, "N_branches":3},
                    {"bus_j":"02", "bus_k":"03", "R_km":0.501, "X_km":0.716, "B_muS_km":47.493, "km":4.42, "N_branches":3},
                    {"bus_j":"03", "bus_k":"04", "R_km":0.501, "X_km":0.716, "B_muS_km":47.493, "km":0.61, "N_branches":3},
                    {"bus_j":"03", "bus_k":"08", "R_km":0.501, "X_km":0.716, "B_muS_km":47.493, "km":1.30, "N_branches":3},
                    {"bus_j":"04", "bus_k":"05", "R_km":0.501, "X_km":0.716, "B_muS_km":47.493, "km":0.56, "N_branches":3},
                    {"bus_j":"05", "bus_k":"06", "R_km":0.501, "X_km":0.716, "B_muS_km":47.493, "km":1.54, "N_branches":3},
                    {"bus_j":"06", "bus_k":"07", "R_km":0.501, "X_km":0.716, "B_muS_km":47.493, "km":0.24, "N_branches":3},
                    {"bus_j":"07", "bus_k":"08", "R_km":0.501, "X_km":0.716, "B_muS_km":47.493, "km":1.67, "N_branches":3},
                    {"bus_j":"08", "bus_k":"09", "R_km":0.501, "X_km":0.716, "B_muS_km":47.493, "km":0.32, "N_branches":3},
                    {"bus_j":"09", "bus_k":"10", "R_km":0.501, "X_km":0.716, "B_muS_km":47.493, "km":0.77, "N_branches":3},
                    {"bus_j":"10", "bus_k":"11", "R_km":0.501, "X_km":0.716, "B_muS_km":47.493, "km":0.33, "N_branches":3},
                ],
    "loads" : [
            #{"bus": "12", "kVA":0e3, "pf": 0.95, "T_i":0.01,"I_max":1200},
            #{"bus": "13", "kVA":0e3, "pf": 0.95, "T_i":0.01,"I_max":1200},
            #{"bus": "14", "kVA":0e3, "pf": 0.95, "T_i":0.01,"I_max":1200}
            ],
    "sources":[
        {"type":"ac3ph3w_ideal", "bus": "01", "S_n":100e3, "U_n" :20e3, "R":0.01, "X":0.1}
            ]}

    from pydae.urisi.urisi_builder import urisi

    grid = urisi(data)
    grid.uz_jacs = False
    grid.build('temp')