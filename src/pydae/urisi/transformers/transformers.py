
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  5 13:04:45 2017

@author: jmmauricio
"""

import numpy as np
import sympy as sym
import difflib

from pydae.urisi.transformers.Dyn11 import add_Dyn11
from pydae.urisi.transformers.Dyg11 import add_Dyg11



def add_trafos(self):


    for trafo in self.transformers:

        if trafo['connection'] == 'Dyn11':
            add_Dyn11(self,trafo)
        elif trafo['connection'] == 'Dyg11':
            add_Dyg11(self,trafo)
        else: 
            add_trafo(self,trafo)

def add_trafo(self, trafo):

        G_primitive = trafo['Y_primitive'].real
        B_primitive = trafo['Y_primitive'].imag

        rl = self.it_branch
        rh = self.it_branch + trafo['N_branches']
        self.G_primitive[rl:rh,rl:rh] = G_primitive
        self.B_primitive[rl:rh,rl:rh] = B_primitive


        for item in trafo['bus_j_nodes']: # the list of nodes '[<bus>.<node>.<node>...]' is created 
            node_j = f"{trafo['bus_j']}.{item}"
            col = self.nodes_list.index(node_j)
            row = self.it_branch

            self.A[row,col] = 1
            self.it_branch +=1  

        for item in  trafo['bus_k_nodes']: # the list of nodes '[<bus>.<node>.<node>...]' is created 
            node_k = f"{trafo['bus_k']}.{item}"
            col = self.nodes_list.index(node_k)
            row = self.it_branch
            self.A[row,col] = 1
            self.it_branch +=1  



def add_trafo_symbolic(self,trafo):

        bus_j_name = trafo['bus_j']
        bus_k_name = trafo['bus_k']
        trafo_name = f'{bus_j_name}_{bus_k_name}'

        tap_a,tap_b,tap_c = sym.symbols(f'tap_a,tap_b,tap_c', real =True)
        tap_a_name,tap_b_name,tap_c_name = sym.symbols(f'tap_a_{trafo_name},tap_b_{trafo_name},tap_c_{trafo_name}', real =True)

        G_primitive = sym.re(trafo['Y_primitive']).subs(tap_a,tap_a_name).subs(tap_b,tap_b_name).subs(tap_c,tap_c_name)
        B_primitive = sym.im(trafo['Y_primitive']).subs(tap_a,tap_a_name).subs(tap_b,tap_b_name).subs(tap_c,tap_c_name)

        rl = self.it_branch
        rh = self.it_branch + trafo['N_branches']
        self.G_primitive[rl:rh,rl:rh] = G_primitive
        self.B_primitive[rl:rh,rl:rh] = B_primitive


        for item in trafo['bus_j_nodes']: # the list of nodes '[<bus>.<node>.<node>...]' is created 
            node_j = f"{trafo['bus_j']}.{item}"
            col = self.nodes_list.index(node_j)
            row = self.it_branch

            self.A[row,col] = 1
            self.it_branch +=1  

        for item in  trafo['bus_k_nodes']: # the list of nodes '[<bus>.<node>.<node>...]' is created 
            node_k = f"{trafo['bus_k']}.{item}"
            col = self.nodes_list.index(node_k)
            row = self.it_branch
            self.A[row,col] = 1
            self.it_branch +=1  
        
        self.dae['u_ini_dict'].update({str(tap_a_name):1.0,str(tap_b_name):1.0,str(tap_c_name):1.0})
        self.dae['u_run_dict'].update({str(tap_a_name):1.0,str(tap_b_name):1.0,str(tap_c_name):1.0})


def add_trafo_monitors(self):

    it_branch = self.it_branch

    for trafo in self.transformers:
        if 'monitor' in trafo:
            if trafo['monitor']:
                i = 0
                bus_j_name = trafo['bus_j']
                bus_k_name = trafo['bus_k']
                for it in trafo['bus_j_nodes']:
                    self.dae['h_dict'].update({f"i_t_{bus_j_name}_{bus_k_name}_1_{it}_r":sym.re(self.I_lines[it_branch+i,0])})
                    self.dae['h_dict'].update({f"i_t_{bus_j_name}_{bus_k_name}_1_{it}_i":sym.im(self.I_lines[it_branch+i,0])})
                    i += 1
                
                for it in trafo['bus_k_nodes']:
                    self.dae['h_dict'].update({f"i_t_{bus_j_name}_{bus_k_name}_2_{it}_r":sym.re(self.I_lines[it_branch+i,0])})
                    self.dae['h_dict'].update({f"i_t_{bus_j_name}_{bus_k_name}_2_{it}_i":sym.im(self.I_lines[it_branch+i,0])})
                    i += 1



        it_branch += trafo['N_branches']



def trafos_preprocess(self):

    for trafo in self.transformers:
        S_n = trafo['S_n_kVA']*1000.0
        if 'U_1_kV' in trafo:
            U_jn = trafo['U_1_kV']*1000.0
        if 'U_2_kV' in trafo:
            U_kn = trafo['U_2_kV']*1000.0
        if 'U_j_kV' in trafo:
            U_jn = trafo['U_j_kV']*1000.0
        if 'U_k_kV' in trafo:
            U_kn = trafo['U_k_kV']*1000.0
        Z_cc_pu = trafo['R_cc_pu'] +1j*trafo['X_cc_pu']
        connection = trafo['connection']
        
        Y_primitive_matrix,nodes_j,nodes_k = trafo_yprim(S_n,U_jn,U_kn,Z_cc_pu,connection=connection)

        trafo.update({'Y_primitive':Y_primitive_matrix}) 
        trafo.update({'bus_j_nodes':nodes_j}) 
        trafo.update({'bus_k_nodes':nodes_k}) 
        
        N_branches = Y_primitive_matrix.shape[0]
        trafo.update({'N_branches':N_branches}) 
    
        self.N_branches += N_branches


def trafo_yprim(S_n,U_1n,U_2n,Z_cc,connection='Dyg11'):
    '''
    Trafo primitive as developed in: (in the paper Ynd11)
    R. C. Dugan and S. Santoso, “An example of 3-phase transformer modeling for distribution system analysis,” 
    2003 IEEE PES Transm. Distrib. Conf. Expo. (IEEE Cat. No.03CH37495), vol. 3, pp. 1028–1032, 2003. 
    
    '''

    connections_list = ['Dyn1', 'Yy_3wires','Dyn5','Dyn11','Dyn11t','Ygd5_3w','Ygd1_3w','Ygd11_3w','ZigZag','Dyg11_3w','Dyg11','Ynd11']

    if connection not in connections_list:
        closest_connection = difflib.get_close_matches(connection, connections_list)
        print('Transformer connection "{:s}" not found, did you mean: "{:s}"?'.format(connection,closest_connection[0]))

    if connection=='Dyn1':
        z_a = 3*Z_cc*1.0**2/S_n
        z_b = 3*Z_cc*1.0**2/S_n
        z_c = 3*Z_cc*1.0**2/S_n
        U_1 = U_1n
        U_2 = U_2n/np.sqrt(3)
        Z_B = np.array([[z_a, 0.0, 0.0],
                        [0.0, z_b, 0.0],
                        [0.0, 0.0, z_c],])                             
        N_a = np.array([[ 1/U_1,     0],
                         [-1/U_1,     0],
                         [     0, 1/U_2],
                         [     0,-1/U_2]])           
        N_row_a = np.hstack((N_a,np.zeros((4,4))))
        N_row_b = np.hstack((np.zeros((4,2)),N_a,np.zeros((4,2))))
        N_row_c = np.hstack((np.zeros((4,4)),N_a))
        
        N = np.vstack((N_row_a,N_row_b,N_row_c))

        B = np.array([[ 1, 0, 0],
                      [-1, 0, 0],
                      [ 0, 1, 0],
                      [ 0,-1, 0],
                      [ 0, 0, 1],
                      [ 0, 0,-1]])
    
        Y_1 = B @ np.linalg.inv(Z_B) @ B.T
        Y_w = N @ Y_1 @ N.T
        A_trafo = np.zeros((7,12))

        A_trafo[0,0] = 1.0
        A_trafo[0,9] = 1.0
        A_trafo[1,1] = 1.0
        A_trafo[1,4] = 1.0
        A_trafo[2,5] = 1.0
        A_trafo[2,8] = 1.0

        A_trafo[3,2] = 1.0
        A_trafo[4,6] = 1.0
        A_trafo[5,10] = 1.0
        
        A_trafo[6,3] = 1.0
        A_trafo[6,7] = 1.0
        A_trafo[6,11] = 1.0

        nodes_j = [0,1,2,3]
        nodes_k = [0,1,2]  


    if connection=='Yy_3wires':
        z_a = 3*Z_cc*1.0**2/S_n
        z_b = 3*Z_cc*1.0**2/S_n
        z_c = 3*Z_cc*1.0**2/S_n
        U_1 = U_1n/np.sqrt(3)
        U_2 = U_2n/np.sqrt(3)
        Z_B = np.array([[z_a, 0.0, 0.0],
                        [0.0, z_b, 0.0],
                        [0.0, 0.0, z_c],])                             
        N_a = np.array([[ 1/U_1,     0],
                         [-1/U_1,     0],
                         [     0, 1/U_2],
                         [     0,-1/U_2]])           
        N_row_a = np.hstack((N_a,np.zeros((4,4))))
        N_row_b = np.hstack((np.zeros((4,2)),N_a,np.zeros((4,2))))
        N_row_c = np.hstack((np.zeros((4,4)),N_a))
        
        N = np.vstack((N_row_a,N_row_b,N_row_c))

        B = np.array([[ 1, 0, 0],
                      [-1, 0, 0],
                      [ 0, 1, 0],
                      [ 0,-1, 0],
                      [ 0, 0, 1],
                      [ 0, 0,-1]])
    
        Y_1 = B @ np.linalg.inv(Z_B) @ B.T
        Y_w = N @ Y_1 @ N.T
        A_trafo = np.zeros((6,12))
        A_trafo[0,0] = 1.0
        A_trafo[1,4] = 1.0
        A_trafo[2,8] = 1.0
        A_trafo[3,2] = 1.0
        A_trafo[4,6] = 1.0
        A_trafo[5,10] = 1.0

        nodes_j = [0,1,2]
        nodes_k = [0,1,2]  


    if connection=='Dyn5':
        z_a = Z_cc*1.0**2/S_n*3
        z_b = Z_cc*1.0**2/S_n*3
        z_c = Z_cc*1.0**2/S_n*3
        U_1 = U_1n
        U_2 = U_2n/np.sqrt(3)
        Z_B = np.array([[z_a, 0.0, 0.0],
                        [0.0, z_b, 0.0],
                        [0.0, 0.0, z_c],])                             
        N_a = np.array([[ 1/U_1,     0],
                         [-1/U_1,     0],
                         [     0, 1/U_2],
                         [     0,-1/U_2]])           
        N_row_a = np.hstack((N_a,np.zeros((4,4))))
        N_row_b = np.hstack((np.zeros((4,2)),N_a,np.zeros((4,2))))
        N_row_c = np.hstack((np.zeros((4,4)),N_a))
        
        N = np.vstack((N_row_a,N_row_b,N_row_c))

        B = np.array([[ 1, 0, 0],
                      [-1, 0, 0],
                      [ 0, 1, 0],
                      [ 0,-1, 0],
                      [ 0, 0, 1],
                      [ 0, 0,-1]])
    
        Y_1 = B @ np.linalg.inv(Z_B) @ B.T
        Y_w = N @ Y_1 @ N.T
        A_trafo = np.zeros((7,12))

        A_trafo[0,1] = 1.0
        A_trafo[0,4] = 1.0
        A_trafo[1,5] = 1.0
        A_trafo[1,8] = 1.0
        A_trafo[2,0] = 1.0
        A_trafo[2,9] = 1.0

        A_trafo[3,2] = 1.0
        A_trafo[4,6] = 1.0
        A_trafo[5,10] = 1.0
        
        A_trafo[6,3] = 1.0
        A_trafo[6,7] = 1.0
        A_trafo[6,11] = 1.0

        nodes_j = [0,1,2]
        nodes_k = [0,1,2,3]  

    if connection=='Dyn11':

        z_a = Z_cc*1.0**2/S_n*3
        z_b = Z_cc*1.0**2/S_n*3
        z_c = Z_cc*1.0**2/S_n*3
        U_1 = U_1n
        U_2 = U_2n/np.sqrt(3)
        Z_B = np.array([[z_a, 0.0, 0.0],
                        [0.0, z_b, 0.0],
                        [0.0, 0.0, z_c],])                             
        N_a = np.array([[ 1/U_1,     0],
                         [-1/U_1,     0],
                         [     0, 1/U_2],
                         [     0,-1/U_2]])           
        N_row_a = np.hstack((N_a,np.zeros((4,4))))
        N_row_b = np.hstack((np.zeros((4,2)),N_a,np.zeros((4,2))))
        N_row_c = np.hstack((np.zeros((4,4)),N_a))

        N = np.vstack((N_row_a,N_row_b,N_row_c))

        B = np.array([[ 1, 0, 0],
                      [-1, 0, 0],
                      [ 0, 1, 0],
                      [ 0,-1, 0],
                      [ 0, 0, 1],
                      [ 0, 0,-1]])
    
        Y_1 = B @ np.linalg.inv(Z_B) @ B.T
        Y_w = N @ Y_1 @ N.T
        A_trafo = np.zeros((7,12))

        A_trafo[0,1] = 1.0
        A_trafo[0,4] = 1.0
        A_trafo[1,5] = 1.0
        A_trafo[1,8] = 1.0
        A_trafo[2,0] = 1.0
        A_trafo[2,9] = 1.0

        A_trafo[3,3] = 1.0
        A_trafo[4,7] = 1.0
        A_trafo[5,11] = 1.0
        
        A_trafo[6,2] = 1.0
        A_trafo[6,6] = 1.0
        A_trafo[6,10] = 1.0

        nodes_j = [0,1,2]
        nodes_k = [0,1,2,3]  

    if connection=='Dyn11t':

        tap_a,tap_b,tap_c = sym.symbols('tap_a,tap_b,tap_c', real = True)


        z_a = Z_cc*1.0**2/S_n*3
        z_b = Z_cc*1.0**2/S_n*3
        z_c = Z_cc*1.0**2/S_n*3
        U_1 = U_1n
        U_2 = U_2n/np.sqrt(3)
        Z_B = np.array([[z_a, 0.0, 0.0],
                        [0.0, z_b, 0.0],
                        [0.0, 0.0, z_c],])                             
        N_a = np.array([[ 1/(U_1*tap_a),     0],
                        [-1/(U_1*tap_a),     0],
                        [     0, 1/U_2],
                        [     0,-1/U_2]])  
        N_b = np.array([[ 1/(U_1*tap_b),     0],
                        [-1/(U_1*tap_b),     0],
                        [     0, 1/U_2],
                        [     0,-1/U_2]])   
        N_c = np.array([[ 1/(U_1*tap_c),     0],
                        [-1/(U_1*tap_c),     0],
                        [     0, 1/U_2],
                        [     0,-1/U_2]])            
        N_row_a = np.hstack((N_a,np.zeros((4,4))))
        N_row_b = np.hstack((np.zeros((4,2)),N_b,np.zeros((4,2))))
        N_row_c = np.hstack((np.zeros((4,4)),N_c))

        N = np.vstack((N_row_a,N_row_b,N_row_c))

        B = np.array([[ 1, 0, 0],
                      [-1, 0, 0],
                      [ 0, 1, 0],
                      [ 0,-1, 0],
                      [ 0, 0, 1],
                      [ 0, 0,-1]])
    
        Y_1 = B @ np.linalg.inv(Z_B) @ B.T
        Y_w = sym.simplify(sym.Matrix(N @ Y_1 @ N.T))
        A_trafo = np.zeros((7,12))

        A_trafo[0,1] = 1.0
        A_trafo[0,4] = 1.0
        A_trafo[1,5] = 1.0
        A_trafo[1,8] = 1.0
        A_trafo[2,0] = 1.0
        A_trafo[2,9] = 1.0

        A_trafo[3,3] = 1.0
        A_trafo[4,7] = 1.0
        A_trafo[5,11] = 1.0
        
        A_trafo[6,2] = 1.0
        A_trafo[6,6] = 1.0
        A_trafo[6,10] = 1.0

        nodes_j = [0,1,2]
        nodes_k = [0,1,2,3]  

    if connection=='Ygd5_3w':  
        z_a = 3*Z_cc*1.0**2/S_n
        z_b = 3*Z_cc*1.0**2/S_n
        z_c = 3*Z_cc*1.0**2/S_n
        U_1 = U_1n #
        U_2 = U_2n*np.sqrt(3)
        Z_B = np.array([[z_a, 0.0, 0.0],
                        [0.0, z_b, 0.0],
                        [0.0, 0.0, z_c],])                             
        N_a = np.array([[ 1/U_1,     0],
                         [-1/U_1,     0],
                         [     0, 1/U_2],
                         [     0,-1/U_2]])           
        N_row_a = np.hstack((N_a,np.zeros((4,4))))
        N_row_b = np.hstack((np.zeros((4,2)),N_a,np.zeros((4,2))))
        N_row_c = np.hstack((np.zeros((4,4)),N_a))
        
        N = np.vstack((N_row_a,N_row_b,N_row_c))

        B = np.array([[ 1, 0, 0],
                      [-1, 0, 0],
                      [ 0, 1, 0],
                      [ 0,-1, 0],
                      [ 0, 0, 1],
                      [ 0, 0,-1]])
    
        Y_1 = B @ np.linalg.inv(Z_B) @ B.T
        Y_w = N @ Y_1 @ N.T
        A_trafo = np.zeros((6,12))

        A_trafo[0,0] = 1.0
        A_trafo[1,4] = 1.0
        A_trafo[2,8] = 1.0
        
        A_trafo[3,3]  = 1.0
        A_trafo[3,6]  = 1.0
        A_trafo[4,7]  = 1.0
        A_trafo[4,10] = 1.0
        A_trafo[5,2]  = 1.0
        A_trafo[5,11] = 1.0

    if connection=='Ygd1_3w':  
        z_a = 3*Z_cc*1.0**2/S_n
        z_b = 3*Z_cc*1.0**2/S_n
        z_c = 3*Z_cc*1.0**2/S_n
        U_1 = U_1n #
        U_2 = U_2n*np.sqrt(3)
        Z_B = np.array([[z_a, 0.0, 0.0],
                        [0.0, z_b, 0.0],
                        [0.0, 0.0, z_c],])                             
        N_a = np.array([[ 1/U_1,     0],
                         [-1/U_1,     0],
                         [     0, 1/U_2],
                         [     0,-1/U_2]])           
        N_row_a = np.hstack((N_a,np.zeros((4,4))))
        N_row_b = np.hstack((np.zeros((4,2)),N_a,np.zeros((4,2))))
        N_row_c = np.hstack((np.zeros((4,4)),N_a))
        
        N = np.vstack((N_row_a,N_row_b,N_row_c))

        B = np.array([[ 1, 0, 0],
                      [-1, 0, 0],
                      [ 0, 1, 0],
                      [ 0,-1, 0],
                      [ 0, 0, 1],
                      [ 0, 0,-1]])
    
        Y_1 = B @ np.linalg.inv(Z_B) @ B.T
        Y_w = N @ Y_1 @ N.T
        A_trafo = np.zeros((6,12))

        A_trafo[0,0] = 1.0
        A_trafo[1,4] = 1.0
        A_trafo[2,8] = 1.0
        
        A_trafo[3,2]  = 1.0
        A_trafo[3,11]  = 1.0
        A_trafo[4,3]  = 1.0
        A_trafo[4,6] = 1.0
        A_trafo[5,7]  = 1.0
        A_trafo[5,10] = 1.0

    if connection=='Ygd11_3w': 
        z_a = Z_cc*1.0**2/S_n
        z_b = Z_cc*1.0**2/S_n
        z_c = Z_cc*1.0**2/S_n
        U_1 = U_1n #
        U_2 = U_2n*np.sqrt(3)
        Z_B = np.array([[z_a, 0.0, 0.0],
                        [0.0, z_b, 0.0],
                        [0.0, 0.0, z_c],])                             
        N_a = np.array([[ 1/U_1,     0],
                         [-1/U_1,     0],
                         [     0, 1/U_2],
                         [     0,-1/U_2]])           
        N_row_a = np.hstack((N_a,np.zeros((4,4))))
        N_row_b = np.hstack((np.zeros((4,2)),N_a,np.zeros((4,2))))
        N_row_c = np.hstack((np.zeros((4,4)),N_a))
        
        N = np.vstack((N_row_a,N_row_b,N_row_c))

        B = np.array([[ 1, 0, 0],
                      [-1, 0, 0],
                      [ 0, 1, 0],
                      [ 0,-1, 0],
                      [ 0, 0, 1],
                      [ 0, 0,-1]])
    
        Y_1 = B @ np.linalg.inv(Z_B) @ B.T
        Y_w = N @ Y_1 @ N.T
        A_trafo = np.zeros((6,12))

        A_trafo[0,1] = 1.0
        A_trafo[1,5] = 1.0
        A_trafo[2,9] = 1.0
        
        A_trafo[3,3]  = 1.0
        A_trafo[3,6]  = 1.0
        A_trafo[4,7]  = 1.0
        A_trafo[4,10] = 1.0
        A_trafo[5,2]  = 1.0
        A_trafo[5,11] = 1.0

    if connection=='ZigZag':   
        z_a = Z_cc*1.0**2/S_n*3
        z_b = Z_cc*1.0**2/S_n*3
        z_c = Z_cc*1.0**2/S_n*3
        U_1 = U_1n #
        U_2 = U_2n
        Z_B = np.array([[z_a, 0.0, 0.0],
                        [0.0, z_b, 0.0],
                        [0.0, 0.0, z_c],])                             


        
        N = np.zeros((12,6))
        N[0,0] =  1.0/U_1
        N[1,0] = -1.0/U_1
        N[6,0] = -1.0/U_1
        N[7,0] =  1.0/U_1

        N[4,2]  =  1.0/U_1
        N[5,2]  = -1.0/U_1
        N[10,2] = -1.0/U_1
        N[11,2] =  1.0/U_1

        N[8,4] =  1.0/U_1
        N[9,4] = -1.0/U_1
        N[2,4] = -1.0/U_1
        N[3,4] =  1.0/U_1
        
        
        N[2,1] =  1.0/U_2
        N[3,1] = -1.0/U_2
    
        N[6,3] =  1.0/U_2
        N[7,3] = -1.0/U_2  

        N[10,5] =  1.0/U_2
        N[11,5] = -1.0/U_2 
        
        #          0  1  2  3  4  5
        # 0 Iw1a   1                   Ia1 0
        # 1 Iw2a  -1                   Ia2 1
        # 2 Iw3a      2                Ib1 2
        # 3 Iw4a     -2                Ib2 3
        # 4 Iw1b         1             Ic1 4
        # 5 Iw2b        -1             Ic2 5
        # 6 Iw3b            2
        # 7 Iw4b           -2
        # 8 Iw1c               1
        # 9 Iw2c              -1
        #10 Iw3c                  2
        #11 Iw4c                 -2
        
        #          0  1  2  3  4  5
        # 0 Iw1a   1                   Ia1 0
        # 1 Iw2a  -1                   Ia2 1
        # 2 Iw3a      2       -1       Ib1 2
        # 3 Iw4a     -2       -1       Ib2 3
        # 4 Iw1b         1             Ic1 4
        # 5 Iw2b        -1             Ic2 5
        # 6 Iw3b  -1        2
        # 7 Iw4b  -1       -2
        # 8 Iw1c               1 
        # 9 Iw2c              -1
        #10 Iw3c        -1        2
        #11 Iw4c        -1       -2
        
        
        B = np.array([[ 1, 0, 0],
                      [-1, 0, 0],
                      [ 0, 1, 0],
                      [ 0,-1, 0],
                      [ 0, 0, 1],
                      [ 0, 0,-1]])
    
        Y_1 = B @ np.linalg.inv(Z_B) @ B.T
        Y_w = N @ Y_1 @ N.T
        A_trafo = np.zeros((7,12))

        A_trafo[0,0] = 1.0
        A_trafo[1,4] = 1.0
        A_trafo[2,8] = 1.0         
        
        A_trafo[6,3]  = 1.0
        A_trafo[6,7]  = 1.0
        A_trafo[6,11] = 1.0
        

        
    if connection=='Dyg11':   
        z_a = 3*Z_cc*1.0**2/S_n
        z_b = 3*Z_cc*1.0**2/S_n
        z_c = 3*Z_cc*1.0**2/S_n
        U_1 = U_1n
        U_2 = U_2n/np.sqrt(3)
        Z_B = np.array([[z_a, 0.0, 0.0],
                        [0.0, z_b, 0.0],
                        [0.0, 0.0, z_c],])                             
        N_a = np.array([[ 1/U_1,     0],
                         [-1/U_1,     0],
                         [     0, 1/U_2],
                         [     0,-1/U_2]])           
        N_row_a = np.hstack((N_a,np.zeros((4,4))))
        N_row_b = np.hstack((np.zeros((4,2)),N_a,np.zeros((4,2))))
        N_row_c = np.hstack((np.zeros((4,4)),N_a))
        
        N = np.vstack((N_row_a,N_row_b,N_row_c))

        B = np.array([[ 1, 0, 0],
                      [-1, 0, 0],
                      [ 0, 1, 0],
                      [ 0,-1, 0],
                      [ 0, 0, 1],
                      [ 0, 0,-1]])
    
        Y_1 = B @ np.linalg.inv(Z_B) @ B.T
        Y_w = N @ Y_1 @ N.T
        A_trafo = np.zeros((6,12))

        A_trafo[0,1] = 1.0
        A_trafo[0,4] = 1.0
        A_trafo[1,5] = 1.0
        A_trafo[1,8] = 1.0
        A_trafo[2,0] = 1.0
        A_trafo[2,9] = 1.0

        A_trafo[3,3] = 1.0
        A_trafo[4,7] = 1.0
        A_trafo[5,11] = 1.0
        
        nodes_j = [0,1,2]
        nodes_k = [0,1,2]  

#    if connection=='Dyg11_3w':
#        z_a = Z_cc*1.0**2/S_n
#        z_b = Z_cc*1.0**2/S_n
#        z_c = Z_cc*1.0**2/S_n
#        U_1 = U_1n/np.sqrt(3)
#        U_2 = U_2n
#        Z_B = np.array([[z_a, 0.0, 0.0],
#                        [0.0, z_b, 0.0],
#                        [0.0, 0.0, z_c],])                             
#        N_a = np.array([[ 1/U_1,     0],
#                         [-1/U_1,     0],
#                         [     0, 1/U_2],
#                         [     0,-1/U_2]])           
#        N_row_a = np.hstack((N_a,np.zeros((4,4))))
#        N_row_b = np.hstack((np.zeros((4,2)),N_a,np.zeros((4,2))))
#        N_row_c = np.hstack((np.zeros((4,4)),N_a))
#        
#        N = np.vstack((N_row_a,N_row_b,N_row_c))
#
#        B = np.array([[ 1, 0, 0],
#                      [-1, 0, 0],
#                      [ 0, 1, 0],
#                      [ 0,-1, 0],
#                      [ 0, 0, 1],
#                      [ 0, 0,-1]])
#    
#        Y_1 = B @ np.linalg.inv(Z_B) @ B.T
#        Y_w = N @ Y_1 @ N.T
#        A_trafo = np.zeros((6,12))
#
#        A_trafo[0,1] = 1.0
#        A_trafo[0,4] = 1.0
#        A_trafo[1,5] = 1.0
#        A_trafo[1,8] = 1.0
#        A_trafo[2,0] = 1.0
#        A_trafo[2,9] = 1.0
#
#        A_trafo[3,3] = 1.0
#        A_trafo[4,7] = 1.0
#        A_trafo[5,11] = 1.0
               
    if connection=='Ynd11':   
        z_a = 3*Z_cc*1.0**2/S_n
        z_b = 3*Z_cc*1.0**2/S_n
        z_c = 3*Z_cc*1.0**2/S_n
        U_1 = U_1n/np.sqrt(3)
        U_2 = U_2n
        Z_B = np.array([[z_a, 0.0, 0.0],
                        [0.0, z_b, 0.0],
                        [0.0, 0.0, z_c],])   

        B = np.array([[ 1, 0, 0],
                      [-1, 0, 0],
                      [ 0, 1, 0],
                      [ 0,-1, 0],
                      [ 0, 0, 1],
                      [ 0, 0,-1]])
                          
        N_a = np.array([[ 1/U_1,     0],
                        [-1/U_1,     0],
                        [     0, 1/U_2],
                        [     0,-1/U_2]])           
        N_row_a = np.hstack((N_a,np.zeros((4,4))))
        N_row_b = np.hstack((np.zeros((4,2)),N_a,np.zeros((4,2))))
        N_row_c = np.hstack((np.zeros((4,4)),N_a))
        
        N = np.vstack((N_row_a,N_row_b,N_row_c))

        Y_1 = B @ np.linalg.inv(Z_B) @ B.T
        Y_w = N @ Y_1 @ N.T
        A_trafo = np.zeros((7,12))
        A_trafo[0,0] = 1.0
        A_trafo[1,4] = 1.0
        A_trafo[2,8] = 1.0
        
        A_trafo[3,1] = 1.0
        A_trafo[3,5] = 1.0
        A_trafo[3,9] = 1.0
        
        A_trafo[4,2] = 1.0
        A_trafo[4,11] = 1.0
        A_trafo[5,3] = 1.0
        A_trafo[5,6] = 1.0
        A_trafo[6,7] = 1.0
        A_trafo[6,10] = 1.0

        nodes_j = [0,1,2,3]
        nodes_k = [0,1,2]   

    if connection in ['Dyn11t']:
        Y_prim = sym.simplify(sym.Matrix(A_trafo @ Y_w @ A_trafo.T))
    else:
        Y_prim = A_trafo @ Y_w @ A_trafo.T



    
    return Y_prim,nodes_j,nodes_k


    # S_n,U_1n,U_2n,Z_cc = 1e6,20e3,400,0.1
    # Y_prim,nodes_j,nodes_k = trafo_yprim(S_n,U_1n,U_2n,Z_cc,connection='Dyn11t')
    # print(Y_prim)
