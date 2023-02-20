#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2020

@author: jmmauricio
"""

import numpy as np
from pydae.tools import get_v,get_i,get_s
import json
from collections import namedtuple
import numba 

class grid(object):
    
    def __init__(self,syst):
    #def bokeh_tools(data):

        self.syst = syst

        self.s_radio_scale = 0.01
        self.s_radio_max = 20
        self.s_radio_min = 1
        
        with np.load('matrices.npz') as data:
            Y_primitive = data['Y_primitive']
            A_conect = data['A_conect']
            nodes_list = data['nodes_list']
            node_sorter = data['node_sorter']
            Y_vv = data['Y_vv']
            Y_vi = data['Y_vi']        
            N_v = int(data['N_v']) 

            
        self.nodes_list = nodes_list
        self.Y_primitive = Y_primitive
        self.A_conect = A_conect
        self.node_sorter = node_sorter
        self.Y_vv = Y_vv
        self.Y_vi = Y_vi
        self.N_v = N_v

        json_file = 'grid_data.json'
        json_file = json_file
        json_data = open(json_file).read().replace("'",'"')
        data = json.loads(json_data)
        
        self.buses = data['buses']
        if 'transformers' in  data:
            self.transformers = data['transformers']
        else:
            self.transformers = []

        self.lines = data['lines']
        self.loads = data['loads']
        
        if 'vscs' in data:
            self.vscs = data['vscs']
        else: self.vscs = []

        
    def dae2vi(self):
        '''
        For obtaining line currents from node voltages after power flow is solved.

        Returns
        -------
        None.

        '''
        n2a = {'1':'a','2':'b','3':'c','4':'n'}
        a2n = {'a':1,'b':2,'c':3,'n':4}
        
        V_node_list = []
        I_node_list = [0.0]*len(self.nodes_list)
        self.I_node_list = I_node_list
        for item in self.nodes_list:
            bus_name,phase_name = item.split('.')
            #i = get_i(self.syst,bus_name,phase_name=n2a[phase_name],i_type='phasor',dq_name='ri')
            #I_node_list += [i]
            v = get_v(self.syst,bus_name,phase_name=n2a[phase_name],v_type='phasor',dq_name='ri')
            V_node_list += [v]


 
        V_node = np.array(V_node_list).reshape(len(V_node_list),1)
        
        V_known = np.copy(V_node[:self.N_v])
        V_unknown = np.copy(V_node[self.N_v:])
        I_unknown = self.Y_vv @ V_known + self.Y_vi @ V_unknown
        #self.I_node = I_node
        self.V_node = V_node
        self.I_unknown = I_unknown
        self.I_known = np.array(I_node_list).reshape(len(I_node_list),1)

        self.I_node = np.vstack((self.I_unknown,self.I_known))
 
        for load in self.loads:
            bus_name = load['bus']
            if load['type'] == '3P+N':
                for ph in ['a','b','c','n']:
                    idx = list(self.nodes_list).index(f"{load['bus']}.{a2n[ph]}") 
                    i_ = get_i(self.syst,'load_' + bus_name,phase_name=ph,i_type='phasor',dq_name='ri')
                    self.I_node[idx] += i_ 

            if load['type'] == '1P+N':
                ph = load['bus_nodes'][0]
                idx = list(self.nodes_list).index(f"{load['bus']}.{ph}") 
                i_ = get_i(self.syst,'load_' + bus_name,phase_name=n2a[str(ph)],i_type='phasor',dq_name='ri')
                self.I_node[idx] += i_ 
                ph = load['bus_nodes'][1]
                idx = list(self.nodes_list).index(f"{load['bus']}.{ph}") 
                i_ = get_i(self.syst,'load_' + bus_name,phase_name=n2a[str(ph)],i_type='phasor',dq_name='ri')
                self.I_node[idx] += i_ 

        for vsc in self.vscs:
            bus_name = vsc['bus_ac']
            phases = ['a','b','c','n']
            if vsc['type'] == 'ac3ph3wvdcq' or vsc['type'] == 'ac3ph3wpq':
                phases = ['a','b','c']

            for ph in phases:
                idx = list(self.nodes_list).index(f"{vsc['bus_ac']}.{a2n[ph]}") 
                i_ = get_i(self.syst,'vsc_' + bus_name,phase_name=ph,i_type='phasor',dq_name='ri')
                self.I_node[idx] += i_ 
                
            if not vsc['type'] == 'ac3ph3wvdcq'  or vsc['type'] == 'ac3ph3wpq':
                bus_name = vsc['bus_dc']
                for ph in ['a','n']:
                    idx = list(self.nodes_list).index(f"{vsc['bus_dc']}.{a2n[ph]}") 
                    i_ = get_i(self.syst,'vsc_' + bus_name,phase_name=ph,i_type='phasor',dq_name='r')
                    self.I_node[idx] += i_ 
                    
                
        I_lines = self.Y_primitive @ self.A_conect.T @ self.V_node
        self.I_lines = I_lines
        

    def get_v(self):
        '''
		Compute phase-neutral and phase-phase voltages from power flow solution and put values 
		in buses dictionary.		
        '''

        res = {} 
        V_sorted = []
        I_sorted = []
        S_sorted = []
        start_node = 0
        self.V_results = self.V_node
        # self.I_results = self.I_node
        
        V_sorted = self.V_node[self.node_sorter]
        I_sorted = self.I_node[self.node_sorter]   
        
        nodes2string = ['v_an','v_bn','v_cn','v_gn']
        for bus in self.buses:
            N_nodes = bus['N_nodes'] 
#            for node in range(5):
#                bus_node = '{:s}.{:s}'.format(str(bus['bus']),str(node))
#                if bus_node in self.nodes:
#                    V = self.V_results[self.nodes.index(bus_node)][0]
#                    V_sorted += [V]
#                    nodes_in_bus += [node]
#            for node in range(5):
#                bus_node = '{:s}.{:s}'.format(str(bus['bus']),str(node))
#                if bus_node in self.nodes:
#                    I = self.I_results[self.nodes.index(bus_node)][0]
#                    I_sorted += [I]
            if N_nodes==3:   # if 3 phases
                v_ag = V_sorted[start_node+0,0]
                v_bg = V_sorted[start_node+1,0]
                v_cg = V_sorted[start_node+2,0]

                i_a = I_sorted[start_node+0,0]
                i_b = I_sorted[start_node+1,0]
                i_c = I_sorted[start_node+2,0]
                
                s_a = (v_ag)*np.conj(i_a)
                s_b = (v_bg)*np.conj(i_b)
                s_c = (v_cg)*np.conj(i_c)
                
                start_node += 3
                bus.update({'v_an':np.abs(v_ag),
                            'v_bn':np.abs(v_bg),
                            'v_cn':np.abs(v_cg),
                            'v_ng':0.0})
                bus.update({'deg_an':np.angle(v_ag, deg=True),
                            'deg_bn':np.angle(v_bg, deg=True),
                            'deg_cn':np.angle(v_cg, deg=True),
                            'deg_ng':np.angle(0, deg=True)})
                bus.update({'v_ab':np.abs(v_ag-v_bg),
                            'v_bc':np.abs(v_bg-v_cg),
                            'v_ca':np.abs(v_cg-v_ag)})
                bus.update({'p_a':s_a.real,
                            'p_b':s_b.real,
                            'p_c':s_c.real})
                bus.update({'q_a':s_a.imag,
                            'q_b':s_b.imag,
                            'q_c':s_c.imag})
                tup = namedtuple('tup',['v_ag', 'v_bg', 'v_cg'])

                res.update({bus['bus']:tup(v_ag,v_bg,v_cg)})
                
            
            if N_nodes==4:   # if 3 phases + neutral
                v_ag = V_sorted[start_node+0,0]
                v_bg = V_sorted[start_node+1,0]
                v_cg = V_sorted[start_node+2,0]
                v_ng = V_sorted[start_node+3,0]
                i_a = I_sorted[start_node+0,0]
                i_b = I_sorted[start_node+1,0]
                i_c = I_sorted[start_node+2,0]
                i_n = I_sorted[start_node+3,0]  
                
                v_an = v_ag-v_ng
                v_bn = v_bg-v_ng                
                v_cn = v_cg-v_ng
                
                s_a = (v_an)*np.conj(i_a)
                s_b = (v_bn)*np.conj(i_b)
                s_c = (v_cn)*np.conj(i_c)
                bus.update({'v_an':np.abs(v_an),
                            'v_bn':np.abs(v_bn),
                            'v_cn':np.abs(v_cn),
                            'v_ng':np.abs(v_ng)})
                bus.update({'deg_an':np.angle(v_ag-v_ng, deg=True),
                            'deg_bn':np.angle(v_bg-v_ng, deg=True),
                            'deg_cn':np.angle(v_cg-v_ng, deg=True),
                            'deg_ng':np.angle(v_ng, deg=True)})
                bus.update({'v_ab':np.abs(v_ag-v_bg),
                            'v_bc':np.abs(v_bg-v_cg),
                            'v_ca':np.abs(v_cg-v_ag)})
                bus.update({'p_a':s_a.real,
                            'p_b':s_b.real,
                            'p_c':s_c.real})
                bus.update({'q_a':s_a.imag,
                            'q_b':s_b.imag,
                            'q_c':s_c.imag})
    
                start_node += 4

                tup = namedtuple('tup',['v_ag', 'v_bg', 'v_cg', 'v_ng','v_an', 'v_bn', 'v_cn'])

                res.update({bus['bus']:tup(v_ag,v_bg,v_cg,v_ng,v_an,v_bn,v_cn)})

        self.V = np.array(V_sorted).reshape(len(V_sorted),1) 
        self.res = res
        
        return 0 #self.V              
        
        
    def get_i(self):
        '''
		Compute line currents from power flow solution and put values 
		in transformers and lines dictionaries.		
        '''
        I_lines  =self.I_lines
        
        it_single_line = 0
        for trafo in self.transformers:
            
            if 'conductors_j' in trafo: 
                cond_1 = trafo['conductors_j']
            else:
                cond_1 = trafo['conductors_1']
            if 'conductors_k' in trafo: 
                cond_2 = trafo['conductors_k']
            else:
                cond_2 = trafo['conductors_2']  
            
            I_1a = (I_lines[it_single_line,0])
            I_1b = (I_lines[it_single_line+1,0])
            I_1c = (I_lines[it_single_line+2,0])
            I_1n = (I_lines[it_single_line+3,0])
            
            I_2a = (I_lines[it_single_line+cond_1+0,0])
            I_2b = (I_lines[it_single_line+cond_1+1,0])
            I_2c = (I_lines[it_single_line+cond_1+2,0])

            if cond_1>3: I_1n = (I_lines[it_single_line+cond_1+3,0])
            if cond_2>3: I_2n = (I_lines[it_single_line+cond_2+3,0])

            #I_n = (I_lines[it_single_line+3,0])
            if cond_1 <=3:
                I_1n = I_1a+I_1b+I_1c
            if cond_2 <=3:
                I_2n = I_2a+I_2b+I_2c
                
            it_single_line += cond_1 + cond_2
            trafo.update({'i_1a_m':np.abs(I_1a)})
            trafo.update({'i_1b_m':np.abs(I_1b)})
            trafo.update({'i_1c_m':np.abs(I_1c)})
            trafo.update({'i_1n_m':np.abs(I_1n)})
            trafo.update({'i_2a_m':np.abs(I_2a)})
            trafo.update({'i_2b_m':np.abs(I_2b)})
            trafo.update({'i_2c_m':np.abs(I_2c)})
            trafo.update({'i_2n_m':np.abs(I_2n)})
            trafo.update({'deg_1a':np.angle(I_1a, deg=True)})
            trafo.update({'deg_1b':np.angle(I_1b, deg=True)})
            trafo.update({'deg_1c':np.angle(I_1c, deg=True)})
            trafo.update({'deg_1n':np.angle(I_1n, deg=True)})
            trafo.update({'deg_2a':np.angle(I_2a, deg=True)})
            trafo.update({'deg_2b':np.angle(I_2b, deg=True)})
            trafo.update({'deg_2c':np.angle(I_2c, deg=True)})
            trafo.update({'deg_2n':np.angle(I_2n, deg=True)})

                        
        self.I_lines = I_lines
        for line in self.lines:
            if line['type'] == 'z':
                N_conductors = len(line['bus_j_nodes'])
                if N_conductors == 3:
                    I_a = (I_lines[it_single_line,0])
                    I_b = (I_lines[it_single_line+1,0])
                    I_c = (I_lines[it_single_line+2,0])
                    #I_n = (I_lines[it_single_line+3,0])
                    I_n = I_a+I_b+I_c
                    
                    alpha = alpha = np.exp(2.0/3*np.pi*1j)
                    i_z =  1/3*(I_a+I_b+I_c)
                    i_p = 1.0/3.0*(I_a + I_b*alpha + I_c*alpha**2)
                    i_n = 1.0/3.0*(I_a + I_b*alpha**2 + I_c*alpha)                
                    it_single_line += N_conductors
                    line.update({'i_j_a_m':np.abs(I_a)})
                    line.update({'i_j_b_m':np.abs(I_b)})
                    line.update({'i_j_c_m':np.abs(I_c)})
                    line.update({'i_j_n_m':np.abs(I_n)})
                    line.update({'deg_j_a':np.angle(I_a, deg=True)})
                    line.update({'deg_j_b':np.angle(I_b, deg=True)})
                    line.update({'deg_j_c':np.angle(I_c, deg=True)})
                    line.update({'deg_j_n':np.angle(I_n, deg=True)})
                    line.update({'i_k_a_m':np.abs(I_a)})
                    line.update({'i_k_b_m':np.abs(I_b)})
                    line.update({'i_k_c_m':np.abs(I_c)})
                    line.update({'i_k_n_m':np.abs(I_n)})
                    line.update({'deg_k_a':np.angle(I_a, deg=True)})
                    line.update({'deg_k_b':np.angle(I_b, deg=True)})
                    line.update({'deg_k_c':np.angle(I_c, deg=True)})
                    line.update({'deg_k_n':np.angle(I_n, deg=True)})
                    line.update({'i_z':np.abs(i_z)})
                    line.update({'i_p':np.abs(i_p)})
                    line.update({'i_n':np.abs(i_n)})
                if N_conductors == 4:
                    I_a = (I_lines[it_single_line,0])
                    I_b = (I_lines[it_single_line+1,0])
                    I_c = (I_lines[it_single_line+2,0])
                    I_n = (I_lines[it_single_line+3,0])
                    it_single_line += N_conductors
                    line.update({'i_j_a_m':np.abs(I_a)})
                    line.update({'i_j_b_m':np.abs(I_b)})
                    line.update({'i_j_c_m':np.abs(I_c)})
                    line.update({'i_j_n_m':np.abs(I_n)})
                    line.update({'deg_j_a':np.angle(I_a, deg=True)})
                    line.update({'deg_j_b':np.angle(I_b, deg=True)})
                    line.update({'deg_j_c':np.angle(I_c, deg=True)})
                    line.update({'deg_j_n':np.angle(I_n, deg=True)})
                    line.update({'i_k_a_m':np.abs(I_a)})
                    line.update({'i_k_b_m':np.abs(I_b)})
                    line.update({'i_k_c_m':np.abs(I_c)})
                    line.update({'i_k_n_m':np.abs(I_n)})
                    line.update({'deg_k_a':np.angle(I_a, deg=True)})
                    line.update({'deg_k_b':np.angle(I_b, deg=True)})
                    line.update({'deg_k_c':np.angle(I_c, deg=True)})
                    line.update({'deg_k_n':np.angle(I_n, deg=True)})
                    
            if line['type'] == 'pi':
                N_conductors = len(line['bus_j_nodes'])
                if N_conductors == 3:
                    I_j_a = I_lines[it_single_line+0,0]+I_lines[it_single_line+3,0]
                    I_j_b = I_lines[it_single_line+1,0]+I_lines[it_single_line+4,0]
                    I_j_c = I_lines[it_single_line+2,0]+I_lines[it_single_line+5,0]
                    I_k_a = I_lines[it_single_line+0,0]-I_lines[it_single_line+6,0]
                    I_k_b = I_lines[it_single_line+1,0]-I_lines[it_single_line+7,0]
                    I_k_c = I_lines[it_single_line+2,0]-I_lines[it_single_line+8,0]
                    
                    #I_n = (I_lines[it_single_line+3,0])
                    I_j_n = I_j_a+I_j_b+I_j_c
                    I_k_n = I_k_a+I_k_b+I_k_c
                    
                    alpha = alpha = np.exp(2.0/3*np.pi*1j)
                    i_z =  1/3*(I_j_a+I_j_b+I_j_c)
                    i_p = 1.0/3.0*(I_j_a + I_j_b*alpha + I_j_c*alpha**2)
                    i_n = 1.0/3.0*(I_j_a + I_j_b*alpha**2 + I_j_c*alpha)                
                    it_single_line += N_conductors*3
                    line.update({'i_j_a_m':np.abs(I_j_a)})
                    line.update({'i_j_b_m':np.abs(I_j_b)})
                    line.update({'i_j_c_m':np.abs(I_j_c)})
                    line.update({'i_j_n_m':np.abs(I_j_n)})
                    line.update({'deg_j_a':np.angle(I_j_a, deg=True)})
                    line.update({'deg_j_b':np.angle(I_j_b, deg=True)})
                    line.update({'deg_j_c':np.angle(I_j_c, deg=True)})
                    line.update({'deg_j_n':np.angle(I_j_n, deg=True)})
                    line.update({'i_k_a_m':np.abs(I_k_a)})
                    line.update({'i_k_b_m':np.abs(I_k_b)})
                    line.update({'i_k_c_m':np.abs(I_k_c)})
                    line.update({'i_k_n_m':np.abs(I_k_n)})
                    line.update({'deg_k_a':np.angle(I_k_a, deg=True)})
                    line.update({'deg_k_b':np.angle(I_k_b, deg=True)})
                    line.update({'deg_k_c':np.angle(I_k_c, deg=True)})
                    line.update({'deg_k_n':np.angle(I_k_n, deg=True)})
                    line.update({'i_z':np.abs(i_z)})
                    line.update({'i_p':np.abs(i_p)})
                    line.update({'i_n':np.abs(i_n)})
                if N_conductors == 4:
                    I_j_a = I_lines[it_single_line+0,0]+I_lines[it_single_line+3,0]
                    I_j_b = I_lines[it_single_line+1,0]+I_lines[it_single_line+4,0]
                    I_j_c = I_lines[it_single_line+2,0]+I_lines[it_single_line+5,0]
                    I_k_a = I_lines[it_single_line+0,0]-I_lines[it_single_line+6,0]
                    I_k_b = I_lines[it_single_line+1,0]-I_lines[it_single_line+7,0]
                    I_k_c = I_lines[it_single_line+2,0]-I_lines[it_single_line+8,0]
                    I_j_n = I_lines[it_single_line+3,0]
                    I_k_n = I_lines[it_single_line+3,0]
                    
                    #I_n = (I_lines[it_single_line+3,0])
                    I_j_n = I_j_a+I_j_b+I_j_c
                    I_k_n = I_k_a+I_k_b+I_k_c
                    
                    alpha = alpha = np.exp(2.0/3*np.pi*1j)
                    i_z =  1/3*(I_j_a+I_j_b+I_j_c)
                    i_p = 1.0/3.0*(I_j_a + I_j_b*alpha + I_j_c*alpha**2)
                    i_n = 1.0/3.0*(I_j_a + I_j_b*alpha**2 + I_j_c*alpha)                
                    it_single_line += N_conductors*3
                    line.update({'i_j_a_m':np.abs(I_j_a)})
                    line.update({'i_j_b_m':np.abs(I_j_b)})
                    line.update({'i_j_c_m':np.abs(I_j_c)})
                    line.update({'i_j_n_m':np.abs(I_j_n)})
                    line.update({'deg_j_a':np.angle(I_j_a, deg=True)})
                    line.update({'deg_j_b':np.angle(I_j_b, deg=True)})
                    line.update({'deg_j_c':np.angle(I_j_c, deg=True)})
                    line.update({'deg_j_n':np.angle(I_j_n, deg=True)})
                    line.update({'i_k_a_m':np.abs(I_k_a)})
                    line.update({'i_k_b_m':np.abs(I_k_b)})
                    line.update({'i_k_c_m':np.abs(I_k_c)})
                    line.update({'i_k_n_m':np.abs(I_k_n)})
                    line.update({'deg_k_a':np.angle(I_k_a, deg=True)})
                    line.update({'deg_k_b':np.angle(I_k_b, deg=True)})
                    line.update({'deg_k_c':np.angle(I_k_c, deg=True)})
                    line.update({'deg_k_n':np.angle(I_k_n, deg=True)})  
                    

    def bokeh_tools(self):

        
        self.bus_tooltip = '''
            <div>
            bus_id = @bus_id &nbsp &nbsp |  u<sub>avg</sub>= @u_avg_pu pu |  u<sub>unb</sub>= @v_unb %
            <table border="1">
                <tr>
                <td>v<sub>an</sub> =  @v_an  &ang; @deg_an V </td> <td> S<sub>a</sub> = @p_a + j@q_a kVA</td>
                </tr>
                      <tr>
                      <td> </td> <td>v<sub>ab</sub>= @v_ab V</td>
                      </tr>
                <tr>
                <td>v<sub>bn</sub> = @v_bn &ang; @deg_bn V </td><td> S<sub>b</sub> = @p_b + j@q_b kVA</td>
                </tr>
                      <tr>
                      <td> </td><td>v<sub>bc</sub>= @v_bc V</td>
                      </tr>
                <tr>
                <td>v<sub>cn</sub>  = @v_cn &ang; @deg_cn V </td>  <td>S<sub>c</sub> = @p_c + j@q_c kVA </td>
                </tr> 
                    <tr>
                     <td> </td> <td>v<sub>ca</sub>= @v_ca V</td>
                    </tr>
               <tr>
                <td>v<sub>ng</sub>    = @v_ng &ang; @deg_ng V</td>  <td>S<sub>abc</sub> = @p_abc + j@q_abc kVA </td>
              </tr>
            </table>
            </div>
            '''
            
        x = [item['pos_x'] for item in self.buses]
        y = [item['pos_y'] for item in self.buses]
        bus_id = [item['bus'] for item in self.buses]
        v_an = ['{:2.2f}'.format(float(item['v_an'])) for item in self.buses]
        v_bn = ['{:2.2f}'.format(float(item['v_bn'])) for item in self.buses]
        v_cn = ['{:2.2f}'.format(float(item['v_cn'])) for item in self.buses]
        v_ng = ['{:2.2f}'.format(float(item['v_ng'])) for item in self.buses]
        sqrt3=np.sqrt(3)
        
        u_avg_pu = []
        v_unb = []
        for item in  self.buses:
            V_base = float(item['U_kV'])*1000.0/sqrt3
            v_an_float = float(item['v_an'])
            v_bn_float = float(item['v_bn'])
            v_cn_float = float(item['v_cn'])
            v_ng_float = float(item['v_ng'])
            
            v_abc = np.array([v_an_float,v_bn_float,v_cn_float])
            v_avg = np.average(v_abc)
            unb = float(np.max(np.abs(v_abc-v_avg))/v_avg)
            v_avg_pu = float(v_avg/V_base)
            u_avg_pu += ['{:2.3f}'.format(v_avg_pu)]
            v_unb += ['{:2.1f}'.format(unb*100)]
            
        
        v_an_pu = ['{:2.4f}'.format(float(item['v_an'])/float(item['U_kV'])/1000.0*sqrt3) for item in self.buses]
        v_bn_pu = ['{:2.4f}'.format(float(item['v_bn'])/float(item['U_kV'])/1000.0*sqrt3) for item in self.buses]
        v_cn_pu = ['{:2.4f}'.format(float(item['v_cn'])/float(item['U_kV'])/1000.0*sqrt3) for item in self.buses]
        v_ng_pu = ['{:2.4f}'.format(float(item['v_ng'])/float(item['U_kV'])/1000.0*sqrt3) for item in self.buses]
        
        deg_an = ['{:2.2f}'.format(float(item['deg_an'])) for item in self.buses]
        deg_bn = ['{:2.2f}'.format(float(item['deg_bn'])) for item in self.buses]
        deg_cn = ['{:2.2f}'.format(float(item['deg_cn'])) for item in self.buses]
        deg_ng = ['{:2.2f}'.format(float(item['deg_ng'])) for item in self.buses]
        v_ab = [item['v_ab'] for item in self.buses]
        v_bc = [item['v_bc'] for item in self.buses]
        v_ca = [item['v_ca'] for item in self.buses]
        
        p_a = ['{:2.2f}'.format(float(item['p_a']/1000)) for item in self.buses]
        p_b = ['{:2.2f}'.format(float(item['p_b']/1000)) for item in self.buses]
        p_c = ['{:2.2f}'.format(float(item['p_c']/1000)) for item in self.buses]
        q_a = ['{:2.2f}'.format(float(item['q_a']/1000)) for item in self.buses]
        q_b = ['{:2.2f}'.format(float(item['q_b']/1000)) for item in self.buses]
        q_c = ['{:2.2f}'.format(float(item['q_c']/1000)) for item in self.buses]   
        p_abc = ['{:2.2f}'.format(float((item['p_a'] +item['p_b']+item['p_c'])/1000)) for item in self.buses] 
        q_abc = ['{:2.2f}'.format(float((item['q_a'] +item['q_b']+item['q_c'])/1000)) for item in self.buses]
        s_radio = []
        s_color = []
        
        for item in self.buses:
            p_total = item['p_a'] + item['p_b'] + item['p_c']
            q_total = item['q_a'] + item['q_b'] + item['q_c']            
            s_total = np.abs(p_total + 1j*q_total)
            scale = self.s_radio_scale
            s_scaled = abs(np.sqrt(s_total))*scale
            if s_scaled<self.s_radio_min :
                s_scaled = self.s_radio_min 
            if s_scaled>self.s_radio_max:
                s_scaled = self.s_radio_max
            s_radio += [s_scaled]
            if p_total>0.0:
                s_color += ['red']
            if p_total<0.0:
                s_color += ['green']
            if p_total==0.0:
                s_color += ['blue']
                                
        self.bus_data = dict(x=x, y=y, bus_id=bus_id,  u_avg_pu=u_avg_pu,  v_unb=v_unb,
                             v_an=v_an, v_bn=v_bn, v_cn=v_cn, v_ng=v_ng, 
                             v_an_pu=v_an_pu, v_bn_pu=v_bn_pu, v_cn_pu=v_cn_pu, 
                             deg_an=deg_an, deg_bn=deg_bn, deg_cn=deg_cn, 
                             deg_ng=deg_ng,v_ab=v_ab,v_bc=v_bc,v_ca=v_ca,
                             p_a=p_a,p_b=p_b,p_c=p_c,
                             q_a=q_a,q_b=q_b,q_c=q_c,
                             p_abc=p_abc,q_abc=q_abc,
                             s_radio=s_radio, s_color=s_color)
        
        self.line_tooltip = '''
            <div>
            line id = @line_id 
            <table border="1">
                <tr>
                <td>I<sub>a</sub> =  @i_a_m &ang; @deg_a A</td>
                </tr>
                <tr>
                <td>I<sub>b</sub> =  @i_b_m &ang; @deg_b A</td>
                </tr>
                <tr>
                <td>I<sub>c</sub> =  @i_c_m &ang; @deg_c A</td>
                </tr>
                <tr>
                <td>I<sub>n</sub> =  @i_n_m &ang; @deg_n A</td>
                </tr>
            </table>            
            </div>
            '''


        self.line_tooltip = '''
            <div>
            line id = @line_id 
            <table border="5">
                <tr >
                    <td>I<sub>ja</sub> =  @i_j_a_m &ang; @deg_j_a </td>
                    <td>I<sub>ka</sub> =  @i_k_a_m &ang; @deg_k_a </td>
                </tr>
                <tr>
                    <td >I<sub>jb</sub> =  @i_j_b_m &ang; @deg_j_b </td>
                    <td >I<sub>kb</sub> =  @i_k_b_m &ang; @deg_k_b </td>
                </tr>
                <tr>
                    <td >I<sub>jc</sub> =  @i_j_c_m &ang; @deg_j_c </td>
                    <td >I<sub>kc</sub> =  @i_k_c_m &ang; @deg_k_c </td>
                </tr>
                <tr>
                    <td >I<sub>jn</sub> =  @i_j_n_m &ang; @deg_j_n </td>
                    <td >I<sub>kn</sub> =  @i_k_n_m &ang; @deg_k_n </td>
                </tr>
            </table>            
            </div>
            '''

            
        bus_id_to_x = dict(zip(bus_id,x))
        bus_id_to_y = dict(zip(bus_id,y))
        
        x_j = [bus_id_to_x[item['bus_j']] for item in self.lines]
        y_j = [bus_id_to_y[item['bus_j']] for item in self.lines]
        x_k = [bus_id_to_x[item['bus_k']] for item in self.lines]
        y_k = [bus_id_to_y[item['bus_k']] for item in self.lines]
        
        x_s = []
        y_s = []
        for line in self.lines:
            x_s += [[ bus_id_to_x[line['bus_j']] , bus_id_to_x[line['bus_k']]]]
            y_s += [[ bus_id_to_y[line['bus_j']] , bus_id_to_y[line['bus_k']]]]
            
        i_j_a_m = [item['i_j_a_m'] for item in self.lines]
        i_j_b_m = [item['i_j_b_m'] for item in self.lines]
        i_j_c_m = [item['i_j_c_m'] for item in self.lines]
        i_j_n_m = [item['i_j_n_m'] for item in self.lines]
        i_k_a_m = [item['i_k_a_m'] for item in self.lines]
        i_k_b_m = [item['i_k_b_m'] for item in self.lines]
        i_k_c_m = [item['i_k_c_m'] for item in self.lines]
        i_k_n_m = [item['i_k_n_m'] for item in self.lines]
        
        deg_j_a = [item['deg_j_a'] for item in self.lines]
        deg_j_b = [item['deg_j_b'] for item in self.lines]
        deg_j_c = [item['deg_j_c'] for item in self.lines]
        deg_j_n = [item['deg_j_n'] for item in self.lines]
        deg_k_a = [item['deg_k_a'] for item in self.lines]
        deg_k_b = [item['deg_k_b'] for item in self.lines]
        deg_k_c = [item['deg_k_c'] for item in self.lines]
        deg_k_n = [item['deg_k_n'] for item in self.lines] 
        
        line_id = ['{:s}-{:s}'.format(item['bus_j'],item['bus_k']) for item in self.lines]
#        self.line_data = dict(x_j=x_j, x_k=x_k, y_j=y_j, y_k=y_k, line_id=line_id,
#                             i_a_m=i_a_m)
        self.line_data = dict(x_s=x_s, y_s=y_s, line_id=line_id,
                             i_j_a_m=i_j_a_m, i_j_b_m=i_j_b_m, i_j_c_m=i_j_c_m, i_j_n_m=i_j_n_m,
                             i_k_a_m=i_k_a_m, i_k_b_m=i_k_b_m, i_k_c_m=i_k_c_m, i_k_n_m=i_k_n_m,                             
                             deg_j_a=deg_j_a, deg_j_b=deg_j_b, deg_j_c=deg_j_c, deg_j_n=deg_j_n,
                             deg_k_a=deg_k_a, deg_k_b=deg_k_b, deg_k_c=deg_k_c, deg_k_n=deg_k_n)


        
        self.transformer_tooltip = '''
            <div>
            transformer id = @trafo_id  
            <table border="5">
                <tr >
                    <td>I<sub>1a</sub> =  @i_1a_m &ang; @deg_1a </td>
                    <td>I<sub>2a</sub> =  @i_2a_m &ang; @deg_2a </td>
                </tr>
                <tr>
                    <td >I<sub>1b</sub> =  @i_1b_m &ang; @deg_1b </td>
                    <td >I<sub>2b</sub> =  @i_2b_m &ang; @deg_2b </td>
                </tr>
                <tr>
                    <td >I<sub>1c</sub> =  @i_1c_m &ang; @deg_1c </td>
                    <td >I<sub>2c</sub> =  @i_2c_m &ang; @deg_2c </td>
                </tr>
                <tr>
                    <td >I<sub>1n</sub> =  @i_1n_m &ang; @deg_1n </td>
                    <td >I<sub>2n</sub> =  @i_2n_m &ang; @deg_2n </td>
                </tr>
            </table>            
            </div>
            '''
            
        bus_id_to_x = dict(zip(bus_id,x))
        bus_id_to_y = dict(zip(bus_id,y))
        
        x_j = [bus_id_to_x[item['bus_j']] for item in self.transformers]
        y_j = [bus_id_to_y[item['bus_j']] for item in self.transformers]
        x_k = [bus_id_to_x[item['bus_k']] for item in self.transformers]
        y_k = [bus_id_to_y[item['bus_k']] for item in self.transformers]
        
        x_s = []
        y_s = []
        for line in self.transformers:
            x_s += [[ bus_id_to_x[line['bus_j']] , bus_id_to_x[line['bus_k']]]]
            y_s += [[ bus_id_to_y[line['bus_j']] , bus_id_to_y[line['bus_k']]]]
            
        i_1a_m = [item['i_1a_m'] for item in self.transformers]
        i_1b_m = [item['i_1b_m'] for item in self.transformers]
        i_1c_m = [item['i_1c_m'] for item in self.transformers]
        i_1n_m = [item['i_1n_m'] for item in self.transformers]

        i_2a_m = [item['i_2a_m'] for item in self.transformers]
        i_2b_m = [item['i_2b_m'] for item in self.transformers]
        i_2c_m = [item['i_2c_m'] for item in self.transformers]
        i_2n_m = [item['i_2n_m'] for item in self.transformers]
        
        deg_1a = [item['deg_1a'] for item in self.transformers]
        deg_1b = [item['deg_1b'] for item in self.transformers]
        deg_1c = [item['deg_1c'] for item in self.transformers]
        deg_1n = [item['deg_1n'] for item in self.transformers]    
        
        deg_2a = [item['deg_2a'] for item in self.transformers]
        deg_2b = [item['deg_2b'] for item in self.transformers]
        deg_2c = [item['deg_2c'] for item in self.transformers]
        deg_2n = [item['deg_2n'] for item in self.transformers]  
        
        trafo_id = ['{:s}-{:s}'.format(item['bus_j'],item['bus_k']) for item in self.transformers]
#        self.line_data = dict(x_j=x_j, x_k=x_k, y_j=y_j, y_k=y_k, line_id=line_id,
#                             i_a_m=i_a_m)
        self.transformer_data = dict(x_s=x_s, y_s=y_s, trafo_id=trafo_id,
                                     i_1a_m=i_1a_m, i_1b_m=i_1b_m, i_1c_m=i_1c_m, i_1n_m=i_1n_m,
                                     deg_1a=deg_1a, deg_1b=deg_1b, deg_1c=deg_1c, deg_1n=deg_1n,
                                     i_2a_m=i_2a_m, i_2b_m=i_2b_m, i_2c_m=i_2c_m, i_2n_m=i_2n_m,
                                     deg_2a=deg_2a, deg_2b=deg_2b, deg_2c=deg_2c, deg_2n=deg_2n)
        
        return self.bus_data

from bokeh.io import output_notebook, show
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, HoverTool
from bokeh.io import push_notebook
from bokeh.resources import INLINE
output_notebook(INLINE)



def plot_results(grid):
    #grid.bokeh_tools()
    
    p = figure(width=600, height=400,
               title='Results')
    
    # trafos:
    source = ColumnDataSource(grid.transformer_data)
    trafo = p.multi_line(source=source, xs='x_s', ys='y_s', color="green", alpha=0.5, line_width=5)
    
    # lines:
    source = ColumnDataSource(grid.line_data)
    lin = p.multi_line(source=source, xs='x_s', ys='y_s', color="red", alpha=0.5, line_width=5)
    
    # buses:
    source = ColumnDataSource(grid.bus_data)
    cr = p.circle(source=source, x='x', y='y', size=15, color="navy", alpha=0.5)
    
    p.add_tools(HoverTool(renderers=[trafo], tooltips=grid.transformer_tooltip))
    p.add_tools(HoverTool(renderers=[lin], tooltips=grid.line_tooltip))
    p.add_tools(HoverTool(renderers=[cr], tooltips=grid.bus_tooltip))
    show(p)
    
    return p


def get_flow(grid_obj,bus_j,bus_k,mode='total',model='pydgrid_pydae'):
    if model == 'pydgrid_pydae':
        v_a   = grid_obj.get_values(f'v_{bus_j}_a_r') + 1j* grid_obj.get_values(f'v_{bus_j}_a_i')
        i_l_a = grid_obj.get_values(f'i_l_{bus_j}_{bus_k}_a_r') + 1j* grid_obj.get_values(f'i_l_{bus_j}_{bus_k}_a_i')
        v_b   = grid_obj.get_values(f'v_{bus_j}_b_r') + 1j* grid_obj.get_values(f'v_{bus_j}_b_i')
        i_l_b = grid_obj.get_values(f'i_l_{bus_j}_{bus_k}_b_r') + 1j* grid_obj.get_values(f'i_l_{bus_j}_{bus_k}_b_i')
        v_c   = grid_obj.get_values(f'v_{bus_j}_c_r') + 1j* grid_obj.get_values(f'v_{bus_j}_c_i')
        i_l_c = grid_obj.get_values(f'i_l_{bus_j}_{bus_k}_c_r') + 1j* grid_obj.get_values(f'i_l_{bus_j}_{bus_k}_c_i')
        s_a = v_a*np.conj(i_l_a)
        s_b = v_b*np.conj(i_l_b)
        s_c = v_c*np.conj(i_l_c)

        if mode == 'total':
            s_t = s_a + s_b + s_c
            return s_t
        if mode == 'abc':
            return s_a,s_b,s_c
        
def set_voltage(grid_obj,bus_name,voltage,phase):
    '''
    Set new power to a grid feeder.

    Parameters
    ----------
    grid_obj : object of pydgrid.grid class
    bus_name : string
        name of the grid feeder bus.
    voltage : real escalar
        phase-phase RMS voltage magnitude
    phase : real escalar.
        phase angle in degree.

    Returns
    -------
    None.

    '''

 
    v_a = voltage/np.sqrt(3)*np.exp(1j*np.deg2rad(phase))
    v_b = voltage/np.sqrt(3)*np.exp(1j*np.deg2rad(phase-240))
    v_c = voltage/np.sqrt(3)*np.exp(1j*np.deg2rad(phase-120))
    grid_obj.set_value(f'v_{bus_name}_a_r',v_a.real)
    grid_obj.set_value(f'v_{bus_name}_a_i',v_a.imag)
    grid_obj.set_value(f'v_{bus_name}_b_r',v_b.real)
    grid_obj.set_value(f'v_{bus_name}_b_i',v_b.imag)
    grid_obj.set_value(f'v_{bus_name}_c_r',v_c.real)
    grid_obj.set_value(f'v_{bus_name}_c_i',v_c.imag)
 
def set_voltages(grid_obj,bus_name,voltages,phases):
    '''
    Set new power to a grid feeder.

    Parameters
    ----------
    grid_obj : object of pydgrid.grid class
    bus_name : string
        name of the grid feeder bus.
    voltage : real escalar
        phase-phase RMS voltage magnitude
    phase : real escalar.
        phase angle in degree.

    Returns
    -------
    None.

    '''

    if isinstance(phases, list):
        v_a = voltages[0]*np.exp(1j*np.deg2rad(phases[0]))
        v_b = voltages[1]*np.exp(1j*np.deg2rad(phases[1]))
        v_c = voltages[2]*np.exp(1j*np.deg2rad(phases[2]))
    else:
        v_a = voltages[0]*np.exp(1j*np.deg2rad(phases))
        v_b = voltages[1]*np.exp(1j*np.deg2rad(phases-240))
        v_c = voltages[2]*np.exp(1j*np.deg2rad(phases-120))    
    
    grid_obj.set_value(f'v_{bus_name}_a_r',v_a.real)
    grid_obj.set_value(f'v_{bus_name}_a_i',v_a.imag)
    grid_obj.set_value(f'v_{bus_name}_b_r',v_b.real)
    grid_obj.set_value(f'v_{bus_name}_b_i',v_b.imag)
    grid_obj.set_value(f'v_{bus_name}_c_r',v_c.real)
    grid_obj.set_value(f'v_{bus_name}_c_i',v_c.imag)
    
    
def phasor2inst(grid_obj,bus_name,magnitude='v',to_bus='',phases=['a','b','c'],Freq = 50,Dt=1e-4):
    
    omega = 2*np.pi*Freq
    out = []
    
    if magnitude == 'v':
        for ph in phases:
            Times = np.arange(0.0,grid_obj.T[-1,0],Dt)
            R = grid_obj.get_values(f'{magnitude}_{bus_name}_{ph}_r') 
            I = grid_obj.get_values(f'{magnitude}_{bus_name}_{ph}_i')
            R_ = np.interp(Times,grid_obj.T[:,0],R)
            I_ = np.interp(Times,grid_obj.T[:,0],I)
            R_I = R_ + 1j*I_
            cplx = np.sqrt(2)*np.exp(1j*omega*Times)*R_I
            out += [cplx.real]

    if magnitude == 'iline':
        for ph in phases:
            Times = np.arange(0.0,grid_obj.T[-1,0],Dt)
            R = grid_obj.get_values(f'i_l_{bus_name}_{to_bus}_{ph}_r') 
            I = grid_obj.get_values(f'i_l_{bus_name}_{to_bus}_{ph}_i')
            R_ = np.interp(Times,grid_obj.T[:,0],R)
            I_ = np.interp(Times,grid_obj.T[:,0],I)
            R_I = R_ + 1j*I_
            cplx = np.sqrt(2)*np.exp(1j*omega*Times)*R_I
            out += [cplx.real]

    return Times,out


def get_voltage(grid_obj,bus_name,output='V_an_m'):
    '''
    Get voltage module of a bus.

    Parameters
    ----------
    grid_obj : object of pydae class
    bus_name : string
        name of the bus.
    output : string
         v_an: a phase to neutral voltage phasor (V).
         v_an_m: a phase to neutral RMS voltage (V).
         v_abcn_m: a,b and c phases to neutral voltage phasors (V).
        
    Returns
    -------
    phase-ground voltage module (V).

    '''
    
    a20 = {'a':0,'b':1,'c':2,'n':3}

    if output in ['v_an','v_bn','v_cn']:
        v_sub = f'v_{bus_name}_{output[-2]}'
        v = grid_obj.get_value(f'{v_sub}_r') + 1j* grid_obj.get_value(f'{v_sub}_i')
        return v
    
    if output in ['V_an_m','V_bn_m','V_cn_m']:
        phase = a20[output[-4]]
        V_sub = f'V_{bus_name}_{phase}'
        
        V = grid_obj.get_value(f'{V_sub}_r') + 1j* grid_obj.get_value(f'{V_sub}_i')
        return np.abs(V)
    
    if output in ['v_abcn']:
        v_list = []
        for ph in ['0','1','2']:
            v_sub = f'v_{bus_name}_{ph}'
            v = grid_obj.get_value(f'{v_sub}_r') + 1j* grid_obj.get_value(f'{v_sub}_i')
            v_list += [v]
        return np.array(v_list).reshape((3,1))

def get_voltages(grid_obj,bus_name,output='V_an_m'):
    '''
    Get array of time simulations with voltage module of a bus.

    Parameters
    ----------
    grid_obj : object of pydae class
    bus_name : string
        name of the bus.
    output : string
         v_an: a phase to neutral voltage phasor (V).
         v_an_m: a phase to neutral RMS voltage (V).
         v_abcn_m: a,b and c phases to neutral voltage phasors (V).
        
    Returns
    -------
    phase-ground voltage module (V).

    '''
    
    a20 = {'a':0,'b':1,'c':2,'n':3}

    if output in ['v_an','v_bn','v_cn']:
        v_sub = f'v_{bus_name}_{output[-2]}'
        v = grid_obj.get_value(f'{v_sub}_r') + 1j* grid_obj.get_value(f'{v_sub}_i')
        return v
    
    if output in ['V_an_m','V_bn_m','V_cn_m']:
        phase = a20[output[-4]]
        V_sub = f'V_{bus_name}_{phase}'
        
        V = grid_obj.get_values(f'{V_sub}_r') + 1j* grid_obj.get_values(f'{V_sub}_i')
        return np.abs(V)
    
    if output in ['v_abcn']:
        v_list = []
        for ph in ['0','1','2']:
            v_sub = f'v_{bus_name}_{ph}'
            v = grid_obj.get_values(f'{v_sub}_r') + 1j* grid_obj.get_values(f'{v_sub}_i')
            v_list += [v]
        return np.array(v_list).reshape((3,1))



    


@numba.njit(cache=True)
def abc2pq(times,v_a,v_b,v_c,i_a,i_b,i_c,omega=2*np.pi*50,theta_0=0.0):
    N_t = len(times)
    Dt = times[1]-times[0] 
    p = np.zeros((N_t,1))
    q = np.zeros((N_t,1))
    for it in range(len(times)):

        theta = Dt*it*omega + theta_0
        v_abc = np.array([[v_a[it]],[v_b[it]],[v_c[it]]])
        T_p = 2.0/3.0*np.array([[ np.cos(theta), np.cos(theta-2.0/3.0*np.pi), np.cos(theta+2.0/3.0*np.pi)],
                                [-np.sin(theta),-np.sin(theta-2.0/3.0*np.pi),-np.sin(theta+2.0/3.0*np.pi)]])

        dq=T_p@v_abc;
        
        v_d = dq[0]
        v_q = dq[1]

        theta = Dt*it*omega + theta_0
        i_abc = np.array([[i_a[it]],[i_b[it]],[i_c[it]]])
        T_p = 2.0/3.0*np.array([[ np.cos(theta), np.cos(theta-2.0/3.0*np.pi), np.cos(theta+2.0/3.0*np.pi)],
                                [-np.sin(theta),-np.sin(theta-2.0/3.0*np.pi),-np.sin(theta+2.0/3.0*np.pi)]])

        i_dq=T_p@i_abc;
        
        i_d = i_dq[0]
        i_q = i_dq[1]
    
        p[it] = 3/2*(v_d*i_d + v_q*i_q)
        q[it] = 3/2*(v_d*i_q - v_q*i_d)
    return p,q


@numba.njit(cache=True)
def abc2dq(times,v_a,v_b,v_c,i_a,i_b,i_c,omega=2*np.pi*50,theta_0=0.0,K_p=0.1,K_i=20.0,T_f=20.0e-3):

    N_t = len(times)
    Dt = times[1]-times[0] 

    v_d = np.zeros((N_t,1))
    v_q = np.zeros((N_t,1))
    i_d = np.zeros((N_t,1))
    i_q = np.zeros((N_t,1))
    p = np.zeros((N_t,1))
    q = np.zeros((N_t,1))
    
    theta = 0.0
    xi = 0.0
    theta_pll = np.zeros((N_t,1)) 
    omega_pll = np.zeros((N_t,1)) 
    dq = np.zeros((2,1)) 
    idx = np.argmax(times>0.08)

    theta_pll[0,0] = theta_0
    #omega_pll = np.zeros((N_t,1)) 
    
    
    for it in range(len(times)-1):

        theta = theta_pll[it,0]
        v_abc = np.array([[v_a[it]],[v_b[it]],[v_c[it]]])
        i_abc = np.array([[i_a[it]],[i_b[it]],[i_c[it]]])
        T_p = 2.0/3.0*np.array([[ np.cos(theta), np.cos(theta-2.0/3.0*np.pi), np.cos(theta+2.0/3.0*np.pi)],
                                [-np.sin(theta),-np.sin(theta-2.0/3.0*np.pi),-np.sin(theta+2.0/3.0*np.pi)]])
    

        v_dq = T_p@v_abc;
        i_dq = T_p@i_abc;
        
        v_d[it+1,0] = v_d[it,0] + Dt/T_f*(v_dq[0,0] - v_d[it,0])
        v_q[it+1,0] = v_q[it,0] + Dt/T_f*(v_dq[1,0] - v_q[it,0])
        i_d[it+1,0] = i_d[it,0] + Dt/T_f*(i_dq[0,0] - i_d[it,0])
        i_q[it+1,0] = i_q[it,0] + Dt/T_f*(i_dq[1,0] - i_q[it,0])
        
        p[it] = 3/2*(v_d[it+1,0] *i_d[it+1,0]  + v_q[it+1,0] *i_q[it+1,0] )
        q[it] = 3/2*(v_q[it+1,0] *i_d[it+1,0]  - v_d[it+1,0] *i_q[it+1,0] )
        
        xi += Dt*v_dq[0,0]
        omega_pll[it,0] = K_p * v_dq[0,0] + K_i * xi + omega
        
        theta_pll[it+1,0] += theta_pll[it,0] + Dt*(omega_pll[it,0])

    omega_pll[it+1,0] = K_p * v_dq[0,0] + K_i * xi + omega
    return theta_pll,omega_pll,v_d,v_q,i_d,i_q,p,q


def change_line(system,bus_j,bus_k, *args,**kwagrs):
    line = kwagrs
    S_base = system.get_value('S_base')
    
    line_name = f"{bus_j}_{bus_k}"
    if 'X_pu' in line:
        if 'S_mva' in line: S_line = 1e6*line['S_mva']
        R = line['R_pu']*S_base/S_line  # in pu of the system base
        X = line['X_pu']*S_base/S_line  # in pu of the system base
    if 'X' in line:
        U_base = system.get_value(f'U_{bus_j}_n') 
        Z_base = U_base**2/S_base
        R = line['R']/Z_base  # in pu of the system base
        X = line['X']/Z_base  # in pu of the system base
    if 'X_km' in line:
        U_base = system.get_value(f'U_{bus_j}_n')
        Z_base = U_base**2/S_base
        R = line['R_km']*line['km']/Z_base  # in pu of the system base
        X = line['X_km']*line['km']/Z_base  # in pu of the system base
    if 'Bs_km' in line:
        U_base = system.get_value(f'U_{bus_j}_n')
        Z_base = U_base**2/S_base
        print('U_base',U_base,'Z_base',Z_base)
        Y_base = 1.0/Z_base
        Bs = line['Bs_km']*line['km']/Y_base  # in pu of the system base
        bs = Bs
        system.set_value(f'bs_{line_name}',bs)
        print(bs)
    G =  R/(R**2+X**2)
    B = -X/(R**2+X**2)
    system.set_value(f"g_{line_name}",G)
    system.set_value(f"b_{line_name}",B)
    
    
def get_line_i(system,bus_from,bus_to,U_kV=66e3):
    
    
    if f"b_{bus_from}_{bus_to}" in system.params_list: 
        bus_j = bus_from
        bus_k = bus_to
        current_direction = 1.0
    elif f"b_{bus_to}_{bus_from}" in system.params_list: 
        bus_j = bus_to
        bus_k = bus_from
        current_direction = -1.0
    else: 
        print(f'No line from {bus_from} to {bus_to}')
        return
    
    line_name = f"{bus_j}_{bus_k}"
    V_j_m = system.get_value(f"V_{bus_j}")
    theta_j = system.get_value(f"theta_{bus_j}")
    V_k_m = system.get_value(f"V_{bus_k}") 
    theta_k = system.get_value(f"theta_{bus_k}")
    
    V_j = V_j_m*np.exp(1j*theta_j)
    V_k = V_k_m*np.exp(1j*theta_k)
    
    Y_jk = system.get_value(f"g_{line_name}") + 1j*system.get_value(f"b_{line_name}")
    S_base = system.get_value('S_base')
    U_base = system.get_value(f"U_{bus_j}_n")
    I_jk_pu = current_direction*Y_jk*(V_j - V_k)
    I_base = S_base/(np.sqrt(3)*U_base)
    I_jk = I_jk_pu*I_base
    
    return I_jk

def get_line_s(system,bus_from,bus_to,U_kV=66e3):
    
    
    if f"b_{bus_from}_{bus_to}" in system.params_list: 
        bus_j = bus_from
        bus_k = bus_to
        current_direction = 1.0
    elif f"b_{bus_to}_{bus_from}" in system.params_list: 
        bus_j = bus_to
        bus_k = bus_from
        current_direction = -1.0
    else: 
        print(f'No line from {bus_from} to {bus_to}')
        return
    
    line_name = f"{bus_j}_{bus_k}"
    V_j_m = system.get_value(f"V_{bus_j}")
    theta_j = system.get_value(f"theta_{bus_j}")
    V_k_m = system.get_value(f"V_{bus_k}") 
    theta_k = system.get_value(f"theta_{bus_k}")
    
    V_j = V_j_m*np.exp(1j*theta_j)
    V_k = V_k_m*np.exp(1j*theta_k)
    
    Y_jk = system.get_value(f"g_{line_name}") + 1j*system.get_value(f"b_{line_name}")
    S_base = system.get_value('S_base')
    U_base = system.get_value(f"U_{bus_j}_n")
    I_jk_pu = current_direction*Y_jk*(V_j - V_k)
    I_base = S_base/(np.sqrt(3)*U_base)
    I_jk = I_jk_pu*I_base
    S_jk_pu = V_j*np.conj(I_jk_pu)
    S_jk = S_base*S_jk_pu
    
    return S_jk


def set_powers(grid_obj,bus_name,s_cplx,mode='urisi_3ph'):
    '''
    Function for simplifying the power setting.

    Parameters
    ----------
    grid_obj : pydae object
        pydae object with grid of type pydgrid.
    bus_name : string
        name of the bus to xhange the power.
    s_cplx : TYPE
        complex power (negative for load, positive for generated).

    Returns
    -------
    None.

    '''
    
    if mode == 'urisi_3ph':
        p = s_cplx.real
        q = s_cplx.imag
        for ph in ['a','b','c']:
            grid_obj.set_value(f'p_{bus_name}_{ph}',p/3)
            grid_obj.set_value(f'q_{bus_name}_{ph}',q/3)
        
    if mode == 'urisi_abc':
        p = s_cplx.real
        q = s_cplx.imag
        for ph in ['a','b','c']:
            grid_obj.set_value(f'p_{bus_name}_{ph}',p/3)
            grid_obj.set_value(f'q_{bus_name}_{ph}',q/3)  
            
def update_loads(grid,data_input):
    
    if type(data_input) == dict:
        data = data_input
        
    if type(data_input) == str:
        data_input = open(data_input).read().replace("'",'"')
        data = json.loads(data_input)
        
    if 'loads' in data:
        for load in data['loads']:
            
            if load['type'] == "1P+N":
                bus = load['bus']
                kVA = load['kVA']
                pf = load['pf']
                p = kVA*1000*pf
                q = np.sqrt((kVA*1000)**2 - p**2)*np.sign(pf)

                grid.set_value(f'p_load_{bus}_1',p)
                grid.set_value(f'q_load_{bus}_1',q)

def v_report(grid,data_input,show=True,model='urisi'):
    '''
    

    Parameters
    ----------
    grid : pydae object
        Pydae object modelling a grid.
    data_input : if dict, a dictionary with the grid parameters
                 if string, the path to the .json file containing grid parameters

    show : string, optional
        If report is print or not.
        
    model : string, optional
        Type of implemented model. The default is 'urisi'.

    Returns
    -------
    dict with the results.

    '''
    
    
    if type(data_input) == dict:
        data = data_input
        
    if type(data_input) == str:
        data_input = open(data_input).read().replace("'",'"')
        data = json.loads(data_input)
        
    buses_dict = {}
        
    for bus in data['buses']:
        if f"v_{bus['bus']}_n_r" in grid.y_ini_list:
            v_n_r,v_n_i = grid.get_mvalue([f"v_{bus['bus']}_n_r",f"v_{bus['bus']}_n_i"])
            v_n_g = v_n_r + 1j*v_n_i
        else:
            v_n_g = 0.0
        #for ph in ['a','b','c']:
        
        ph = 'a'
        v_a_r,v_a_i =  grid.get_mvalue([f"v_{bus['bus']}_{ph}_r",f"v_{bus['bus']}_{ph}_i"])
        v_a_g = v_a_r + 1j*v_a_i
        v_a_n = v_a_g - v_n_g
        v_a_m = np.abs(v_a_n)
        v_a_a = np.rad2deg(np.angle(v_a_n))

        ph = 'b'
        v_b_r,v_b_i =  grid.get_mvalue([f"v_{bus['bus']}_{ph}_r",f"v_{bus['bus']}_{ph}_i"])
        v_b_g = v_b_r + 1j*v_b_i
        v_b_n = v_b_g - v_n_g
        v_b_m = np.abs(v_b_n)
        v_b_a = np.rad2deg(np.angle(v_b_n))

        ph = 'c'
        v_c_r,v_c_i =  grid.get_mvalue([f"v_{bus['bus']}_{ph}_r",f"v_{bus['bus']}_{ph}_i"])
        v_c_g = v_c_r + 1j*v_c_i
        v_c_n = v_c_g - v_n_g
        v_c_m = np.abs(v_c_n)
        v_c_a = np.rad2deg(np.angle(v_c_n))
 
        alpha = alpha = np.exp(2.0/3*np.pi*1j)
        v_0 =  1/3*(v_a_g+v_b_g+v_c_g)
        v_1 = 1.0/3.0*(v_a_g + v_b_g*alpha + v_c_g*alpha**2)
        v_2 = 1.0/3.0*(v_a_g + v_b_g*alpha**2 + v_c_g*alpha)
        
        # compute unbalanced as in Kersting 1ers edition pg. 266
        v_m_array = [v_a_m,v_b_m,v_c_m]
        v_m_min = np.min(v_m_array)
        v_m_max = np.max(v_m_array)
        max_dev = v_m_max - v_m_min
        v_avg = np.sum(v_m_array)/3
        unbalance = max_dev/v_avg
        
        bus[f"v_{bus['bus']}_{'a'}n"] = v_a_m
        
        if show:
            print(f"v_{bus['bus']}_{'a'}n: {v_a_m:7.1f}| {v_a_a:6.1f}º V,    v_{bus['bus']}_{'a'}g: {np.abs(v_a_g):7.1f}| {np.angle(v_a_g,deg=True):6.1f}º V,    v_1 = {np.abs(v_1):7.1f} V, unb = {unbalance*100:3.2f}%")
            print(f"v_{bus['bus']}_{'b'}n: {v_b_m:7.1f}| {v_b_a:6.1f}º V,    V_{bus['bus']}_{'b'}g: {np.abs(v_b_g):7.1f}| {np.angle(v_b_g,deg=True):6.1f}º V,    v_2 = {np.abs(v_2):7.1f} V")
            print(f"v_{bus['bus']}_{'c'}n: {v_c_m:7.1f}| {v_c_a:6.1f}º V,    V_{bus['bus']}_{'c'}g: {np.abs(v_c_g):7.1f}| {np.angle(v_c_g,deg=True):6.1f}º V,    v_0 = {np.abs(v_0):7.1f} V")                   
            
            print(f"  V_{bus['bus']}_ng: {np.abs(v_n_g):8.1f}| {np.angle(v_n_g,deg=True):8.1f}º V")
        
        buses_dict[bus['bus']] = {f"v_{bus['bus']}_{'a'}n":v_b_m,f"v_{bus['bus']}_{'b'}n":v_b_m,f"v_{bus['bus']}_{'c'}n":v_c_m}
        buses_dict[bus['bus']].update({f'v_unb':unbalance,'v_ng':np.abs(v_n_g)})

    return buses_dict        


 


def report_trafos(grid,data_input,model='urisi'):
    '''
    

    Parameters
    ----------
    grid : pydae object
        Pydae object modelling a grid.
    data_input : if dict, a dictionary with the grid parameters
                 if string, the path to the .json file containing grid parameters

    model : string, optional
        Type of implemented model. The default is 'urisi'.

    Returns
    -------
    None.

    '''
    
    
    if type(data_input) == dict:
        data = data_input
        
    if type(data_input) == str:
        data_input = open(data_input).read().replace("'",'"')
        data = json.loads(data_input)
        
    for trafo in data['transformers']:
        
        bus_j_name = trafo['bus_j']
        bus_k_name = trafo['bus_k']
        
        for ph in ['a','b','c']:
            i_r,i_i = grid.get_mvalue([f"i_t_{bus_j_name}_{bus_k_name}_1_{ph}_r",
                                       f"i_t_{bus_j_name}_{bus_k_name}_1_{ph}_i"])
            i_1 = i_r + 1j*i_i
            i_m = np.abs(i_1)
            phi = np.arctan2(i_i,i_r)
            print(f"I_1_{ph}: {i_m:8.1f} |{np.rad2deg(phi):6.1f}º A")
            
        for ph in ['a','b','c']:
            i_r,i_i = grid.get_mvalue([f"i_t_{bus_j_name}_{bus_k_name}_2_{ph}_r",
                                       f"i_t_{bus_j_name}_{bus_k_name}_2_{ph}_i"])
            i_1 = i_r + 1j*i_i
            i_m = np.abs(i_1)
            phi = np.arctan2(i_i,i_r)
            print(f"I_2_{ph}: {i_m:8.1f} |{np.rad2deg(phi):6.1f}º A")
            
        if f"i_t_{bus_j_name}_{bus_k_name}_2_n_r" in grid.outputs_list:
            i_r,i_i = grid.get_mvalue([f"i_t_{bus_j_name}_{bus_k_name}_2_n_r",
                                       f"i_t_{bus_j_name}_{bus_k_name}_2_n_i"])
            i_1 = i_r + 1j*i_i
            i_m = np.abs(i_1)
            phi = np.arctan2(i_i,i_r)            
            print(f"I_2_n: {i_m:8.1f} |{np.rad2deg(phi):6.1f}º A")


def load_shape(grid,data_input,model='urisi'):
    '''
    

    Parameters
    ----------
    grid : pydae object
        Pydae object modelling a grid.
    data_input : if dict, a dictionary with the grid parameters
                 if string, the path to the .json file containing grid parameters

    model : string, optional
        Type of implemented model. The default is 'urisi'.

    Returns
    -------
    None.

    '''
    
    
    if type(data_input) == dict:
        data = data_input
        
    if type(data_input) == str:
        data_input = open(data_input).read().replace("'",'"')
        data = json.loads(data_input)
        
def read_shapes(data_input):
    '''
    

    Parameters
    ----------
    data_input : if string, the path to the .json file containing grid parameters

    Returns
    -------
    dict with the shapes.

    '''
    
    data_input = open(data_input).read().replace("'",'"')
    data = json.loads(data_input)
    shapes = {}
    for item in data['shapes']:
        values = np.array(data['shapes'][item])
        shapes.update({item:{'t':values[:,0],'val':values[:,1]}})
        
    
    
    return shapes