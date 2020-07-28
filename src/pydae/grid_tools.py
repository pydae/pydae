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
            
        self.nodes_list = nodes_list
        self.Y_primitive = Y_primitive
        self.A_conect = A_conect
        self.node_sorter = node_sorter

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



        
    def dae2vi(self):
        n2a = {'1':'a','2':'b','3':'c','4':'n'}
        V_node_list = []
        I_node_list = []        
        for item in self.nodes_list:
            bus_name,phase_name = item.split('.')
            i = get_i(self.syst,bus_name,phase_name=n2a[phase_name],i_type='phasor',dq_name='ri')
            I_node_list += [i]
            v = get_v(self.syst,bus_name,phase_name=n2a[phase_name],v_type='phasor',dq_name='ri')
            V_node_list += [v]

        I_node = np.array(I_node_list).reshape(len(I_node_list),1)
        V_node = np.array(V_node_list).reshape(len(V_node_list),1)
        self.I_node = I_node
        self.V_node = V_node

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
        self.I_results = self.I_node
        
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