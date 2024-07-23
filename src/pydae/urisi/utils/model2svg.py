from pydae.svg_tools import svg
from pydae.urisi.utils import get_v, get_i
import numpy as np
import os
import json
import hjson

class model2svg(svg):

    def __init__(self, model, grid_data, svg_file):

        super().__init__(svg_file)
        self.model = model
        self.set_grid(grid_data)

        self.V_min_pu = 0.95
        self.V_max_pu = 1.05

        self.set_buses_title()
        self.set_lines_title()
        self.set_transformers_title()

        self.set_lines_currents_colors()
        self.set_buses_voltages_colors()
        
                
    def set_grid(self,grid_data):
        
        if type(grid_data) == dict:
            data = grid_data

        if type(grid_data) == str:
            if os.path.splitext(grid_data)[1] == '.json':
                with open(grid_data,'r') as fobj:
                    data = json.loads(fobj.read().replace("'",'"'))
            if os.path.splitext(grid_data)[1] == '.hjson':
                with open(grid_data,'r') as fobj:
                    data = hjson.loads(fobj.read().replace("'",'"'))

        self.grid_data = data


    def set_buses_title(self):

        load_buses = [bus_name['bus'] for bus_name in self.grid_data['loads']]

        for bus in self.grid_data['buses']:

            if not 'N_nodes' in bus: bus.update({'N_nodes':4})

            if bus['N_nodes'] == 4:
                bus_name = bus['name']
                U_base = bus['U_kV']*1000
                V_base = U_base/np.sqrt(3)
                string = 'Voltages and powers:\n'
                for phn in ['an','bn','cn','ng']:
                    V = get_v(self.model,bus_name,f'V_{phn}_m')
                    V_pu = V/V_base

                    p = 0.0
                    q = 0.0
                    if bus_name in load_buses:
                        ph = phn[0]
                        if  not ph == 'n':
                            p = self.model.get_value(f'p_load_{bus_name}_{ph}')
                            q = self.model.get_value(f'q_load_{bus_name}_{ph}')

                    if not phn == 'ng':  
                        if q < 0.0:    
                            string += f"V{phn} = {V:7.1f} V ({V_pu:4.2f} pu), S{phn} = {p/1e3:5.1f} - {np.abs(q)/1e3:5.1f}j kVA \n"
                        else:
                            string += f"V{phn} = {V:7.1f} V ({V_pu:4.2f} pu), S{phn} = {p/1e3:5.1f} + {np.abs(q)/1e3:5.1f}j kVA \n"

                    else:
                        string += f"V{phn} = {V:7.1f} V ({V_pu:4.2f} pu)\n"
    
            if bus['N_nodes'] == 3:
                bus_name = bus['name']
                U_base = bus['U_kV']*1000
                string = 'Voltages:\n'
                for phn in ['ab','bc','ca']:
                    U = get_v(self.model,bus_name,f'U_{phn}_m')
                    U_pu = U/U_base
                    string += f"U{phn} = {U:7.1f} V, {U_pu:4.2f} pu\n"

            self.set_title(bus_name,string)        

    def set_lines_title(self):

        for line in self.grid_data['lines']:
            bus_j = line['bus_j']
            bus_k = line['bus_k']

            string = 'Currents:\n'
            for it,ph in enumerate(['a','b','c','n']):

                line_name = f'l_{bus_j}_{it}_{bus_k}_{it}'
                I = get_i(self.model,bus_j,bus_k,type=f'I_{ph}_m')
                string += f"I{ph} = {I:6.1f} A\n"

            for it,ph in enumerate(['a','b','c','n']):  
                line_name = f'l_{bus_j}_{it}_{bus_k}_{it}'
                self.set_title(line_name,string) 

    def set_lines_currents_colors(self):
        
        for line in self.grid_data['lines']:
            #if not 'monitor'in line or not 'vsc_line' in line: continue
            if 'monitor' in line:
                if not line['monitor']: continue
            if 'vsc_line' in line:
                if not line['vsc_line']: continue
                    
            bus_j = line['bus_j']
            bus_k = line['bus_k']

            z2a = {'0':'a','1':'b','2':'c','3':'n'}
            for ph in ['0','1','2','3']:
                line_id = f'l_{bus_j}_{ph}_{bus_k}_{ph}'
                if  f'i_{line_id}_r' in self.model.outputs_list:
                    if 'code' in line:
                        I_max = self.grid_data["line_codes"][line['code']]['I_max']
                    else:
                        I_max = line['I_max']

                    i_r = self.model.get_value(f'i_{line_id}_r') 
                    i_i = self.model.get_value(f'i_{line_id}_i') 
                    i = i_r + 1j*i_i
                    i_abs = np.abs(i)

                    if i_abs < 1e-3: continue
                    i_sat = np.clip((i_abs/I_max)**2*255,0,255)
                    self.set_color('line',line_id,(int(i_sat),0,0))

    def set_buses_voltages_colors(self):
        
        for bus in self.grid_data['buses']:

            if not 'N_nodes' in bus: bus['N_nodes'] == 4

            if bus['N_nodes'] == 4:

                bus_name = bus['name']
                U_base = bus['U_kV']*1000
                V_base = U_base/np.sqrt(3)
                string = 'Voltages:\n'
                V_pu_max_deviation = 0.0
                for phn in ['an','bn','cn']:
                    V = get_v(self.model,bus_name,f'V_{phn}_m')
                    V_pu = V/V_base
                    DV = V_pu - 1.0
                    if np.abs(DV) > np.abs(V_pu_max_deviation):
                        V_pu_max_deviation = DV
                V_pu = V_pu_max_deviation + 1.0
               
                # when V_pu = V_med_pu color = 0, when V_pu = V_max_pu color = 255 (red)
                # when V_pu = V_med_pu color = 0, when V_pu = V_min_pu color = 255 (blue)
                if V_pu < 1:
                    blue = np.clip(255*((V_pu - 1)/(self.V_min_pu - 1))**2,0,255)
                    self.set_color('rect',f'{bus["name"]}',(0,0,int(blue)))  
                if V_pu > 1:
                    red  = np.clip(255*((V_pu - 1)/(self.V_max_pu - 1))**2,0,255)
                    self.set_color('rect',f'{bus["name"]}',(int(red),0,0)) 
                
            if bus['N_nodes'] == 3:

                bus_name = bus['name']
                U_base = bus['U_kV']*1000
                string = 'Voltages:\n'
                for phn in ['ab','bc','ca']:
                    U = get_v(self.model,bus_name,f'U_{phn}_m')
                    U_pu = U/U_base
            
                # when V_pu = V_med_pu color = 0, when V_pu = V_max_pu color = 255 (red)
                # when V_pu = V_med_pu color = 0, when V_pu = V_min_pu color = 255 (blue)
                if U_pu < 1:
                    blue = np.clip(255*((U_pu - 1)/(self.V_min_pu - 1))**2,0,255)
                    self.set_color('rect',f'{bus["name"]}',(0,0,int(blue)))  
                if U_pu > 1:
                    red  = np.clip(255*((U_pu - 1)/(self.V_max_pu - 1))**2,0,255)
                    self.set_color('rect',f'{bus["name"]}',(int(red),0,0)) 
                

                # if acdc == 'ac':
                #     if V_pu < v_ac_min:
                #         self.post_data.update({'v_ac_min':{'bus':bus['name'],'value':V_pu}})
                #         v_ac_min = V_pu
                #     if V_pu > v_ac_max:
                #         self.post_data.update({'v_ac_max':{'bus':bus['name'],'value':V_pu}})
                #         v_ac_

    def set_transformers_title(self):

        for trafo in self.grid_data['transformers']:
            bus_j = trafo['bus_j']
            bus_k = trafo['bus_k']

            string = 'Currents:\n'

            I_1_a = self.model.get_value(f'i_t_{bus_j}_{bus_k}_1_0_r') + 1j*self.model.get_value(f'i_t_{bus_j}_{bus_k}_1_0_i')
            I_1_b = self.model.get_value(f'i_t_{bus_j}_{bus_k}_1_1_r') + 1j*self.model.get_value(f'i_t_{bus_j}_{bus_k}_1_1_i')
            I_1_c = self.model.get_value(f'i_t_{bus_j}_{bus_k}_1_2_r') + 1j*self.model.get_value(f'i_t_{bus_j}_{bus_k}_1_2_i')

            I_2_a = self.model.get_value(f'i_t_{bus_j}_{bus_k}_2_0_r') + 1j*self.model.get_value(f'i_t_{bus_j}_{bus_k}_2_0_i')
            I_2_b = self.model.get_value(f'i_t_{bus_j}_{bus_k}_2_1_r') + 1j*self.model.get_value(f'i_t_{bus_j}_{bus_k}_2_1_i')
            I_2_c = self.model.get_value(f'i_t_{bus_j}_{bus_k}_2_2_r') + 1j*self.model.get_value(f'i_t_{bus_j}_{bus_k}_2_2_i')
            I_2_n = self.model.get_value(f'i_t_{bus_j}_{bus_k}_2_3_r') + 1j*self.model.get_value(f'i_t_{bus_j}_{bus_k}_2_3_i')

            string += f"Ia = {np.abs(I_1_a):6.1f} A\tIa = {np.abs(I_2_a):6.1f} A \n"
            string += f"Ib = {np.abs(I_1_b):6.1f} A\tIb = {np.abs(I_2_b):6.1f} A \n"
            string += f"Ic = {np.abs(I_1_c):6.1f} A\tIc = {np.abs(I_2_c):6.1f} A \n"
            string += f"\t\t\t  In = {np.abs(I_2_n):6.1f} A \n"

            self.set_title(f'trafo_{bus_j}_{bus_k}_g',string) 