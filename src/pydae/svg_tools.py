# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 10:33:57 2021

@author: jmmau
"""

from xml.etree import ElementTree as ET
from xml.etree.ElementTree import Element

import numpy as np
import json
import svgwrite

class svg():
    
    def __init__(self,input_file):
        ET.register_namespace("","http://www.w3.org/2000/svg")
        ET.register_namespace("inkscape","http://www.inkscape.org/namespaces/inkscape")
        ET.register_namespace("sodipodi","http://sodipodi.sourceforge.net/DTD/sodipodi-0.dtd") 
        ET.register_namespace("cc","http://creativecommons.org/ns#") 
        ET.register_namespace("rdf","http://www.w3.org/1999/02/22-rdf-syntax-ns#")      
        self.tree = ET.parse(input_file)
        self.root = self.tree.getroot()
        self.g_list = self.root.findall(".//{http://www.w3.org/2000/svg}g")
        self.N_steps = 1000
        self.input_file = input_file
        self.begin_click = False
        self.begin = ''
        self.anim_id = ''
        self.anim_i = 0
        
        self.V_min_pu = 0.95
        self.V_max_pu = 1.05
        
        self.post_data = {}

    def set_size(self,width,height):
        self.root.attrib['width']  = f'{width}px'
        self.root.attrib['height'] = f'{height}px'
        
    def save(self,output_file=''):
        if output_file=='':
            output_file = f"{self.input_file.replace('.svg','')}_anim.svg"
        self.tree.write(output_file)
        
    def set_text(self,text_id,string):
        for text in self.root.findall('.//{http://www.w3.org/2000/svg}text'):
            if text.attrib['id'] == text_id: 
                text.text = string
        #for tspan in text_obj.findall('.//{http://www.w3.org/2000/svg}tspan'):
        #    tspan.text = string
 
    def set_title(self,text_id,string):
        for text in self.root.findall('.//{http://www.w3.org/2000/svg}text'):
            if text.attrib['id'] == text_id: text_obj = text
        for tspan in text_obj.findall('.//{http://www.w3.org/2000/svg}tspan'):
            tspan.text = string
            
    def set_rect_style(self,object_id,new_style):
        #style="fill:#337ab7"
        for rect in self.root.findall('.//{http://www.w3.org/2000/svg}rect'):
            
            if rect.attrib['id'] == object_id: 
                #print(rect.attrib['style'])
                #if 'style' in rect.attrib:
                #    rect.attrib['style'].update(new_style)
                #else:
                rect.attrib['style'] = new_style

    def set_line_style(self,object_id,new_style):
        #style="fill:#337ab7"
        for rect in self.root.findall('.//{http://www.w3.org/2000/svg}line'):
            if 'id' in rect.attrib:
                if rect.attrib['id'] == object_id: 
                    #if 'style' in rect.attrib:
                    #    rect.attrib['style'].update(new_style)
                    #else:
                    rect.attrib['style'] = new_style

    def set_path_style(self,object_id,new_style):
        #style="fill:#337ab7"
        for path in self.root.findall('.//{http://www.w3.org/2000/svg}path'):
            if 'id' in path.attrib:
                if path.attrib['id'] == object_id: 
                    #if 'style' in rect.attrib:
                    #    rect.attrib['style'].update(new_style)
                    #else:
                    path.attrib['style'] = new_style
                    print(path.attrib['style'])


    def set_color(self,type_,id_,rgb):
        if type_ == 'rect':
            self.set_rect_style(id_,f"fill:#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}")
            
        if type_ == 'line':
            self.set_line_style(id_,f"stroke:#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}")

        if type_ == 'path':
            self.set_path_style(id_,f"stroke:#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}")


    def tostring(self):
        self.svg_str = ET.tostring(self.root).decode()
        return self.svg_str
    
    def set_grid(self,grid,grid_data):
        self.grid=grid
        
        if type(grid_data) == dict:
            data = grid_data

        if type(grid_data) == str:
            data_input = open(grid_data).read().replace("'",'"')
            data = json.loads(data_input)
            
        self.grid_data = data

    def set_tooltips(self, output_file):

        self.set_tooltips_v2(output_file)


    def set_lines_currents_v1(self):
        
        for line in self.grid_data['lines']:
            #if not 'monitor'in line or not 'vsc_line' in line: continue
            if 'monitor' in line:
                if not line['monitor']: continue
            if 'vsc_line' in line:
                if not line['vsc_line']: continue
                    
            bus_j = line['bus_j']
            bus_k = line['bus_k']

            for ph in ['a','b','c','n']:

                I_max = self.grid_data["line_codes"][line['code']]['I_max']
                i_r = self.grid.get_value(f'i_l_{bus_j}_{bus_k}_{ph}_r') 
                i_i = self.grid.get_value(f'i_l_{bus_j}_{bus_k}_{ph}_i') 
                i = i_r + 1j*i_i
                i_abs = np.abs(i)
                #print(f'l_{bus_j}_{bus_k}_{ph} = {i_abs:8.1f} A')
                if i_abs < 1e-3: continue
                i_sat = np.clip((i_abs/I_max)**2*255,0,255)
                self.set_color('line',f'l_{bus_j}_{bus_k}_{ph}',(int(i_sat),0,0))

    def set_buses_voltages_v1(self):
        
        v_ac_min = 20e6
        min_dc_voltage = 20e6
        min_n_ac_voltage = 20e6
        min_n_dc_voltage = 20e6    
        
        v_ac_max = 0
        max_dc_voltage = 0
        max_n_ac_voltage = 0
        max_n_dc_voltage = 0  


        for bus in self.grid_data['buses']:
            v_r = self.grid.get_value(f'v_{bus["bus"]}_a_r') 
            v_i = self.grid.get_value(f'v_{bus["bus"]}_a_i') 
            v = v_r + 1j*v_i
            v_abs = np.abs(v)
            V_med_pu= 0.5*(self.V_max_pu + self.V_min_pu)
            acdc = 'ac'
            if 'acdc' in bus:
                if bus['acdc'] == 'DC':
                    V_nom = bus['U_kV']*1000
                    acdc = 'dc'
                if bus['acdc'] == 'AC':
                    V_nom = bus['U_kV']/np.sqrt(3)*1000  
                    acdc = 'ac'
            else:
                V_nom = bus['U_kV']/np.sqrt(3)*1000 
                acdc = 'ac'
                
            V_pu = v_abs/V_nom
            
            # when V_pu = V_med_pu color = 0, when V_pu = V_max_pu color = 255 (red)
            # when V_pu = V_med_pu color = 0, when V_pu = V_min_pu color = 255 (blue)
            if V_pu < V_med_pu:
                blue = np.clip(255*((V_pu - V_med_pu)/(self.V_min_pu - V_med_pu))**2,0,255)
                self.set_color('rect',f'{bus["bus"]}',(0,0,int(blue)))  
            if V_pu > V_med_pu:
                red  = np.clip(255*((V_pu - V_med_pu)/(self.V_max_pu - V_med_pu))**2,0,255)
                self.set_color('rect',f'{bus["bus"]}',(int(red),0,0)) 
            
            if acdc == 'ac':
                if V_pu < v_ac_min:
                    self.post_data.update({'v_ac_min':{'bus':bus['bus'],'value':V_pu}})
                    v_ac_min = V_pu
                if V_pu > v_ac_max:
                    self.post_data.update({'v_ac_max':{'bus':bus['bus'],'value':V_pu}})
                    v_ac_max = V_pu   

    def set_tooltips_v2(self, output_file):
            
        s = self
        grid = self.grid
        
        for bus in s.grid_data['buses']:
            bus_id = bus['name']
            bus_elm_list = s.root.findall(f".//*[@id='{bus_id}']")
            if len(bus_elm_list) > 0:
                bus_elm = bus_elm_list[0]
            else:
                print(f'SVG element {bus_id} not found')
                continue

            if f"V_{bus['name']}_3_r" in grid.y_ini_list:
                v_n_r,v_n_i = grid.get_mvalue([f"V_{bus['name']}_3_r",f"V_{bus['name']}_3_i"])
                v_n = v_n_r + 1j*v_n_i
            else:
                v_n = 0.0
            for ph,da in zip(['0','1','2'],['data1','data2','data3']):
                V_r_id,V_i_id = f"V_{bus['name']}_{ph}_r",f"V_{bus['name']}_{ph}_i"
                if V_r_id in grid.y_ini_list:
                    v_r,v_i =  grid.get_mvalue([V_r_id,V_i_id])
                    v = v_r + 1j*v_i
                    v_m = np.abs(v-v_n)
                if 'acdc' in bus:
                    v_base = bus['U_kV']*1000/2
                else:
                    v_base = bus['U_kV']*1000/np.sqrt(3)
                bus_elm.attrib[da] =f"{v_m:20.1f} V / {v_m/v_base:9.2f} pu "
            bus_elm.attrib['data4'] =f"{np.abs(v_n):20.1f} V"   
            bus_elm.attrib['data0'] = 'bus'

            p_load_a,q_load_a = -0.001,-0.001
            p_load_b,q_load_b = -0.001,-0.001
            p_load_c,q_load_c = -0.001,-0.001
            p_load = 0.0
            q_load = 0.0
            if f"p_load_{bus_id}_a" in grid.u_ini_list:  # ac load
                p_load_a = grid.get_value(f"p_load_{bus_id}_a")
                q_load_a = grid.get_value(f"q_load_{bus_id}_a")
                p_load_b = grid.get_value(f"p_load_{bus_id}_b")
                q_load_b = grid.get_value(f"q_load_{bus_id}_b")
                p_load_c = grid.get_value(f"p_load_{bus_id}_c")
                q_load_c = grid.get_value(f"q_load_{bus_id}_c")
                p_load = p_load_a + p_load_b + p_load_c
                q_load = q_load_a + q_load_b + q_load_c
            if f"p_load_{bus_id}" in grid.u_ini_list:    # dc load
                p_load = grid.get_value(f"p_load_{bus_id}")
                q_load = 0.0

            bus_elm.attrib['data5'] =f"{p_load/1e3:20.1f} kW" 
            bus_elm.attrib['data6'] =f"{q_load/1e3:20.1f} kvar" 
            bus_elm.attrib['data7'] ="xx" 

            bus_elm.attrib['class'] = 'tooltip-trigger'

        for line in s.grid_data['lines']:

            bus_j = line['bus_j']
            bus_k = line['bus_k']

            for ph in ['0','1','2','3']:

                line_id = f'l_{bus_j}_{ph}_{bus_k}_{ph}'
                line_svg =  s.root.findall(f".//*[@id='{line_id}']")
                if len(line_svg)>0:
                    line_elm = line_svg[0]
                else:
                    continue
                line_elm.attrib['data0'] = 'line'

                for phi,da in zip(['0','1','2','3'],['data1','data2','data3','data4']):
                    meas_line_id = f'l_{bus_j}_{phi}_{bus_k}_{phi}'

                    i_m = 0.0

                    if f'i_{meas_line_id}_r' in grid.outputs_list:
                        i_r,i_i =  grid.get_mvalue([f"i_{meas_line_id}_r",f"i_{meas_line_id}_i"])
                        i = i_r + 1j*i_i
                        i_m = np.abs(i)      
                    v_base = 1.0
                    line_elm.attrib[da] =f"{i_m:20.1f} A / {i_m/v_base:9.2f} pu "

                line_elm.attrib['class'] = 'tooltip-trigger'

        for trafo in s.grid_data['transformers']:

            bus_j = trafo['bus_j']
            bus_k = trafo['bus_k']

            for wind in [1,2]:
                trafo_id = f'trafo_{bus_j}_{bus_k}_{wind}'

                trafo_svg =  s.root.findall(f".//*[@id='{trafo_id}']")
                if len(trafo_svg)>0:
                    trafo_elm = trafo_svg[0]
                else:
                    print(f'No trafo {trafo_id} found')
                    continue

                trafo_elm.attrib['data0'] = 'trafo'


                z2a = {'0':'a','1':'b','2':'c','3':'n'}
                string_1 = ''
                for ph in ['0','1','2']:
                    i_r_id = f'i_t_{bus_j}_{bus_k}_1_{ph}_r'
                    i_i_id = f'i_t_{bus_j}_{bus_k}_1_{ph}_i'
                    if i_r_id in grid.outputs_list:
                        i_r,i_i =  grid.get_mvalue([i_r_id,i_i_id])
                        i = i_r + 1j*i_i
                        i_m = np.abs(i)  
                        string_1 += f'{z2a[ph]}:  {i_m:0.2f}\t     '

                string_2 = ''
                for ph in ['0','1','2','3']:
                    i_r_id = f'i_t_{bus_j}_{bus_k}_2_{ph}_r'
                    i_i_id = f'i_t_{bus_j}_{bus_k}_2_{ph}_i'
                    if i_r_id in grid.outputs_list:
                        i_r,i_i =  grid.get_mvalue([i_r_id,i_i_id])
                        i = i_r + 1j*i_i
                        i_m = np.abs(i)  
                        string_2 += f'{z2a[ph]}: {i_m:0.2f}     '

                trafo_elm.attrib['data1'] = string_1 + ' A'
                trafo_elm.attrib['data2'] = string_2 + ' A'
                trafo_elm.attrib['data3'] = ''
                trafo_elm.attrib['data4'] = ''
            trafo_elm.attrib['class'] = 'tooltip-trigger'

        if 'vscs' in self.grid_data:
            vscs_list = self.grid_data['vscs']
        else:
            vscs_list = []
            
        for vsc in vscs_list:

            if not 'bus_ac' in vsc: continue

            bus_ac = vsc['bus_ac']
            bus_dc = vsc['bus_dc']
            vsc_id = f'vsc_{bus_ac}_{bus_dc}'

            vsc_svg =  s.root.findall(f".//*[@id='{vsc_id}']")
            if len(vsc_svg)>0:
                vsc_elm = vsc_svg[0]
            else:
                print(f'No VSC {vsc_id} found')
                continue
            vsc_elm.attrib['data0'] = 'vsc'
            p_dc = grid.get_value(f'p_vsc_{bus_dc}')
            vsc_elm.attrib['data1'] = 'VSC'

            p_ac = grid.get_value(f'p_vsc_{bus_ac}')   
            p_dc = grid.get_value(f'p_vsc_{bus_dc}')   
            vsc_elm.attrib['data2'] = f'  Pac = {p_ac/1e3:0.1f} kW,  Pdc = {p_dc/1e3:0.1f} kW'

            p_loss = grid.get_value(f'p_vsc_loss_{bus_ac}')
            vsc_elm.attrib['data3'] = f'  Losses = {p_loss/1e3:0.1f} kW'

            vsc_elm.attrib['class'] = 'tooltip-trigger'
            
        script = '''
        <![CDATA[
                (function() {
                    var svg = document.getElementById('document_id');
                    var tooltip_1 = svg.getElementById('tooltip1');
                var tooltipText_1_1 = tooltip_1.getElementsByTagName('text')[0];
                var tooltipText_1_2 = tooltip_1.getElementsByTagName('text')[1];
                var tooltipText_1_3 = tooltip_1.getElementsByTagName('text')[2];
                var tooltipText_1_4 = tooltip_1.getElementsByTagName('text')[3];
                var tooltipText_1_5 = tooltip_1.getElementsByTagName('text')[4];
                var tooltipText_1_6 = tooltip_1.getElementsByTagName('text')[5];
                    var tooltip_2 = svg.getElementById('tooltip2');
                var tooltipText_2_1 = tooltip_2.getElementsByTagName('text')[0];
                var tooltipText_2_2 = tooltip_2.getElementsByTagName('text')[1];
                var tooltipText_2_3 = tooltip_2.getElementsByTagName('text')[2];
                var tooltipText_2_4 = tooltip_2.getElementsByTagName('text')[3];
                var tooltipText_2_5 = tooltip_2.getElementsByTagName('text')[4];
                var tooltipText_2_6 = tooltip_2.getElementsByTagName('text')[5];
                    var tooltip_3 = svg.getElementById('tooltip3');
                var tooltipText_3_1 = tooltip_3.getElementsByTagName('text')[0];
                var tooltipText_3_2 = tooltip_3.getElementsByTagName('text')[1];
                var tooltipText_3_3 = tooltip_3.getElementsByTagName('text')[2];
                var tooltipText_3_4 = tooltip_3.getElementsByTagName('text')[3];
        
                var tooltipRects = tooltip_1.getElementsByTagName('rect');
                var triggers = svg.getElementsByClassName('tooltip-trigger');
                for (var i = 0; i < triggers.length; i++) {
                    triggers[i].addEventListener('mousemove', showTooltip);
                    triggers[i].addEventListener('mouseout', hideTooltip);
                }
                function showTooltip(evt) {
                    var CTM = svg.getScreenCTM();
                    var x_unsat =  (evt.clientX - CTM.e) / CTM.a
                    var x = x_unsat;
                    if  (x_unsat>x_max) {var x = x_max}
                    if  (x_unsat<x_min) {var x = x_min}
        
                    
                    var y_unsat = (evt.clientY - CTM.f) / CTM.d + 30;
                    var y = y_unsat;
                    if  (y_unsat>y_max) {var y = y_max - y_height}
                    
                    
                    if (evt.target.getAttributeNS(null, "data0") == "bus") {
                        tooltip_1.setAttributeNS(null, "transform", "translate(" + x + " " + y + ")");
                        tooltip_1.setAttributeNS(null, "visibility", "visible")}
                    
                    if (evt.target.getAttributeNS(null, "data0") == "line") {
                        tooltip_2.setAttributeNS(null, "transform", "translate(" + x + " " + y + ")");
                        tooltip_2.setAttributeNS(null, "visibility", "visible")}
        
                    if (evt.target.getAttributeNS(null, "data0") == "vsc") {
                        tooltip_3.setAttributeNS(null, "transform", "translate(" + x + " " + y + ")");
                        tooltip_3.setAttributeNS(null, "visibility", "visible")}
                        
                    if (evt.target.getAttributeNS(null, "data0") == "trafo") {
                        tooltip_3.setAttributeNS(null, "transform", "translate(" + x + " " + y + ")");
                        tooltip_3.setAttributeNS(null, "visibility", "visible")}    
                        
                    tooltipText_1_1.firstChild.data = evt.target.getAttributeNS(null, "data1");
                    tooltipText_1_2.firstChild.data = evt.target.getAttributeNS(null, "data2");
                    tooltipText_1_3.firstChild.data = evt.target.getAttributeNS(null, "data3");
                    tooltipText_1_4.firstChild.data = evt.target.getAttributeNS(null, "data4");
                    tooltipText_1_5.firstChild.data = evt.target.getAttributeNS(null, "data5");
                    tooltipText_1_6.firstChild.data = evt.target.getAttributeNS(null, "data6");
            
                    tooltipText_2_1.firstChild.data = evt.target.getAttributeNS(null, "data1");
                    tooltipText_2_2.firstChild.data = evt.target.getAttributeNS(null, "data2");
                    tooltipText_2_3.firstChild.data = evt.target.getAttributeNS(null, "data3");
                    tooltipText_2_4.firstChild.data = evt.target.getAttributeNS(null, "data4");
                    tooltipText_2_5.firstChild.data = evt.target.getAttributeNS(null, "data5");
                    tooltipText_2_6.firstChild.data = evt.target.getAttributeNS(null, "data6");
        
                    tooltipText_3_1.firstChild.data = evt.target.getAttributeNS(null, "data1");
                    tooltipText_3_2.firstChild.data = evt.target.getAttributeNS(null, "data2");
                    tooltipText_3_3.firstChild.data = evt.target.getAttributeNS(null, "data3");
                    tooltipText_3_4.firstChild.data = evt.target.getAttributeNS(null, "data4");
                    
                        var length = tooltipText_1_1.getComputedTextLength()+200;
                        for (var i = 0; i < tooltipRects.length; i++) {
                            tooltipRects[i].setAttributeNS(null, "width", length + 8);
                        }
                    }
                    function hideTooltip(evt) {
                        tooltip_1.setAttributeNS(null, "visibility", "hidden");
                        tooltip_2.setAttributeNS(null, "visibility", "hidden");
                        tooltip_3.setAttributeNS(null, "visibility", "hidden");
                    }
                })()
            ]]>'''
        
        
        
        element = Element('script')
        element.attrib['type'] ="text/ecmascript"
        element.attrib['id'] = "script15"
        element.text = 'scriptplace'
        self.root.append(element)
        
        self.set_lines_currents_v2()
        self.set_buses_voltages_v2()
            
        out = self.tostring().replace('scriptplace',script)
        width = float(self.root.attrib['width'])
        height = float(self.root.attrib['height'])
        document_id = self.root.attrib['id']
        
        tooltip_width = 320
        tooltip_height = 100
        
        out=out.replace('x_max',f'{width-tooltip_width/2}' )
        out=out.replace('x_min',f'{tooltip_width/2}' )
        out=out.replace('y_max',f'{height-tooltip_height}' )
        out=out.replace('y_height',f'{tooltip_height}' )
        out=out.replace('document_id', document_id)
        
        with open(output_file,'w') as fobj:
            fobj.write(out)     
            
    def set_lines_currents_v2(self):
        
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
                if  f'i_{line_id}_r' in self.grid.outputs_list:
                    
                    I_max = self.grid_data["line_codes"][line['code']]['I_max']
                    i_r = self.grid.get_value(f'i_{line_id}_r') 
                    i_i = self.grid.get_value(f'i_{line_id}_i') 
                    i = i_r + 1j*i_i
                    i_abs = np.abs(i)
                    #print(f'l_{bus_j}_{bus_k}_{ph} = {i_abs:8.1f} A')
                    if i_abs < 1e-3: continue
                    i_sat = np.clip((i_abs/I_max)**2*255,0,255)
                    self.set_color('line',line_id,(int(i_sat),0,0))

    def set_buses_voltages_v2(self):
        
        v_ac_min = 20e6
        min_dc_voltage = 20e6
        min_n_ac_voltage = 20e6
        min_n_dc_voltage = 20e6    
        
        v_ac_max = 0
        max_dc_voltage = 0
        max_n_ac_voltage = 0
        max_n_dc_voltage = 0  


        for bus in self.grid_data['buses']:
            v_r = self.grid.get_value(f'V_{bus["name"]}_0_r') 
            v_i = self.grid.get_value(f'V_{bus["name"]}_0_i') 
            v = v_r + 1j*v_i
            v_abs = np.abs(v)
            V_med_pu= 0.5*(self.V_max_pu + self.V_min_pu)
            acdc = 'ac'
            if 'acdc' in bus:
                if bus['acdc'] == 'DC':
                    V_nom = bus['U_kV']*1000/2
                    acdc = 'dc'
                if bus['acdc'] == 'AC':
                    V_nom = bus['U_kV']/np.sqrt(3)*1000  
                    acdc = 'ac'
            else:
                V_nom = bus['U_kV']/np.sqrt(3)*1000 
                acdc = 'ac'
                
            V_pu = v_abs/V_nom
            
            # when V_pu = V_med_pu color = 0, when V_pu = V_max_pu color = 255 (red)
            # when V_pu = V_med_pu color = 0, when V_pu = V_min_pu color = 255 (blue)
            if V_pu < V_med_pu:
                blue = np.clip(255*((V_pu - V_med_pu)/(self.V_min_pu - V_med_pu))**2,0,255)
                self.set_color('rect',f'{bus["name"]}',(0,0,int(blue)))  
            if V_pu > V_med_pu:
                red  = np.clip(255*((V_pu - V_med_pu)/(self.V_max_pu - V_med_pu))**2,0,255)
                self.set_color('rect',f'{bus["name"]}',(int(red),0,0)) 
            
            if acdc == 'ac':
                if V_pu < v_ac_min:
                    self.post_data.update({'v_ac_min':{'bus':bus['name'],'value':V_pu}})
                    v_ac_min = V_pu
                if V_pu > v_ac_max:
                    self.post_data.update({'v_ac_max':{'bus':bus['name'],'value':V_pu}})
                    v_ac_max = V_pu   

    def set_tooltips_v1(self, output_file):
            
        s = self
        grid = self.grid
        
        for bus in s.grid_data['buses']:
            bus_id = bus['bus']
            bus_elm_list = s.root.findall(f".//*[@id='{bus_id}']")
            if len(bus_elm_list) > 0:
                bus_elm = bus_elm_list[0]
            else:
                print(f'SVG element {bus_id} not found')
                continue
    
            if f"v_{bus['bus']}_n_r" in grid.y_ini_list:
                v_n_r,v_n_i = grid.get_mvalue([f"v_{bus['bus']}_n_r",f"v_{bus['bus']}_n_i"])
                v_n = v_n_r + 1j*v_n_i
            else:
                v_n = 0.0
            for ph,da in zip(['a','b','c'],['data1','data2','data3']):
                v_r,v_i =  grid.get_mvalue([f"v_{bus['bus']}_{ph}_r",f"v_{bus['bus']}_{ph}_i"])
                v = v_r + 1j*v_i
                v_m = np.abs(v-v_n)
                if 'acdc' in bus:
                    v_base = bus['U_kV']*1000
                else:
                    v_base = bus['U_kV']*1000/np.sqrt(3)
                bus_elm.attrib[da] =f"{v_m:20.1f} V / {v_m/v_base:9.2f} pu "
            bus_elm.attrib['data4'] =f"{np.abs(v_n):20.1f} V"   
            bus_elm.attrib['data0'] = 'bus'
    
            p_load_a,q_load_a = -0.001,-0.001
            p_load_b,q_load_b = -0.001,-0.001
            p_load_c,q_load_c = -0.001,-0.001
            if f"p_load_{bus_id}_a" in grid.u_ini_list:
                p_load_a = grid.get_value(f"p_load_{bus_id}_a")
                q_load_a = grid.get_value(f"q_load_{bus_id}_a")
                p_load_b = grid.get_value(f"p_load_{bus_id}_b")
                q_load_b = grid.get_value(f"q_load_{bus_id}_b")
                p_load_c = grid.get_value(f"p_load_{bus_id}_c")
                q_load_c = grid.get_value(f"q_load_{bus_id}_c")
            p_load = p_load_a + p_load_b + p_load_c
            q_load = q_load_a + q_load_b + q_load_c
    
            bus_elm.attrib['data5'] =f"{p_load/1e3:20.1f} kW" 
            bus_elm.attrib['data6'] =f"{q_load/1e3:20.1f} kvar" 
            bus_elm.attrib['data7'] ="xx" 
    
            bus_elm.attrib['class'] = 'tooltip-trigger'
    
        for line in s.grid_data['lines']:
            bus_j = line['bus_j']
            bus_k = line['bus_k']
    
            for ph in ['a','b','c','n']:
    
                line_id = f'l_{bus_j}_{bus_k}_{ph}'
                line_svg =  s.root.findall(f".//*[@id='{line_id}']")
                if len(line_svg)>0:
                    line_elm = line_svg[0]
                else:
                    continue
                line_elm.attrib['data0'] = 'line'
    
                for phi,da in zip(['a','b','c','n'],['data1','data2','data3','data4']):
                    meas_line_id = f'l_{bus_j}_{bus_k}_{phi}'
    
                    i_m = 0.0
                    if f'i_{line_id}_r' in grid.outputs_list:
                        i_r,i_i =  grid.get_mvalue([f"i_{meas_line_id}_r",f"i_{meas_line_id}_i"])
                        i = i_r + 1j*i_i
                        i_m = np.abs(i)      
                    v_base = 1.0
                    line_elm.attrib[da] =f"{i_m:20.1f} A / {i_m/v_base:9.2f} pu "
    
                line_elm.attrib['class'] = 'tooltip-trigger'
    
        for trafo in s.grid_data['transformers']:
            bus_j = trafo['bus_j']
            bus_k = trafo['bus_k']
    
            for wind in [1,2]:
                trafo_id = f'trafo_{bus_j}_{bus_k}_{wind}'
    
                trafo_svg =  s.root.findall(f".//*[@id='{trafo_id}']")
                if len(trafo_svg)>0:
                    trafo_elm = trafo_svg[0]
                else:
                    print(f'No trafo {trafo_id} found')
                    continue
    
                trafo_elm.attrib['data0'] = 'trafo'
    
                string_1 = ''
                for ph in ['a','b','c']:
                    i_r,i_i =  grid.get_mvalue([f'i_t_{bus_j}_{bus_k}_1_{ph}_r',f'i_t_{bus_j}_{bus_k}_1_{ph}_i'])
                    i = i_r + 1j*i_i
                    i_m = np.abs(i)  
                    string_1 += f'{ph}:  {i_m:0.2f}\t'
    
                string_2 = ''
                for ph in ['a','b','c','n']:
                    i_r,i_i =  grid.get_mvalue([f'i_t_{bus_j}_{bus_k}_2_{ph}_r',f'i_t_{bus_j}_{bus_k}_2_{ph}_i'])
                    i = i_r + 1j*i_i
                    i_m = np.abs(i)  
                    string_2 += f'{ph}: {i_m:0.2f}     '
    
                trafo_elm.attrib['data1'] = string_1 + ' A'
                trafo_elm.attrib['data2'] = string_2 + ' A'
                trafo_elm.attrib['data3'] = 'trafo'
                trafo_elm.attrib['data4'] = 'trafo'
            trafo_elm.attrib['class'] = 'tooltip-trigger'
    
        if 'vscs' in self.grid_data:
            vscs_list = self.grid_data['vscs']
        else:
            vscs_list = []
            
        for vsc in vscs_list:

            if not 'bus_ac' in vsc: continue

            bus_ac = vsc['bus_ac']
            bus_dc = vsc['bus_dc']
            vsc_id = f'vsc_{bus_ac}_{bus_dc}'
    
            vsc_svg =  s.root.findall(f".//*[@id='{vsc_id}']")
            if len(vsc_svg)>0:
                vsc_elm = vsc_svg[0]
            else:
                print(f'No VSC {vsc_id} found')
                continue
            vsc_elm.attrib['data0'] = 'vsc'
            p_dc = grid.get_value(f'p_vsc_{bus_dc}')
            vsc_elm.attrib['data1'] = 'VSC'
    
            p_ac = grid.get_value(f'p_vsc_{bus_ac}')   
            p_dc = grid.get_value(f'p_vsc_{bus_dc}')   
            vsc_elm.attrib['data2'] = f'  Pac = {p_ac/1e3:0.1f} kW,  Pdc = {p_dc/1e3:0.1f} kW'
    
            p_loss = grid.get_value(f'p_vsc_loss_{bus_ac}')
            vsc_elm.attrib['data3'] = f'  Losses = {p_loss/1e3:0.1f} kW'
    
            vsc_elm.attrib['class'] = 'tooltip-trigger'
            
        script = '''
        <![CDATA[
        		(function() {
        			var svg = document.getElementById('document_id');
        			var tooltip_1 = svg.getElementById('tooltip1');
                var tooltipText_1_1 = tooltip_1.getElementsByTagName('text')[0];
                var tooltipText_1_2 = tooltip_1.getElementsByTagName('text')[1];
                var tooltipText_1_3 = tooltip_1.getElementsByTagName('text')[2];
                var tooltipText_1_4 = tooltip_1.getElementsByTagName('text')[3];
                var tooltipText_1_5 = tooltip_1.getElementsByTagName('text')[4];
                var tooltipText_1_6 = tooltip_1.getElementsByTagName('text')[5];
        			var tooltip_2 = svg.getElementById('tooltip2');
                var tooltipText_2_1 = tooltip_2.getElementsByTagName('text')[0];
                var tooltipText_2_2 = tooltip_2.getElementsByTagName('text')[1];
                var tooltipText_2_3 = tooltip_2.getElementsByTagName('text')[2];
                var tooltipText_2_4 = tooltip_2.getElementsByTagName('text')[3];
                var tooltipText_2_5 = tooltip_2.getElementsByTagName('text')[4];
                var tooltipText_2_6 = tooltip_2.getElementsByTagName('text')[5];
        			var tooltip_3 = svg.getElementById('tooltip3');
                var tooltipText_3_1 = tooltip_3.getElementsByTagName('text')[0];
                var tooltipText_3_2 = tooltip_3.getElementsByTagName('text')[1];
                var tooltipText_3_3 = tooltip_3.getElementsByTagName('text')[2];
                var tooltipText_3_4 = tooltip_3.getElementsByTagName('text')[3];
        
                var tooltipRects = tooltip_1.getElementsByTagName('rect');
                var triggers = svg.getElementsByClassName('tooltip-trigger');
                for (var i = 0; i < triggers.length; i++) {
                    triggers[i].addEventListener('mousemove', showTooltip);
                    triggers[i].addEventListener('mouseout', hideTooltip);
                }
                function showTooltip(evt) {
                    var CTM = svg.getScreenCTM();
                    var x_unsat =  (evt.clientX - CTM.e) / CTM.a
                    var x = x_unsat;
                    if  (x_unsat>x_max) {var x = x_max}
                    if  (x_unsat<x_min) {var x = x_min}
        
                   
                    var y_unsat = (evt.clientY - CTM.f) / CTM.d + 30;
                    var y = y_unsat;
                    if  (y_unsat>y_max) {var y = y_max - y_height}
                   
                    
                    if (evt.target.getAttributeNS(null, "data0") == "bus") {
                        tooltip_1.setAttributeNS(null, "transform", "translate(" + x + " " + y + ")");
                        tooltip_1.setAttributeNS(null, "visibility", "visible")}
                    
                    if (evt.target.getAttributeNS(null, "data0") == "line") {
                        tooltip_2.setAttributeNS(null, "transform", "translate(" + x + " " + y + ")");
                        tooltip_2.setAttributeNS(null, "visibility", "visible")}
        
                    if (evt.target.getAttributeNS(null, "data0") == "vsc") {
                        tooltip_3.setAttributeNS(null, "transform", "translate(" + x + " " + y + ")");
                        tooltip_3.setAttributeNS(null, "visibility", "visible")}
                        
                    if (evt.target.getAttributeNS(null, "data0") == "trafo") {
                        tooltip_3.setAttributeNS(null, "transform", "translate(" + x + " " + y + ")");
                        tooltip_3.setAttributeNS(null, "visibility", "visible")}    
                        
                    tooltipText_1_1.firstChild.data = evt.target.getAttributeNS(null, "data1");
                    tooltipText_1_2.firstChild.data = evt.target.getAttributeNS(null, "data2");
                    tooltipText_1_3.firstChild.data = evt.target.getAttributeNS(null, "data3");
                    tooltipText_1_4.firstChild.data = evt.target.getAttributeNS(null, "data4");
                    tooltipText_1_5.firstChild.data = evt.target.getAttributeNS(null, "data5");
                    tooltipText_1_6.firstChild.data = evt.target.getAttributeNS(null, "data6");
         
                    tooltipText_2_1.firstChild.data = evt.target.getAttributeNS(null, "data1");
                    tooltipText_2_2.firstChild.data = evt.target.getAttributeNS(null, "data2");
                    tooltipText_2_3.firstChild.data = evt.target.getAttributeNS(null, "data3");
                    tooltipText_2_4.firstChild.data = evt.target.getAttributeNS(null, "data4");
                    tooltipText_2_5.firstChild.data = evt.target.getAttributeNS(null, "data5");
                    tooltipText_2_6.firstChild.data = evt.target.getAttributeNS(null, "data6");
        
                    tooltipText_3_1.firstChild.data = evt.target.getAttributeNS(null, "data1");
                    tooltipText_3_2.firstChild.data = evt.target.getAttributeNS(null, "data2");
                    tooltipText_3_3.firstChild.data = evt.target.getAttributeNS(null, "data3");
                    tooltipText_3_4.firstChild.data = evt.target.getAttributeNS(null, "data4");
                    
        				var length = tooltipText_1_1.getComputedTextLength()+200;
        				for (var i = 0; i < tooltipRects.length; i++) {
        					tooltipRects[i].setAttributeNS(null, "width", length + 8);
        				}
        			}
        			function hideTooltip(evt) {
        				tooltip_1.setAttributeNS(null, "visibility", "hidden");
        				tooltip_2.setAttributeNS(null, "visibility", "hidden");
        				tooltip_3.setAttributeNS(null, "visibility", "hidden");
        			}
        		})()
            ]]>'''
        
        
        
        element = Element('script')
        element.attrib['type'] ="text/ecmascript"
        element.attrib['id'] = "script15"
        element.text = 'scriptplace'
        self.root.append(element)
        
        self.set_lines_currents()
        self.set_buses_voltages()
            
        out = self.tostring().replace('scriptplace',script)
        width = float(self.root.attrib['width'])
        height = float(self.root.attrib['height'])
        document_id = self.root.attrib['id']
        
        tooltip_width = 320
        tooltip_height = 100
        
        out=out.replace('x_max',f'{width-tooltip_width/2}' )
        out=out.replace('x_min',f'{tooltip_width/2}' )
        out=out.replace('y_max',f'{height-tooltip_height}' )
        out=out.replace('y_height',f'{tooltip_height}' )
        out=out.replace('document_id', document_id)
        
        with open(output_file,'w') as fobj:
            fobj.write(out)  
    
            
class animatesvg():
    
    def __init__(self,input_file,group_id):
        ET.register_namespace("","http://www.w3.org/2000/svg")
        ET.register_namespace("inkscape","http://www.inkscape.org/namespaces/inkscape")
        ET.register_namespace("sodipodi","http://sodipodi.sourceforge.net/DTD/sodipodi-0.dtd") 
        ET.register_namespace("cc","http://creativecommons.org/ns#") 
        ET.register_namespace("rdf","http://www.w3.org/1999/02/22-rdf-syntax-ns#")      
        self.tree = ET.parse(input_file)
        self.root = self.tree.getroot()
        self.g_list = self.root.findall(".//{http://www.w3.org/2000/svg}g")
        self.group_id = group_id
        self.N_steps = 1000
        self.input_file = input_file
        self.begin_click = False
        self.begin = ''
        self.anim_id = ''
        self.anim_i = 0
 
    def set_size(self,width,height):
        self.root.attrib['width']  = f'{width}px'
        self.root.attrib['height'] = f'{height}px'
        
    def rotate(self,times,angle,x,y):
        if type(x) == float:
            x = times*0+x
        if type(y) == float:
            y = times*0+y            
        t_end = times[-1]
        N_t = len(times)   
        keyTimes = ""
        keyPoints = ""
        for it in range(N_t):
            keyTimes  += f'{times[it]/t_end};'
            keyPoints += f'{angle[it]},{x[it]},{y[it]};'         
        keyTimes  = keyTimes[:-1].replace("'",'"')    
        keyPoints = keyPoints[:-1].replace("'",'"') 
        
        for item in self.g_list:
            if item.attrib['id'] == self.group_id:

                anim = ET.Element("animateTransform")
                if self.anim_id != '':
                    anim.set('id',self.anim_id)
                anim.set('calcMode',"linear")
                anim.set('additive',"sum")
                anim.set('attributeType',"xml")
                anim.set('attributeName',"transform")
                anim.set('type',"rotate")
                anim.set('dur',f"{t_end}s")
                anim.set('fill',"freeze") 
                #anim.set('repeatCount',"indefinite") 
                if self.begin != '':
                    anim.set('begin',self.begin)
                if self.begin_click:
                    anim.set('begin',"click")
                anim.set('values',f"{keyPoints}")
                anim.set('keyTimes',f"{keyTimes}")  
                item.insert(0, anim)
                
    def translate(self,times,x,y):  
        t_end = times[-1]
        N_t = len(times)   
        keyTimes = ""
        keyPoints = ""
        for it in range(N_t):
            keyTimes  += f'{times[it]/t_end:0.5f};'
            keyPoints += f'{float(x[it]):0.3f},{float(y[it]):0.3f};'         
        keyTimes  = keyTimes[:-1].replace("'",'"')    
        keyPoints = keyPoints[:-1].replace("'",'"') 

        for item in self.g_list:
            if item.attrib['id'] == self.group_id:
                anim = ET.Element("animateTransform")
                if self.anim_id != '':
                    anim.set('id',self.anim_id)
                anim.set('calcMode',"linear")
                anim.set('additive',"sum")
                anim.set('attributeType',"xml")
                anim.set('attributeName',"transform")
                anim.set('type',"translate")
                anim.set('dur',f"{t_end}s")
                anim.set('fill',"freeze") 
                #anim.set('repeatCount',"indefinite") 
                if self.begin != '':
                    anim.set('begin',self.begin)
                if self.begin_click:
                    anim.set('begin',"click")
                anim.set('values',f"{keyPoints}")
                anim.set('keyTimes',f"{keyTimes}")  
                item.insert(0, anim)

    def scale(self,times,x_0,y_0,sx,sy):  
        t_end = times[-1]
        N_t = len(times)   
        keyTimes = ""
        keyPoints = ""
        for it in range(N_t):
            keyTimes  += f'{times[it]/t_end:0.5f};'
            keyPoints += f'{sx[it]:0.3f},{sy[it]:0.3f};'         
        keyTimes  = keyTimes[:-1].replace("'",'"')    
        keyPoints = keyPoints[:-1].replace("'",'"') 

        for item in self.g_list:
            if item.attrib['id'] == self.group_id:
                anim = ET.Element("animateTransform")
                if self.anim_id != '':
                    anim.set('id',self.anim_id)
                
                anim.set('calcMode',"linear")
                anim.set('additive',"sum")
                anim.set('attributeType',"xml")
                anim.set('attributeName',"transform")
                anim.set('type',"scale")
                anim.set('dur',f"{t_end}s")
                anim.set('fill',"freeze") 
                #anim.set('repeatCount',"indefinite") 
                if self.begin != '':
                    anim.set('begin',self.begin)
                if self.begin_click:
                    anim.set('begin',"click")
                anim.set('values',f"{keyPoints}")
                anim.set('keyTimes',f"{keyTimes}")  
                item.insert(0, anim)
        x = times*0+x_0*(1-sx)
        y = times*0+y_0*(1-sy)
        self.translate(times,x,y)


    def save(self,output_file=''):
        if output_file=='':
            output_file = f"{self.input_file.replace('.svg','')}_anim.svg"
        self.tree.write(output_file)
        
    def reduce(self,times,values):
        N_t = len(times)
        mask_values = values
        
        mask = np.array([True]*N_t)
        increment = np.abs((mask_values.max()-mask_values.min())/self.N_steps)
        mask[0:(N_t-1)] =  np.abs(np.diff(mask_values))>increment
        mask[0] = True
        mask[-1] = True
        
        self.mask = mask
        self.increment = increment
        return self.mask
    
    
def grid2svg(input_json,output_svg):
    '''
    

    Parameters
    ----------
    input_json : if string name of the file containing the grid data.
                 if dict, a dictionary with the grid data.
    
    output_svg : string
        path for the output SVG file.

    Returns
    -------
    None.

    '''
    

    if type(input_json) == dict:
        data = input_json
        
    if type(input_json) == str:
        json_data = open(input_json).read().replace("'",'"')
        data = json.loads(json_data)
                
                
                
    

    dwg = svgwrite.Drawing(output_svg, profile='full', size=(640, 800))

    buses = data['buses']
    lines = data['lines']
    trafos = data['transformers']

    offset_x = 100
    offset_y = 50
    scale_x = 1.5
    scale_y = 1.5

    bus2posxy = {}
    for bus in buses:
        pos_x = (bus['pos_x']+offset_x)*scale_x
        pos_y = (offset_y-bus['pos_y'])*scale_y
        bus2posxy.update({bus['bus']:{'pos_x':pos_x,'pos_y':pos_y}})

        dwg.add(svgwrite.shapes.Rect(insert=(pos_x-2.5, pos_y-2.5), size=(15, 15),id = bus['bus']))
        #dwg.add(dwg.text(bus['bus'], insert=(pos_x+3, pos_y-3), fill='black'))
        dwg.add(dwg.text(bus['bus'], insert=(pos_x+5, pos_y-5), fill='black',id = f"{bus['bus']}_tt")) #,visibility='hidden'))

        #a = svgwrite.animate.Set(href=f"{bus['bus']}_tt", attributeName="visibility",
                             #begin="hidden",
        #                     to="visible",
        #                     begin=f"{bus['bus']}.mouseover",
        #                     end=f"{bus['bus']}.mouseout")

        #dwg.add(a)

    for line in lines:

        x1 = bus2posxy[line['bus_j']]['pos_x']
        y1 = bus2posxy[line['bus_j']]['pos_y']
        x2 = bus2posxy[line['bus_k']]['pos_x']
        y2 = bus2posxy[line['bus_k']]['pos_y']
        bus_j = line['bus_j']
        bus_k = line['bus_k']

        v = 0.0
        h = 0.0

        vertical = True
        Dv = 0.0
        Dh = 3.0
        if np.abs(x2 - x1) < np.abs(y2 - y1):
            Dv = 3.0
            Dh = 0.0
            vertical = False

        for ph in ['a','b','c','n']:
            dwg.add(dwg.line(start=(x1+v, y1+h), end=(x2+v,y2+h), 
                             stroke=svgwrite.rgb(10, 10, 16, '%'), id = f'l_{bus_j}_{bus_k}_{ph}',
                             stroke_width=2))
            v += Dv
            h += Dh

    for trafo in trafos:

        x1 = bus2posxy[trafo['bus_j']]['pos_x']
        y1 = bus2posxy[trafo['bus_j']]['pos_y']
        x2 = bus2posxy[trafo['bus_k']]['pos_x']
        y2 = bus2posxy[trafo['bus_k']]['pos_y']

        dwg.add(dwg.line(start=(x1, y1), end=(x2,y2), stroke=svgwrite.rgb(10, 10, 16, '%')))


    dwg.save()
    
    
def results2svg(grid,data_input,svg_input,svg_output):
    
    
    if type(data_input) == dict:
        data = data_input
        
    if type(data_input) == str:
        data_input = open(data_input).read().replace("'",'"')
        data = json.loads(data_input)

    svg_new = svg(svg_input)

    buses = data['buses']
    lines = data['lines']

    trafos = data['transformers']

    for line in lines:
        #if not 'monitor'in line or not 'vsc_line' in line: continue
        if 'monitor' in line:
            if not line['monitor']: continue
        if 'vsc_line' in line:
            if not line['vsc_line']: continue
        bus_j = line['bus_j']
        bus_k = line['bus_k']

        for ph in ['a','b','c','n']:

            I_max = data["line_codes"][line['code']]['I_max']
            i_r = grid.get_value(f'i_l_{bus_j}_{bus_k}_{ph}_r') 
            i_i = grid.get_value(f'i_l_{bus_j}_{bus_k}_{ph}_i') 
            i = i_r + 1j*i_i
            i_abs = np.abs(i)
            #print(f'l_{bus_j}_{bus_k}_{ph} = {i_abs:8.1f} A')
            if i_abs < 1e-3: continue
            i_sat = np.clip(i_abs**2/I_max**2*255,0,255)
            svg_new.set_color('line',f'l_{bus_j}_{bus_k}_{ph}',(int(i_sat),0,0))

    for bus in buses:
        #if not 'monitor'in line or not 'vsc_line' in line: continue

        
        U_nom  = bus['U_kV']
        v_r = grid.get_value(f'v_{bus["bus"]}_a_r') 
        v_i = grid.get_value(f'v_{bus["bus"]}_a_i') 
        v = v_r + 1j*v_i
        v_abs = np.abs(v)
        #print(f'l_{bus_j}_{bus_k}_{ph} = {i_abs:8.1f} A')
        if 'acdc' in bus:
            if bus['acdc'] == 'DC':
                red  = np.clip(((v_abs/(1e3*U_nom))**2-1)*255,0,255)
                blue = np.clip(3*(1-(v_abs/(1e3*U_nom))**2)*255,0,255)
                svg_new.set_color('rect',f'{bus["bus"]}',(int(red),0,int(blue)))  
            if bus['acdc'] == 'AC':
                red  = np.clip(((np.sqrt(3)*v_abs/(1e3*U_nom))**2-1)*255,0,255)
                blue = np.clip(3*(1-(np.sqrt(3)*v_abs/(1e3*U_nom))**2)*255,0,255)
                svg_new.set_color('rect',f'{bus["bus"]}',(int(red),0,int(blue)))  
        else:
            red  = np.clip(((np.sqrt(3)*v_abs/(1e3*U_nom))**2-1)*255,0,255)
            blue = np.clip(3*(1-(np.sqrt(3)*v_abs/(1e3*U_nom))**2)*255,0,255)
            svg_new.set_color('rect',f'{bus["bus"]}',(int(red),0,int(blue)))

    svg_new.save(svg_output)
    
    return svg_new