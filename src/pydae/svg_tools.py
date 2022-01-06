# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 10:33:57 2021

@author: jmmau
"""

from xml.etree import ElementTree as ET
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
                
    def set_color(self,type_,id_,rgb):
        if type_ == 'rect':
            self.set_rect_style(id_,f"fill:#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}")
            
        if type_ == 'line':
            self.set_line_style(id_,f"stroke:#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}")
            
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
        
    def set_lines_currents(self):
        
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

    def set_buses_voltages(self):
        
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