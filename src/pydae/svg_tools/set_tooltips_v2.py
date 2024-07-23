
from xml.etree.ElementTree import Element
import numpy as np



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

    if 'transformers' in s.grid_data:
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
    
    set_lines_currents_v2(self)
    set_buses_voltages_v2(self)
        
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
                if 'code' in line:
                    I_max = self.grid_data["line_codes"][line['code']]['I_max']
                else:
                    I_max = line['I_max']

                
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