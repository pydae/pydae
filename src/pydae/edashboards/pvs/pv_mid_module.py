import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
import pydae.svg_tools as st 
import pydae.build_cffi as db
from pydae.bmapu import bmapu_builder

plt.style.use('https://raw.githubusercontent.com/pydae/pydae/master/src/pydae/edashboards/presentation.mplstyle')
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

class dashboard():
    
    def __init__(self):

        
        self.colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        self.params = {'v_dc_ref_1':1.35,"K_pdc_1":100,'irrad_1':1000,'mode_1':2}

        self.data_pv = {"I_sc":3.87,"V_oc":42.1,"I_mpp":3.56,"V_mpp":33.7,"N_s":72,
        "K_vt":-0.160,"K_it":0.065,"R_pv_s": 0.5602, "R_pv_sh": 1862, "K_d": 1.3433}

    def ini(self):

        import pv_mid

        self.model = pv_mid.model()
        self.model.Dt = 0.01
        self.model.decimation = 1
        self.model.ini(self.params,'xy_0.json')
        
        
    def build(self):

        data = {
        "system":{"name":"smib","S_base":100e6, "K_p_agc":0.0,"K_i_agc":0.0,"K_xif":0.01},       
        "buses":[{"name":"1", "P_W":0.0,"Q_var":0.0,"U_kV":20.0},
                    {"name":"2", "P_W":0.0,"Q_var":0.0,"U_kV":20.0}
                ],
        "lines":[{"bus_j":"1", "bus_k":"2", "X_pu":0.05,"R_pu":0.0,"Bs_pu":0.0,"S_mva":100.0}],
        "pvs":[
            {"type":"pv_1","bus":"1","S_n":1e6,"U_n":400.0,      
             "R_s":0.01,"X_s":0.05,
             "K_pdc":100,"C_dc":10.5,
             "I_sc":8,"V_oc":42.1,"I_mpp":3.56,"V_mpp":33.7,"N_s":72,
             "K_vt":-0.160,"K_it":0.065,"R_pv_s": 0.5602, "R_pv_sh": 1862, "K_d": 1.3433,
             "N_ms":25,"N_mp":250}],
        "genapes":[{"bus":"2","S_n":100e6,"F_n":50.0,"X_v":0.001,"R_v":0.0,"K_delta":0.001,"K_alpha":1e-6}]
        }

        data['pvs'][0].update(self.data_pv)

        grid = bmapu_builder.bmapu(data)
        #grid.checker()
        grid.uz_jacs = True
        grid.verbose = False
        grid.build('pv_mid')

    def widgets(self):
        
        model = self.model
        colors = self.colors

        plt.ioff()
        plt.clf()

        self.s = st.svg(r"pv_dashboard.svg")

        self.html = widgets.HTML(
            value= self.s.tostring(),
            placeholder='',
            description='',
        )

        tab_0 = widgets.VBox([widgets.Text(description='Sn')])
        self.tab_1_sld_irrad = widgets.FloatSlider(description='irrad',min=0, max=1200, step=100, value=1000)
        self.tab_1_sld_temp_deg = widgets.FloatSlider(description='temp (deg)',min=10, max=75, step=1, value=25)
        tab_1 = widgets.HBox([self.tab_1_sld_irrad,self.tab_1_sld_temp_deg])

        tab_2_0 = widgets.RadioButtons(
            options=['Manual', 'Power from MPPT', 'Power from speed control'],
            description='Mode:',    disabled=False)
        self.sld_p_s = widgets.FloatSlider(description='p<sub>s</sub>*',min=0, max=1, step=0.1, value=0)
        self.sld_v_dc = widgets.FloatSlider(description='v<sub>dc</sub>*',min=0.8, max=2.0, step=0.1, value=1.2)
        self.sld_q_s = widgets.FloatSlider(description='q<sub>s</sub>*',min=-1, max=1, step=0.1, value=0)
        self.sld_i_d = widgets.FloatSlider(description='i<sub>d</sub>*',min=-1, max=1, step=0.1, value=0)
        self.sld_i_q = widgets.FloatSlider(description='i<sub>q</sub>*',min=-0.1, max=0.1, step=0.1, value=0)

        tab_2_1 = widgets.VBox([self.sld_p_s,
                                ])
        tab_2 = widgets.HBox([tab_2_0,tab_2_1])

        self.options_mode = widgets.RadioButtons(
            options=['Normal', 'LVRT'],
            description='Mode:',    disabled=False)

        tab_3_0 = widgets.HBox([self.options_mode,widgets.VBox([self.sld_v_dc,self.sld_q_s]),widgets.VBox([self.sld_i_d,self.sld_i_q])])
        tab_3 = widgets.HBox([tab_3_0])

        self.sld_V_g = widgets.FloatSlider(description='V<sub>g</sub>',min=0.2, max=1.2, step=0.1, value=1.0)
        self.sld_SCR = widgets.FloatSlider(description='SCR',min=10, max=100, step=1, value=10)

        tab_grid = widgets.HBox([self.sld_V_g,self.sld_SCR])

        self.tab = widgets.Tab()
        self.tab.children = [tab_1,tab_3,tab_grid]

        #tab.set_title(0, 'PV')
        self.tab.set_title(0, 'Enviroment')
        #tab.set_title(2, 'MPPT')
        self.tab.set_title(1, 'VSC')
        self.tab.set_title(2, 'Grid')

        self.params.update({'v_dc_ref_1':self.sld_v_dc.value,
                            "K_pdc_1":100,'irrad_1':1000,'b_1_2':-0.1,'g_1_2':0.0})

        model.ini(self.params,'xy_0.json')
        
    def update(self,change):
        
        model = self.model
        S_n_1_MVA = model.get_value('S_n_1')/1e6
        S_b_s_MVA = 100.0
        b_1_2 = -self.sld_SCR.value*S_n_1_MVA/S_b_s_MVA
        self.params.update({'v_dc_ref_1':self.sld_v_dc.value,"K_pdc_1":100,'q_s_ref_1':self.sld_q_s.value,
                'irrad_1':self.tab_1_sld_irrad.value,'temp_deg_1':self.tab_1_sld_temp_deg.value,
                'i_sd_i_ref_1':self.sld_i_d.value, 'i_sq_i_ref_1':self.sld_i_q.value, 
                'b_1_2':b_1_2,
                'v_ref_2':self.sld_V_g.value})
        model.ini(self.params,'xy_0.json')
        
        if self.options_mode.value == 'LVRT':
            model.ini({'v_dc_ref_1':1.8,"K_pdc_1":1,'q_s_ref_1':0.0,
                    'irrad_1':self.tab_1_sld_irrad.value,'temp_deg_1':self.tab_1_sld_temp_deg.value,
                    'i_sd_i_ref_1':self.sld_i_d.value, 'i_sq_i_ref_1':self.sld_i_q.value, 
                    'b_1_2':b_1_2,
                    'v_ref_2':self.sld_V_g.value})

        # s.set_tspan('p_g', f'={model.get_value("p_g_"):0.2f}')
        # s.set_tspan('q_g', f'={model.get_value("q_g_"):0.2f}')
        I_s = (model.get_value("i_si_1")**2 + model.get_value("i_sr_1")**2)**0.5
        p_s = model.get_value("p_s_1")
        q_s = model.get_value("q_s_1")   
        s_s = p_s+ 1j*q_s
        phi_s = np.angle(s_s) 
        pf_s = np.sign(q_s)*np.cos(phi_s)

        self.s.set_tspan('I_s', f'={I_s:4.2f}')
        self.s.set_tspan('i_d_ref', f'={model.get_value("i_sd_ref_1"):5.2f}')
        self.s.set_tspan('i_q_ref', f'={model.get_value("i_sq_ref_1"):5.2f}')
        self.s.set_tspan('m', f'={model.get_value("m_ref_1"):0.2f}')
        self.s.set_tspan('p_s', f'={p_s:5.2f}')
        self.s.set_tspan('q_s', f'={q_s:5.2f}')
        self.s.set_tspan('v_dc', f'={model.get_value("v_dc_1"):0.2f}')
        self.s.set_tspan('i_dc', f'={model.get_value("i_pv_pu_1"):0.2f}')
        self.s.set_tspan('v_pv', f'={model.get_value("v_pv_1"):4.1f} V')
        self.s.set_tspan('i_pv', f'={model.get_value("i_pv_1"):4.1f} A')
        self.s.set_tspan('p_pv', f'={model.get_value("p_pv_1"):5.1f} W')
        self.s.set_tspan('V_pcc', f'={model.get_value("V_1"):5.2f}')
        self.s.set_tspan('V_g', f'={model.get_value("V_2"):5.2f}')
        self.s.set_tspan('pf_s', f'={pf_s:5.2f}')

        # s.set_tspan('beta', f'={np.abs(model.get_value("beta_")):5.2f}')

        self.html.value = self.s.tostring() 


    def show(self):

        # Link the slider value to the text element update function
        self.tab_1_sld_irrad.observe(self.update, names='value')
        self.tab_1_sld_temp_deg.observe(self.update, names='value')
        self.sld_v_dc.observe(self.update, names='value')
        self.sld_q_s.observe(self.update, names='value')
        self.sld_i_d.observe(self.update, names='value')
        self.sld_i_q.observe(self.update, names='value')
        self.sld_V_g.observe(self.update, names='value')
        self.sld_SCR.observe(self.update, names='value')
        self.options_mode.observe(self.update, names='value')

        self.update(0)
        self.layout = widgets.VBox([self.html,self.tab])
        display(self.layout)