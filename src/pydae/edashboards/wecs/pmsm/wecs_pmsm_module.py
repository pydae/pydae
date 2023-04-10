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
        self.data = {
        "system":{"name":"wecs_pmsm_test","S_base":100e6, "K_p_agc":0.0,"K_i_agc":0.0,"K_xif":0.01},       
        "buses":[{"name":"1", "P_W":0.0,"Q_var":0.0,"U_kV":20.0},
                    {"name":"2", "P_W":0.0,"Q_var":0.0,"U_kV":20.0}
                ],
        "lines":[{"bus_j":"1", "bus_k":"2", "X_pu":0.05,"R_pu":0.01,"Bs_pu":1e-6,"S_mva":2.0}],
        "wecs":[
            {"type":"pmsm_1","bus":"1","S_n":1e6,
                "H_t":4.0,"H_r":1.0, "w_tr":5.0, "d_tr":0.01,
                "R_m":0.01,"L_m":0.05,"Phi_m":1.0,
                "R_s":0.02,"X_s":0.05,
                "K_pdc":0.1,"C_dc":0.5}],
        "genapes":[{"bus":"2","S_n":1e9,"F_n":50.0,"X_v":0.001,"R_v":0.0,"K_delta":0.001,"K_alpha":1e-6}]
        }

    def ini(self):

        import wecs_pmsm

        self.model = wecs_pmsm.model()
        self.model.Dt = 0.01
        self.model.decimation = 1
        self.model.ini({'K_pdc_1':50,"C_dc_1":0.01,"nu_w_1":10,
           "K_p_beta_1":1,"K_i_beta_1":1.0,"T_beta_1":1.0},'xy_0.json')
        
        
    def build(self):



        grid = bmapu_builder.bmapu(self.data)
        #grid.checker()
        grid.uz_jacs = True
        grid.verbose = False
        grid.build('wecs_pmsm')


    def widgets(self):
        
        model = self.model
        colors = self.colors

        plt.ioff()
        plt.clf()

        self.s = st.svg(r"wecs_pmsm_dashboard.svg")

        self.html = widgets.HTML(
            value= self.s.tostring(),
            placeholder='',
            description='',
        )

        # tab_0 = widgets.VBox([widgets.Text(description='Sn')])
        self.sld_nu_w = widgets.FloatSlider(description='Wind (m/s)',min=4, max=20, step=1, value=10)
        # self.tab_1_sld_temp_deg = widgets.FloatSlider(description='temp (deg)',min=10, max=45, step=1, value=25)
        tab_env = widgets.HBox([self.sld_nu_w])

        # tab_2_0 = widgets.RadioButtons(
        #     options=['Manual', 'Power from MPPT', 'Power from speed control'],
        #     description='Mode:',    disabled=False)
        # self.sld_p_s = widgets.FloatSlider(description='p<sub>s</sub>*',min=0, max=1, step=0.1, value=0)
        # self.sld_v_dc = widgets.FloatSlider(description='v<sub>dc</sub>*',min=0.8, max=2.0, step=0.1, value=1.2)
        self.sld_q_s = widgets.FloatSlider(description='q<sub>s</sub>*',min=-1, max=1, step=0.1, value=0)
        # self.sld_i_d = widgets.FloatSlider(description='i<sub>d</sub>*',min=-1, max=1, step=0.1, value=0)
        # self.sld_i_q = widgets.FloatSlider(description='i<sub>q</sub>*',min=-0.1, max=0.1, step=0.1, value=0)

        # tab_2_1 = widgets.VBox([self.sld_p_s,
        #                         ])
        # tab_2 = widgets.HBox([tab_2_0,tab_2_1])

        # self.options_mode = widgets.RadioButtons(
        #     options=['Normal', 'LVRT'],
        #     description='Mode:',    disabled=False)

        tab_vsc_g = widgets.HBox([widgets.VBox([self.sld_q_s])])
        # tab_3 = widgets.HBox([tab_3_0])

        # self.sld_V_g = widgets.FloatSlider(description='V<sub>g</sub>',min=0.2, max=1.2, step=0.1, value=1.0)
        # self.sld_SCR = widgets.FloatSlider(description='SCR',min=1, max=10, step=1, value=10)

        self.sld_p_ext = widgets.FloatSlider(description='p<sub>ext</sub>',min=-1.0, max=0, step=0.1, value=0.0)
        self.sld_beta_ext = widgets.FloatSlider(description='𝛽<sub>ext</sub>',min=0.0, max=60, step=5.0, value=0.0)

        tab_courtailment = widgets.HBox([self.sld_p_ext,self.sld_beta_ext])

        # tab_grid = widgets.HBox([self.sld_V_g,self.sld_SCR])

        self.tab = widgets.Tab()
        self.tab.children = [tab_env,tab_vsc_g,tab_courtailment]

        #tab.set_title(0, 'PV')
        self.tab.set_title(0, 'Enviroment')
        #tab.set_title(2, 'MPPT')
        self.tab.set_title(1, 'VSC-G')
        self.tab.set_title(2, 'Courtailment')
        # self.tab.set_title(2, 'Grid')

        nu_w =self.sld_nu_w.value
        omega_0 = np.clip(nu_w/10,0.4,1.2)
        xy_0 = {
            "V_1": 1.0,
            "theta_1": 0.0,
            "V_2": 1.0,
            "theta_2": 0.0,
            "omega_coi": 1.0,
            "omega_2": 1.0,
            "theta_tr_1": 0.0,
            "omega_t_1": omega_0,
            "omega_r_1": omega_0,
            "xi_beta_1":1,
            "beta_1":0.0,
            "i_sr_1": 1,
            "v_dc_1": 1.5,
            "p_w_mppt_lpf_1":1.2,
            "v_mq_1":1.0,
            "v_tq_ref_1":1.0
        }


        self.model.ini({'K_pdc_1':50,"C_dc_1":0.01,"nu_w_1":nu_w,
                "K_p_beta_1":1,"K_i_beta_1":1.0,"T_beta_1":1.2},xy_0)#'xy_0.json')
        
    def update(self,change):
        
        model = self.model

        nu_w =self.sld_nu_w.value
        Omega_r_max = 1.35
        omega_0 = np.clip(nu_w/10,0.4,Omega_r_max)
        beta_0 = np.clip((nu_w-13)*3,0,90)
        xy_0 = {
            "V_1": 1.0,
            "theta_1": 0.0,
            "V_2": 1.0,
            "theta_2": 0.0,
            "omega_coi": 1.0,
            "omega_2": 1.0,
            "theta_tr_1": 0.0,
            "omega_t_1": omega_0,
            "omega_r_1": omega_0,
            "xi_beta_1":beta_0,
            "beta_1":beta_0,
            "i_sr_1": 1,
            "v_dc_1": 1.5,
            "p_w_mppt_lpf_1":omega_0,
            "v_mq_1":1.0,
            "v_tq_ref_1":1.0
        }

        self.model.ini({'K_pdc_1':50,"C_dc_1":0.01,"nu_w_1":nu_w,'q_s_ref_1':self.sld_q_s.value,
                "K_p_beta_1":1,"K_i_beta_1":1.0,"T_beta_1":1.2,"Omega_r_max_1":Omega_r_max,
                "beta_ext_1":self.sld_beta_ext.value,"p_ref_ext_1":self.sld_p_ext.value},xy_0)#'xy_0.json')

        # s.set_tspan('p_g', f'={model.get_value("p_g_"):0.2f}')
        # s.set_tspan('q_g', f'={model.get_value("q_g_"):0.2f}')
        I_s = (model.get_value("i_si_1")**2 + model.get_value("i_sr_1")**2)**0.5
        self.s.set_tspan('I_s', f'={I_s:4.2f}')
        self.s.set_tspan('i_sd_ref', f'={model.get_value("i_sd_ref_1"):5.2f}')
        self.s.set_tspan('i_sq_ref', f'={model.get_value("i_sq_ref_1"):5.2f}')
        self.s.set_tspan('i_md_ref', f'={0:5.2f}')
        self.s.set_tspan('i_mq_ref', f'={model.get_value("i_mq_ref_1"):5.2f}')
        # self.s.set_tspan('m', f'={model.get_value("m_ref_1"):0.2f}')
        self.s.set_tspan('p_s', f'={model.get_value("p_s_1"):5.2f}')
        self.s.set_tspan('q_s', f'={model.get_value("q_s_1"):5.2f}')
        self.s.set_tspan('p_m', f'={model.get_value("p_m_1"):5.2f}')
        self.s.set_tspan('q_m', f'={model.get_value("q_m_1"):5.2f}')
        self.s.set_tspan('v_dc', f'={model.get_value("v_dc_1"):0.2f}')
        self.s.set_tspan('nu_w', f'={model.get_value("nu_w_1"):3.0f} m/s')
        self.s.set_tspan('beta', f'={model.get_value("beta_1"):4.1f}º')
        self.s.set_tspan('omega_t', f'={model.get_value("omega_t_1"):5.2f}')
        self.s.set_tspan('V_pcc', f'={model.get_value("V_1"):5.2f}')
        self.s.set_tspan('V_g', f'={model.get_value("V_2"):5.2f}')
        # s.set_tspan('beta', f'={np.abs(model.get_value("beta_")):5.2f}')

        self.html.value = self.s.tostring() 


    def show(self):

        # Link the slider value to the text element update function
        self.sld_nu_w.observe(self.update, names='value')
        #self.tab_1_sld_temp_deg.observe(self.update, names='value')
        # self.sld_v_dc.observe(self.update, names='value')
        self.sld_q_s.observe(self.update, names='value')
        self.sld_p_ext.observe(self.update, names='value')
        self.sld_beta_ext.observe(self.update, names='value')
        # self.sld_i_d.observe(self.update, names='value')
        # self.sld_i_q.observe(self.update, names='value')
        # self.sld_V_g.observe(self.update, names='value')
        # self.sld_SCR.observe(self.update, names='value')
        # self.options_mode.observe(self.update, names='value')

        self.update(0)
        self.layout = widgets.VBox([self.html,self.tab])
        display(self.layout)