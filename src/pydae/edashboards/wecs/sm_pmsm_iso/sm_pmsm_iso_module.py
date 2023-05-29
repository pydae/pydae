import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
import pydae.svg_tools as st 
import pydae.build_cffi as db
from pydae.bmapu import bmapu_builder


import ipywidgets
plt.style.use('https://raw.githubusercontent.com/pydae/pydae/master/src/pydae/edashboards/presentation.mplstyle')
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
plt.rcParams['font.family'] =  ['sans-serif']

class dashboard():
    
    def __init__(self):

        
        self.colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        self.data = {
        "system":{"name":"sm_pmsm_iso","S_base":100e6, "K_p_agc":0.01,"K_i_agc":0.01,"K_xif":0.0},       
        "buses":[{"name":"1", "P_W":0.0,"Q_var":0.0,"U_kV":0.4},
                 {"name":"2", "P_W":0.0,"Q_var":0.0,"U_kV":0.4},
                 {"name":"3", "P_W":0.0,"Q_var":0.0,"U_kV":0.4},
                ],
        "lines":[
            {"bus_j":"1", "bus_k":"2", "X_pu":0.05,"R_pu":0.01,"Bs_pu":0,"S_mva":2.0},
            {"bus_j":"2", "bus_k":"3", "X_pu":0.05,"R_pu":0.01,"Bs_pu":0,"S_mva":10.0}
                 ],
        "wecs":[
            {"type":"pmsm_1","bus":"1","S_n":1e6,
                "H_t":4.0,"H_r":1.0, "w_tr":5.0, "d_tr":0.01,
                "R_m":0.01,"L_m":0.05,"Phi_m":1.0,
                "R_s":0.02,"X_s":0.05,
                "K_pdc":0.1,"C_dc":0.5}],
        "syns":[
                {"bus":"2","S_n":2e6,
                    "X_d":1.81,"X1d":0.3, "T1d0":8.0,    
                    "X_q":1.81,"X1q":0.55,"T1q0":0.4,  
                    "R_a":0.02,"X_l": 0.2, 
                    "H":5.0,"D":1.0,
                    "Omega_b":314.1592653589793,"omega_s":1.0,"K_sec":1.0,
                    "avr":{"type":"sexs","K_a":100.0,"T_a":0.1,"T_b":0.1,"T_e":0.1,"E_min":-10.0,"E_max":10.0,"v_ref":1.0},
                    "gov":{"type":"tgov1","Droop":0.05,"T_1":1.0,"T_2":1.0,"T_3":1.0,"D_t":0.0,"p_c":0.0,"K_sec":1.0},
                    "K_delta":0.01}]
        }

    def ini(self):

        import sm_pmsm_iso

        self.model = sm_pmsm_iso.model()
        self.model.Dt = 0.01
        self.model.decimation = 1

        nu_w =11.0
        omega_0 = np.clip(nu_w/10,0.4,1.2)
        xy_0 = {
            "V_1": 1.0,
            "theta_1": 0.0,
            "V_2": 1.0,
            "theta_2": 0.0,
            "V_3": 1.0,
            "theta_3": 0.0,
            "omega_coi": 1.0,
            "omega_3": 1.0,
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

        self.P_l_0 = 2.3e6
        params = {'K_pdc_1':50,"C_dc_1":0.01,"nu_w_1":nu_w,"P_3":-self.P_l_0,'K_a_2':200,
                "K_p_beta_1":1,"K_i_beta_1":1.0,"T_beta_1":1.2,"p_ref_ext_1":-0.0} 
        params.update({'K_p_pll_1':10, 'K_i_pll_1':100.0, 'T_pll_1':0.1,
                    'K_f_1':0, 'K_h_1':0})


        self.model.ini(params,xy_0)
        
        self.model.run(20.0,{})
        self.model.post();
           
    def build(self):



        grid = bmapu_builder.bmapu(self.data)
        #grid.checker()
        grid.uz_jacs = True
        grid.verbose = False
        grid.build('sm_pmsm_iso')

    def widgets(self):
        model = self.model
        colors = self.colors

        plt.ioff()
        plt.clf()

        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(9, 4))

        self.line_p_g_2 = axes[0,0].plot(model.Time,model.get_values('p_g_2')*model.get_value('S_n_2')/1e6, label='SM Power (MW)')
        self.line_p_s_1 = axes[0,0].plot(model.Time,model.get_values('p_s_1')*model.get_value('S_n_1')/1e6, label='WECS Power (MW)')

        self.line_omega_2 = axes[1,0].plot(model.Time, model.get_values('omega_2')*50, label='System frequency (Hz)', color=colors[1])
 
        self.line_omega_t = axes[0,1].plot(model.Time, model.get_values('omega_t_1'), label='WECS Turbine Speed (pu)')
        self.line_p_w = axes[1,1].plot(model.Time, model.get_values('p_w_1'), label='WECS Turbine Power (pu)')

        axes[0,0].set_ylim((0,3))
        axes[1,0].set_ylim((0.95*50,1.02*50))
        axes[0,1].set_ylim((0.8,1.3))
        axes[1,1].set_ylim((0.0,1.0))

        axes[0,0].grid(True)
        axes[1,0].grid(True)
        axes[0,1].grid(True)
        axes[1,1].grid(True)
        axes[0,0].legend(loc='best')
        axes[1,0].legend(loc='best')
        axes[0,1].legend(loc='best')
        axes[1,1].legend(loc='best')

        axes[1,0].set_xlabel('Time (s)')  
        axes[1,1].set_xlabel('Time (s)') 

        #self.fig.canvas.draw_idle()
        fig.savefig('results.svg')

        self.s = st.svg(r"results.svg")

        self.html = ipywidgets.HTML(
            value= self.s.tostring(),
            placeholder='',
            description='',
        )
        
        #axes[0].set_title('Par en función de la velocidad')
        #axes[1].set_title('Corriente en función de la velocidad')


        self.sld_P_l = ipywidgets.FloatSlider(orientation='horizontal',description = "P<sub>l</sub>", 
                                        value=0, min=0.0,max= 1.0, 
                                        step=0.1)
        self.sld_Q_l = ipywidgets.FloatSlider(orientation='horizontal',description = "Q<sub>l</sub>", 
                                        value=0, min=-100.0,max= 100, 
                                        step=1)


        self.sld_K_f = ipywidgets.FloatSlider(orientation='horizontal',description = "K<sub>f</sub>", 
                                        value=model.get_value('K_f_1'), min=0.0,max= 30, 
                                        step=1)


        self.sld_K_h = ipywidgets.FloatSlider(orientation='horizontal',description = "K<sub>h</sub>", 
                                        value=model.get_value('K_h_1'), min=0,max= 10, 
                                        step=1)

        self.fig = fig
        self.axes = axes

    def update(self,change):
        
        model = self.model

        DP_l = self.sld_P_l.value
        DQ_l = self.sld_Q_l.value
        K_f_1 = self.sld_K_f.value
        K_h_1 = self.sld_K_h.value

        model.Dt = 0.01
        model.ini({'P_3':-self.P_l_0,'Q_3':-0.0,'K_f_1':K_f_1,'K_h_1':K_h_1},'xy_0.json')
        model.run( 1.0,{})
        model.run(10,{'P_3':-DP_l*1e6-self.P_l_0})
        model.Dt = 0.1
        model.run(20,{'P_3':-DP_l*1e6-self.P_l_0})

        model.post();


        self.line_p_g_2[0].set_data(model.Time,model.get_values('p_g_2')*model.get_value('S_n_2')/1e6)
        self.line_p_s_1[0].set_data(model.Time,model.get_values('p_s_1')*model.get_value('S_n_1')/1e6)

        self.line_omega_2[0].set_data(model.Time, model.get_values('omega_2')*50)

        self.line_p_w[0].set_data(model.Time, model.get_values('p_w_1'))
        self.line_omega_t[0].set_data(model.Time, model.get_values('omega_t_1'))

        #self.fig.canvas.draw_idle()

        self.fig.savefig('results_1.svg')
        self.s = st.svg(r"results_1.svg")
        self.html.value = self.s.tostring()

    def show(self):

        self.sld_P_l.observe(self.update, names='value')
        #self.sld_Q_l.observe(self.update, names='value')

        self.sld_K_h.observe(self.update, names='value')
        self.sld_K_f.observe(self.update, names='value')

        self.update(0)
        layout_row1 = ipywidgets.HBox([self.html])
        PQ = ipywidgets.VBox([self.sld_P_l])
        Ks = ipywidgets.VBox([self.sld_K_f,self.sld_K_h])
        layout_row2 = ipywidgets.HBox([PQ,Ks])

        self.layout = ipywidgets.VBox([layout_row1,layout_row2])
        display(self.layout)
        

    
if __name__ == '__main__':
    from pydae.edashboards.wecs.sm_pmsm_iso.sm_pmsm_iso_module import dashboard
    db = dashboard()
    db.build()

    db.ini()
    db.widgets()
    db.show()
    db.update(0)
        