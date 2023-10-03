import pydae.build_cffi as db
from pydae.bmapu import bmapu_builder
from pydae.build_v2 import builder

import matplotlib.pyplot as plt
plt.style.use(r'https://raw.githubusercontent.com/pydae/pydae/master/src/pydae/edashboards/presentation.mplstyle')

import numpy as np
import ipywidgets
import pydae.svg_tools as st 
 


class dashboard():
    
    def __init__(self):

        pass


    def ini(self):
    
        import spvib_dq_d

        model = spvib_dq_d.model()
        params = {}
        params.update({f'irrad_POI':1000+(np.random.rand()-0.5)*0})
        params.update({f'p_s_ppc_POI':1.0,f'q_s_ppc_POI':0})
        params.update({f'N_pv_s_POI':20, f'N_pv_p_POI':200})

        #model.report_u()
        #model.report_x()
        #model.report_y()
        model.ini(params,'xy_0.json')

        model.run(3.0,{})
        model.post();
        
        self.model = model
        
        #plt.style.use('presentation.mplstyle')
        self.colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        #self.widgets()
        
    def widgets(self):
        
        model = self.model
        colors = self.colors

        plt.ioff()
        plt.clf()


        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(9, 4), frameon=False)
        fig.canvas.toolbar_visible = False

        self.line_p = axes[0].plot(model.Time, model.get_values('p_s_POI'), label='$\sf p$', color=colors[0])
        self.line_q = axes[1].plot(model.Time, model.get_values('q_s_POI'), label='$\sf q$', color=colors[1])

        # self.line_omega = axes[1,0].plot(model.Time, model.get_values('omega_1'), label='$\sf \omega$', color=colors[1])
        # self.line_v_1 = axes[0,1].plot(model.Time, model.get_values('V_1'), label='$\sf V_1$', color=colors[5])
        # #line_theta_1 = axes[0,1].plot(T, Y[:,syst.y_list.index('theta_1')], label='$\sf \\theta_1$')
        # self.line_p_t = axes[1,1].plot(model.Time, model.get_values('p_g_1'), label='$\sf P_t$', color=colors[2])
        # self.line_q_t = axes[1,1].plot(model.Time, model.get_values('q_g_1'), label='$\sf Q_t$', color=colors[0])

        # y_labels = ['$\delta$','$\omega$','$P_t$']

        axes[0].set_ylim((0,1.2))
        axes[1].set_ylim((-1.2,1.2))
        # axes[0].set_ylim((0.8,1.2))
        # axes[1].set_ylim((-0.5,1.5))

        # axes[0,0].grid(True)
        # axes[1,0].grid(True)
        # axes[0,1].grid(True)
        # axes[1,1].grid(True)
        # axes[0,0].legend(loc='best')
        # axes[1,0].legend(loc='best')
        # axes[0,1].legend(loc='best')
        # axes[1,1].legend(loc='best')

        axes[0].set_xlabel('Time (s)')  
        axes[1].set_xlabel('Time (s)') 

        fig.tight_layout()
        fig.savefig('results.svg')
        
        self.fig = fig
        # #axes[0].set_title('Par en función de la velocidad')
        # #axes[1].set_title('Corriente en función de la velocidad')


        self.sld_p_ppc = ipywidgets.FloatSlider(orientation='horizontal',description = "p<sub>ppc</sub>", 
                                        value=model.get_value('p_s_ppc_POI'), min=0.0,max= 1.2, 
                                        step=.1)
        self.sld_q_ppc = ipywidgets.FloatSlider(orientation='horizontal',description = "q<sub>ppc</sub>", 
                                        value=model.get_value('q_s_ppc_POI'), min=-1.2,max= 1.2, 
                                        step=.1)

        self.sld_T_lp1 = ipywidgets.FloatSlider(orientation='horizontal',description = "T_lp1", 
                                        value=model.get_value('T_lp1_POI'), min=0.01,max= 2.0, 
                                        step=.1)
        self.sld_T_lp2 = ipywidgets.FloatSlider(orientation='horizontal',description = "T_lp2", 
                                        value=model.get_value('T_lp2_POI'), min=0.01,max= 2.0, 
                                        step=.1)
        self.sld_PRamp = ipywidgets.FloatSlider(orientation='horizontal',description = "PRamp", 
                                        value=model.get_value('PRamp_POI'), min=0.1,max= 4.0, 
                                        step=.1)
        self.sld_QRamp = ipywidgets.FloatSlider(orientation='horizontal',description = "QRamp", 
                                        value=model.get_value('QRamp_POI'), min=0.1,max= 4.0, 
                                        step=.1)

        # self.sld_v_f = ipywidgets.FloatSlider(orientation='horizontal',description = "v<sub>f</sub>", 
        #                                 value=model.get_value('v_f_1'), min=0.5,max= 4, 
        #                                 step=.1)

        # self.prog_c = ipywidgets.IntProgress(
        #     value=100,
        #     min=0,
        #     max=120,
        #     step=1,
        #     description='SM Load:',
        #     bar_style='', # 'success', 'info', 'warning', 'danger' or ''
        #     orientation='horizontal' 
        # )

        # self.prog_damp = ipywidgets.IntProgress(
        #     value=10,
        #     min=0,
        #     max=20,
        #     step=1,
        #     description='ζ = 1.0',
        #     bar_style='', # 'success', 'info', 'warning', 'danger' or ''
        #     orientation='horizontal' 
        # )

        self.s = st.svg(r"results.svg")

        self.html = ipywidgets.HTML(
            value= self.s.tostring(),
            placeholder='',
            description='',
        )

        tab_refs = ipywidgets.HBox([ipywidgets.HBox([self.sld_p_ppc,self.sld_q_ppc])])

        layout_1 = ipywidgets.VBox([self.sld_T_lp1,self.sld_T_lp2])
        layout_2 = ipywidgets.VBox([self.sld_PRamp,self.sld_QRamp])
        tab_dynamic = ipywidgets.HBox([layout_1,layout_2])

        self.tabs = ipywidgets.Tab()
        self.tabs.children = [tab_refs,tab_dynamic]

        self.tabs.set_title(0, 'References')
        self.tabs.set_title(1, 'VSC Dynamic')
        
    def update(self,change):
        
        model = self.model

        p_ppc = self.sld_p_ppc.value
        q_ppc = self.sld_q_ppc.value

        T_lp1 = self.sld_T_lp1.value
        T_lp2 = self.sld_T_lp2.value

        PRamp = self.sld_PRamp.value
        QRamp = self.sld_QRamp.value

        model.decimation = 1
        model.Dt = 0.01
        model.ini({'p_s_ppc_POI':1.0,'q_s_ppc_POI':0.0,
                   'T_lp1_POI':T_lp1,'T_lp2_POI':T_lp2,
                   'PRamp_POI':PRamp,'QRamp_POI':QRamp},'xy_0.json')
        model.run( 1.0,{})
        model.run(3,{'p_s_ppc_POI':p_ppc,'q_s_ppc_POI':q_ppc})

        model.post();

        self.line_p[0].set_data(model.Time, model.get_values('p_s_POI'))
        self.line_q[0].set_data(model.Time, model.get_values('q_s_POI'))

        # self.line_omega[0].set_data(model.Time, model.get_values('omega_1'))
        # self.line_v_1[0].set_data(model.Time, model.get_values('V_1'))
        # #line_theta_1 = axes[0,1].plot(T, Y[:,syst.y_list.index('theta_1')], label='$\sf \\theta_1$')
        # self.line_p_t[0].set_data(model.Time, model.get_values('p_g_1'))
        # self.line_q_t[0].set_data(model.Time, model.get_values('q_g_1'))

        # i_d, i_q = model.get_mvalue(['i_d_1','i_q_1'])
        # c = (i_d**2+i_q**2)**0.5

        # self.prog_c.bar_style = 'success'
        # if c>0.9:
        #     self.prog_c.bar_style = 'warning'
        # if c>1.0:
        #     self.prog_c.bar_style = 'danger'
        # self.prog_c.value = 100*c

        self.fig.canvas.draw_idle()
        self.fig.savefig('results.svg')

        self.s = st.svg(r"results.svg")

        self.html.value = self.s.tostring()

    def show(self):

        self.sld_p_ppc.observe(self.update, names='value')
        self.sld_q_ppc.observe(self.update, names='value')
        self.sld_T_lp1.observe(self.update, names='value')
        self.sld_T_lp2.observe(self.update, names='value')
        self.sld_PRamp.observe(self.update, names='value')
        self.sld_QRamp.observe(self.update, names='value')

        layout_row1 = ipywidgets.HBox([self.html])

        layout = ipywidgets.VBox([layout_row1,
                                  self.tabs])
        self.layout = layout
        display(self.layout)

    def build(self):

        S_pv_mva = 1.0

        data = {
            "system":{"name":f"spvib_dq_d","S_base":100e6,"K_p_agc":0.0,"K_i_agc":0.0,"K_xif":0.01},
            "buses":[
                {"name": "POI","P_W":0.0,"Q_var":0.0,"U_kV":0.4},
                {"name":"GRID","P_W":0.0,"Q_var":0.0,"U_kV":0.4}
            ],
            "lines":[
                {"bus_j":"POI","bus_k": "GRID","X_pu":0.01,"R_pu":0.0,"Bs_pu":0.0,"S_mva":1}
                ],
            "pvs":[{"bus":"POI","type":"pv_dq_d","S_n":1e6,"U_n":400.0,"F_n":50.0,"X_s":0.1,"R_s":0.01,"monitor":False,
                                    "I_sc":8,"V_oc":42.1,"I_mp":3.56,"V_mp":33.7,"K_vt":-0.160,"K_it":0.065,"N_pv_s":25,"N_pv_p":250}],
            "sources":[{"type":"genape",
                "bus":"GRID","S_n":1000e6,"F_n":50.0,"X_v":0.001,"R_v":0.0,
                "K_delta":0.001,"K_alpha":1e-6}]
            }

        grid = bmapu_builder.bmapu(data)

        grid.uz_jacs = False
        grid.verbose = False
        grid.construct(f'spvib_dq_d')
        b = builder(grid.sys_dict,verbose=True)
        b.dict2system()
        b.functions()
        b.jacobians()
        b.cwrite()
        b.template()
        b.compile()






