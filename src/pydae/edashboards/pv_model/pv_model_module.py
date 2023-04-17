import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
import pydae.svg_tools as st 
import pydae.build_cffi as db
from pydae.bmapu.pvs.utils.pv_builder import pv_model

plt.style.use('https://raw.githubusercontent.com/pydae/pydae/master/src/pydae/edashboards/presentation.mplstyle')
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
plt.rcParams['font.family'] =  ['sans-serif']

class dashboard():
    
    def __init__(self):

        
        self.colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        self.data = {"I_sc":3.87,"V_oc":42.1,"I_mpp":3.56,"V_mpp":33.7,"N_s":72,
        "K_vt":-0.160,"K_it":0.065,"R_pv_s": 0.5602, "R_pv_sh": 1862, "K_d": 1.3433}


    def ini(self):

        import pv_test

        self.model = pv_test.model()
        self.model.Dt = 0.01
        self.model.decimation = 1
        self.model.ini(self.data,{'i_pv':3,'v_mpp':40})
        self.plot_V_max = self.data['V_oc']*1.2
        self.plot_P_max = self.data['V_mpp']*self.data['I_mpp']*1.2
        self.plot_I_max = self.data['I_sc']*1.2
        
    def build(self):

        pv = pv_model('pv_test')
        pv.model_params(self.data)
        pv.build()

        self.pv = pv

    def compute_curve(self,V):

        self.I = np.zeros(len(self.V))
    
        irrad = self.sld_irrad.value
        temp_deg = self.sld_temp_deg.value

        for it,v in enumerate(self.V):
            self.model.ini({'v_pv':v,'irrad':irrad,'temp_deg':temp_deg})
            self.I[it] = self.model.get_value('i_pv')

    def compute_point(self,v):
        irrad = self.sld_irrad.value
        temp_deg = self.sld_temp_deg.value
        self.v_pv = v
        self.model.ini({'v_pv':v,'irrad':irrad,'temp_deg':temp_deg})
        self.i_pv  = self.model.get_value('i_pv')


    def widgets(self):
        
        model = self.model
        colors = self.colors

        plt.ioff()
        plt.clf()

        self.sld_irrad = widgets.FloatSlider(description='irrad',min=0, max=1200, step=50, value=1000)
        self.sld_temp_deg = widgets.FloatSlider(description='temp (deg)',min=0, max=75, step=1, value=25)
        self.sld_v_pv = widgets.FloatSlider(description='v<sub>pv</sub>*',min=0.0, max=self.data['V_oc']*1.2, step=1, value=33)
        
        self.V = np.arange(0,50,1.0)
        self.compute_curve(self.V)

        fig,axes = plt.subplots(ncols=2,figsize=(6,3))

        self.curve_vi = axes[0].plot(self.V,self.I, color=colors[0]);
        self.curve_vp = axes[1].plot(self.V,self.V*self.I, color=colors[1]);

        self.compute_point(self.sld_v_pv.value)

        #self.curve_vi_point = axes[0].plot(self.V,self.I,'o', color='k');
        # self.curve_vp_point = axes[1].plot(self.v_pv,self.v_pv*self.i_pv,'o', color='k');

        axes[0].set_ylim(0,self.plot_I_max)
        axes[0].set_xlim(0,self.plot_V_max)  
         
        axes[1].set_ylim(0,self.plot_P_max)
        axes[1].set_xlim(0,self.plot_V_max)    
        axes[1].set_xlim(0,self.plot_V_max)    

        axes[0].set_xlabel(r'$\sf v_{pv}$ (V)')
        axes[1].set_xlabel(r'$\sf v_{pv}$ (V)')
        axes[0].set_ylabel(r'$\sf i_{pv}$ (A)')
        axes[1].set_ylabel(r'$\sf p_{pv}$ (W)')

        fig.savefig('curve_iv_pv.svg')

        self.s_pv_fig = st.svg(r"pv_model_db.svg")
        self.html_pv_fig = widgets.HTML(
            value= self.s_pv_fig.tostring(),
            placeholder='',
            description='',
        )

        self.s_curves = st.svg(r"curve_iv_pv.svg")
        self.html_curves = widgets.HTML(
            value= self.s_curves.tostring(),
            placeholder='',
            description='',
        )

        self.fig = fig
        self.axes = axes

        
    def update(self,change):
        
        self.V = np.arange(0,self.data['V_oc']*1.25,1.0)
        self.compute_curve(self.V)

        self.curve_vi[0].set_data(self.V,self.I)
        self.curve_vp[0].set_data(self.V,self.V*self.I)

        self.compute_point(self.sld_v_pv.value)

        #self.curve_vi_point.set_data(self.V,self.I+0.1)
        # self.curve_vp_point[0].set_data(self.v_pv,self.v_pv*self.i_pv)


        self.fig.savefig('curve_iv_pv.svg')
        self.s_curves = st.svg(r"curve_iv_pv.svg")

        self.html_curves.value = self.s_curves.tostring()

        self.s_pv_fig.set_tspan('v_pv', f'={self.model.get_value("v_pv"):4.1f} V')
        self.s_pv_fig.set_tspan('i_pv', f'={self.model.get_value("i_pv"):4.1f} A')
        self.s_pv_fig.set_tspan('p_pv', f'={self.model.get_value("v_pv")*self.model.get_value("i_pv"):5.1f} W')

        self.html_pv_fig.value = self.s_pv_fig.tostring()

    def show(self):

        # Link the sliders value to the update function
        self.sld_irrad.observe(self.update, names='value')
        self.sld_temp_deg.observe(self.update, names='value')
        self.sld_v_pv.observe(self.update, names='value')

        self.update(0)
        self.layout = widgets.HBox([widgets.VBox([self.html_pv_fig,
                                                  self.sld_irrad,
                                                  self.sld_temp_deg,
                                                  self.sld_v_pv]),
                                    widgets.VBox([self.html_curves])])
                                    
        display(self.layout)