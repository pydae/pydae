import numpy as np
import matplotlib.pyplot as plt
import ipywidgets
plt.style.use('https://raw.githubusercontent.com/pydae/pydae/master/src/pydae/edashboards/presentation.mplstyle')
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
import sympy as sym
import pydae.build_cffi as db

class dashboard():
    
    def __init__(self):

        self.colors = colors
        
        # model.Dt = 0.01
        # model.decimation = 1
        # model.ini({'v_ref_1':1.0,'p_m_1':0.0},'xy_0.json')
        # model.run(30.0,{})
        # model.post();
        
        # self.model = model
        
        # self.colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        # self.widgets()

    def build(self):

        u_12  = sym.Symbol(f"u_12", real=True)
        timer  = sym.Symbol(f"timer", real=True)

        x_12  = sym.Symbol(f"x_12", real=True)
        x_t  = sym.Symbol(f"x_t", real=True)

        t_1  = sym.Symbol(f"t_1", real=True)

        T_1 = sym.Symbol(f"T_1", real=True) 
        T_2 = sym.Symbol(f"T_2", real=True)
        omega = sym.Symbol(f"omega", real=True)
        K_sin = sym.Symbol(f"K_sin", real=True)


        u = u_12 + K_sin*sym.sin(omega*t_1)
        z_12 = (u - x_12)*T_1/T_2 + x_12
        dx_12 =  (u - x_12)/T_2      # lead compensator state
        dt_1  = timer - (1-timer)*t_1
        
        params_dict = {'T_1':1.0,'T_2':1.0, 'omega':2*np.pi*10, 'K_sin':0.0}

        sys_dict = {'name':'ctrl_leadlag',
            'params_dict':params_dict,
            'f_list':[dx_12, dt_1],
            'g_list':[],
            'x_list':[ x_12, t_1],
            'y_ini_list':[],
            'y_run_list':[],
            'u_ini_dict':{'u_12':0.0,'timer':0.0},
            'u_run_dict':{'u_12':0.0,'timer':0.0},
            'h_dict':{'z':z_12, 'u':u, 't_1':t_1}
        }

        bldr = db.builder(sys_dict)
        bldr.build()

    def start(self):
        import ctrl_leadlag 
        self.model = ctrl_leadlag.model()
        self.T_1 = 1.0
        self.T_2 = 1.0
        self.omega = 1.0
        self.amplitude = 1.0

    def simulate(self):

        self.model.ini({'timer':0.0,'T_1':self.T_1,'T_2':self.T_2},'xy_0.json')
        self.model.run(1,{})
        self.model.run(10,{'timer':1.0})

        self.model.post()
      
    def widgets(self):
        
        self.simulate()
        model = self.model
        colors = self.colors

        plt.ioff()
        plt.clf()


        fig, axes = plt.subplots(nrows=1, figsize=(9, 4), frameon=False)
        fig.canvas.toolbar_visible = False

        self.line_u = axes.plot(model.Time, model.get_values('u'), label='$u$', color=colors[0])
        self.line_z = axes.plot(model.Time, model.get_values('z'), label='$z$', color=colors[1])
        # self.line_omega = axes[1,0].plot(model.Time, model.get_values('omega_1'), label='$\sf \omega$', color=colors[1])
        # self.line_v_1 = axes[0,1].plot(model.Time, model.get_values('V_1'), label='$\sf V_1$', color=colors[5])
        # #line_theta_1 = axes[0,1].plot(T, Y[:,syst.y_list.index('theta_1')], label='$\sf \\theta_1$')
        # self.line_p_t = axes[1,1].plot(model.Time, model.get_values('p_g_1'), label='$\sf P_t$', color=colors[2])
        # self.line_q_t = axes[1,1].plot(model.Time, model.get_values('q_g_1'), label='$\sf Q_t$', color=colors[0])

        # y_labels = ['$\delta$','$\omega$','$P_t$']

        # axes[0,0].set_ylim((-1,2))
        # axes[1,0].set_ylim((0.95,1.05))
        # axes[0,1].set_ylim((0.8,1.2))
        # axes[1,1].set_ylim((-0.5,1.5))

        # axes[0,0].grid(True)
        # axes[1,0].grid(True)
        # axes[0,1].grid(True)
        # axes[1,1].grid(True)
        # axes[0,0].legend(loc='best')
        # axes[1,0].legend(loc='best')
        # axes[0,1].legend(loc='best')
        # axes[1,1].legend(loc='best')

        # axes[1,0].set_xlabel('Time (s)')  
        # axes[1,1].set_xlabel('Time (s)') 

        fig.tight_layout()
        
        self.fig = fig
        #axes[0].set_title('Par en función de la velocidad')
        #axes[1].set_title('Corriente en función de la velocidad')

        fig.savefig('out.svg')
        self.sld_T_1 = ipywidgets.FloatSlider(orientation='horizontal',description = "T1", 
                                        value=self.T_1, min=0.0,max= 2.0, 
                                        step=.1)

        self.sld_T_2 = ipywidgets.FloatSlider(orientation='horizontal',description = "T2", 
                                        value=self.T_2, min=0.0,max= 2.0, 
                                        step=.1)

        self.sld_omega = ipywidgets.FloatSlider(orientation='horizontal',description = "omega", 
                                        value=self.omega, min=0.0,max= 10.0, 
                                        step=.1)
        self.sld_amplitude = ipywidgets.FloatSlider(orientation='horizontal',description = "Amplitude", 
                                        value=self.amplitude, min=0.0,max= 2.0, 
                                        step=.1)
        
        tab_leadlag = ipywidgets.VBox([self.sld_T_1,self.sld_T_2])
        tab_input = ipywidgets.VBox([self.sld_omega,self.sld_amplitude])

        self.tab = ipywidgets.Tab()
        self.tab.children = [tab_input,tab_leadlag]

        self.tab.set_title(0, 'Input')
        self.tab.set_title(1, 'Lead-Lag')

    def update(self,change):
        
        model = self.model

        self.simulate()

        
        T_1 = self.sld_T_1.value
        T_2 = self.sld_T_2.value

        self.line_u.set_data(model.Time, model.get_values('u'))
        self.line_z.set_data(model.Time, model.get_values('z'))
        self.fig.canvas.draw_idle()


    def show(self):

        self.sld_T_1.observe(self.update, names='value')
        self.sld_T_2.observe(self.update, names='value')
        self.sld_omega.observe(self.update, names='value')
        self.sld_amplitude.observe(self.update, names='value')

        layout_row1 = ipywidgets.HBox([self.fig.canvas])
        layout_row2 = ipywidgets.HBox([self.tab])

        layout = ipywidgets.VBox([layout_row1,layout_row2])
        self.layout = layout
        display(self.layout)
     

if __name__ == "__main__":

    d = dashboard()
    d.build()
    d.start()
    d.widgets()
    
    

