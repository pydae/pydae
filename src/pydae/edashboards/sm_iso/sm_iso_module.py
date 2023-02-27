import numpy as np
import matplotlib.pyplot as plt
import ipywidgets
plt.style.use('https://raw.githubusercontent.com/pydae/pydae/master/src/pydae/edashboards/presentation.mplstyle')
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

class dashboard():
    
    def __init__(self, model):
        
        model.Dt = 0.01
        model.decimation = 1
        model.ini({'v_ref_1':1.0,'P_1':0.0},'xy_0.json')
        model.run(20.0,{})
        model.post();
        
        self.model = model
        
        self.colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        self.widgets()
        
    def widgets(self):
        
        model = self.model
        colors = self.colors

        plt.ioff()
        plt.clf()


        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(9, 4), frameon=False)
        fig.canvas.toolbar_visible = False

        self.line_v_f = axes[0,0].plot(model.Time, model.get_values('v_f_1'), label='$\sf v_f$', color=colors[3])
        self.line_p_m = axes[0,0].plot(model.Time, model.get_values('p_m_1'), label='$\sf p_m$', color=colors[4])

        self.line_omega = axes[1,0].plot(model.Time, model.get_values('omega_1'), label='$\sf \omega$', color=colors[1])
        self.line_v_1 = axes[0,1].plot(model.Time, model.get_values('V_1'), label='$\sf V_1$', color=colors[5])
        #line_theta_1 = axes[0,1].plot(T, Y[:,syst.y_list.index('theta_1')], label='$\sf \\theta_1$')
        self.line_p_t = axes[1,1].plot(model.Time, model.get_values('p_g_1'), label='$\sf P_t$', color=colors[2])
        self.line_q_t = axes[1,1].plot(model.Time, model.get_values('q_g_1'), label='$\sf Q_t$', color=colors[0])

        y_labels = ['$\delta$','$\omega$','$P_t$']

        axes[0,0].set_ylim((0,3))
        axes[1,0].set_ylim((0.95,1.05))
        axes[0,1].set_ylim((0.8,1.2))
        axes[1,1].set_ylim((-0.5,1.5))

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

        fig.tight_layout()
        
        self.fig = fig
        #axes[0].set_title('Par en función de la velocidad')
        #axes[1].set_title('Corriente en función de la velocidad')


        self.sld_P_l = ipywidgets.FloatSlider(orientation='horizontal',description = "P<sub>l</sub>", 
                                        value=model.get_value('P_2'), min=0.0,max= 120, 
                                        step=1)
        self.sld_Q_l = ipywidgets.FloatSlider(orientation='horizontal',description = "Q<sub>l</sub>", 
                                        value=model.get_value('Q_2'), min=-100.0,max= 100, 
                                        step=1)

        self.sld_v_ref_1 = ipywidgets.FloatSlider(orientation='horizontal',description = "V<sup>*</sup>", 
                                        value=model.get_value('v_ref_1'), min=0.85,max= 1.15, 
                                        step=.01)

        self.prog_c = ipywidgets.IntProgress(
            value=100,
            min=0,
            max=120,
            step=1,
            description='SM Load:',
            bar_style='', # 'success', 'info', 'warning', 'danger' or ''
            orientation='horizontal' 
        )

        self.prog_damp = ipywidgets.IntProgress(
            value=10,
            min=0,
            max=20,
            step=1,
            description='ζ = 1.0',
            bar_style='', # 'success', 'info', 'warning', 'danger' or ''
            orientation='horizontal' 
        )
        
    def update(self,change):
        
        model = self.model

        P_l = self.sld_P_l.value
        Q_l = self.sld_Q_l.value
        v_ref_1 = self.sld_v_ref_1.value


        #model = smib.smib_class()
        #model.Dt = 0.01
        #model.decimation = 1
        model.Dt = 0.01
        model.ini({'P_1':0.0,'Q_1':0.0,'v_ref_1':1.0},'xy_0.json')
        model.run( 1.0,{})
        model.run(10,{'P_1':-P_l*1e6,'Q_1':-Q_l*1e6,'v_ref_1':v_ref_1})
        model.Dt = 0.1
        model.run(20,{'P_1':-P_l*1e6,'Q_1':-Q_l*1e6,'v_ref_1':v_ref_1})

        model.post();


        self.line_v_f[0].set_data(model.Time, model.get_values('v_f_1'))
        self.line_p_m[0].set_data(model.Time, model.get_values('p_m_1'))

        self.line_omega[0].set_data(model.Time, model.get_values('omega_1'))
        self.line_v_1[0].set_data(model.Time, model.get_values('V_1'))
        #line_theta_1 = axes[0,1].plot(T, Y[:,syst.y_list.index('theta_1')], label='$\sf \\theta_1$')
        self.line_p_t[0].set_data(model.Time, model.get_values('p_g_1'))
        self.line_q_t[0].set_data(model.Time, model.get_values('q_g_1'))

        i_d, i_q = model.get_mvalue(['i_d_1','i_q_1'])
        c = (i_d**2+i_q**2)*0.5

        self.prog_c.bar_style = 'success'
        if c>0.9:
            self.prog_c.bar_style = 'warning'
        if c>1.0:
            self.prog_c.bar_style = 'danger'
        self.prog_c.value = 100*c

        self.fig.canvas.draw_idle()


    def show(self):

        self.sld_P_l.observe(self.update, names='value')
        self.sld_Q_l.observe(self.update, names='value')

        self.sld_v_ref_1.observe(self.update, names='value')

        layout_row1 = ipywidgets.HBox([self.fig.canvas])
        PQ = ipywidgets.VBox([self.sld_P_l,self.sld_Q_l])
        layout_row2 = ipywidgets.HBox([PQ,self.sld_v_ref_1,self.prog_c])

        layout = ipywidgets.VBox([layout_row1,layout_row2])
        self.layout = layout
        display(self.layout)
     