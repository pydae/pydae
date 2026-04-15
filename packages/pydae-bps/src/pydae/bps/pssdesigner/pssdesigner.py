"""This is a Sample Python file."""

import numpy as np
import scipy.optimize as sopt
import pydae.ctrl as ctrl
import matplotlib.pyplot as plt


class designer:

    def __init__(self):

        self.T_wo = 5
        self.T_wo1 = 5
        self.T_wo2 = 5

        self.C_phase = 1.0
        self.C_gain = 0.1

        self.gain_ref = 1.0

        self.pss_type = 'pss_1wo_2ll'


    def set_freqs(self, freqs):

        self.freqs = np.array(freqs)

    def set_plant_ss(self,A,B,C,D):
    #     '''
    #     The plant that have to be compensated in state space format:

    #     Δẋ = A*Δx + B*Δu
    #     Δz = C*Δx + D*Δu

    #     '''

    #     self.sys = ct.ss(A,B,C,D)
    #     #self.sys_r = ct.modelsimp.balred(self.sys,order)
    #     self.G_plant = ct.ss2tf(self.sys)
    #     G_plant = self.G_plant

        self.A = A
        self.B = B
        self.C = C
        self.D = D
        self.N_x = A.shape[0]

    def plant_eval(self, omega):

        A = self.A  
        B = self.B  
        C = self.C  
        D = self.D  

        I = np.eye(self.N_x)

        s = 1j*omega
        G = C @ np.linalg.inv(s*I - A) @ B + D # función de transferencia

        return G





    def plant_angles_eval(self):

        angles = []

        for freq in self.freqs:
            omega = 2*np.pi*freq
            G = self.plant_eval(omega)
            phase = np.angle(G,deg=False)

            angles += [phase]

        self.plant_angles = np.array(angles)
        print(self.plant_angles)

    # def reduce(self,A,B,C,D, order=5):

    #     self.sys = ct.ss(A,B,C,D)
    #     #self.sys_r = ct.modelsimp.balred(self.sys,order)
    #     self.G_plant = ct.ss2tf(self.sys)

    def pss_1wo_2ll(self,x, omega):
        '''
        POD with 2 washouts and three lead-lag compensators
        '''

        omega_1,angle_1,omega_2,angle_2, K_stab = x

        T_wo = self.T_wo
        T_1,T_2 = ctrl.lead_design(angle_1,omega_1)
        T_3,T_4 = ctrl.lead_design(angle_2,omega_2)

        s = 1j*omega 

        G_wo = (T_wo*s)/(T_wo*s+1)
        G_ll_12 = (T_1*s + 1)/(T_2*s+1) 
        G_ll_34 = (T_3*s + 1)/(T_4*s+1)

        G_pss = K_stab * G_wo * G_ll_12 * G_ll_34

        self.results = {'T_1':T_1,'T_2':T_2,'T_3':T_3,'T_4':T_4, 'K_stab':K_stab}
        self.results.update({'omega_1':omega_1,'angle_1':angle_1,'omega_2':omega_2,'angle_2':angle_2})
        self.x = x
        return G_pss
 

    def pss_2wo_3ll(self,x):
        '''
        POD with 2 washouts and three lead-lag compensators
        '''

        omega_1,angle_1,omega_2,angle_2,omega_3,angle_3, K_stab = x

        T_1,T_2 = ctrl.lead_design(angle_1,omega_1)
        T_3,T_4 = ctrl.lead_design(angle_2,omega_2)
        T_5,T_6 = ctrl.lead_design(angle_3,omega_3)

        G_l12 = ct.tf([T_1,1],[T_2,1])
        G_l34 = ct.tf([T_3,1],[T_4,1])
        G_l56 = ct.tf([T_5,1],[T_6,1])


        self.G_l12 = G_l12
        self.G_l34 = G_l34
        self.G_l56 = G_l56

        self.G_pod = ct.series(K_stab,self.G_wo1,self.G_wo2,G_l12,G_l34,G_l56)

        self.results = {'T_1':T_1,'T_2':T_2,'T_3':T_3,'T_4':T_4,'T_5':T_5,'T_6':T_6, 'K_stab':K_stab}
        self.results.update({'omega_1':omega_1,'angle_1':angle_1,'omega_2':omega_2,'angle_2':angle_2,'omega_3':omega_3,'angle_3':angle_3})


        return self.G_pod


    def obj(self,x):

        self.x = x

        # G_pod = self.pod_3_eval(x)

        # G_plant = self.G_plant

        # G = ct.series(G_pod,G_plant)

        # self.G = G
        # self.G_pod = G_pod

        self.objective = 0.0
        for freq,angle in zip(self.freqs,self.plant_angles):

            if self.pss_type == 'pss_1wo_2ll':
                G_pss_value = self.pss_1wo_2ll(x,2*np.pi*freq)

            angle_pss = np.angle(G_pss_value,deg=False)
            module_pss = np.abs(G_pss_value)
            self.objective += self.C_phase *(-angle - angle_pss)**2
            self.objective += self.C_gain *(self.gain_ref - module_pss)**2

        return self.objective

    def bode(self, G, freq_min=0.05,freq_max=1.7):

        freqs = np.linspace(freq_min,freq_max,500)
        omegas = 2*np.pi*freqs

        s = 1j*omegas
        G_out = G(s)

        G_m = np.abs(G_out)
        G_ang = np.unwrap(np.angle(G_out))

        return omegas,G_m, G_ang


    def run(self):

        if self.pss_type == 'pss_1wo_2ll':

            x_0 = [0.2*2*np.pi, 1.0, 1.5*2*np.pi, 1.0, 10]
            bounds = [(0.1*2*np.pi,2.5*2*np.pi),(-np.pi/3,np.pi/3)]*2 + [(0.0,100)]

        if self.pss_type == 'pss_2wo_3ll':

            x_0 = [0.1,1, 0.5 , 2, 0.5, 3, 10]
            bounds = [(0.1*2*np.pi,2.5*2*np.pi),(-np.pi/3,np.pi/3)]*3 + [(1,100)]


        res = sopt.minimize(self.obj,x_0,bounds=bounds, method='Nelder-Mead')
        # # 'Nelder-Mead','Powell','SLSQP'
        # print(res)

        res = sopt.differential_evolution(self.obj,bounds=bounds) # direct,dual_annealing,differential_evolution

        print(res)
        self.res = res

    def report(self, fig_file='bode.png'):

        print(self.results)

        fig, axes = plt.subplots(nrows = 2)

        omegas = np.arange(0.1,2.0,0.05)*2*np.pi
        G_plants = omegas*0.0+0j
        G_psss = omegas*0.0+0j 
        G_comps = omegas*0.0+0j

        for it, item in enumerate(omegas):
            G_plants[it] = self.plant_eval(omegas[it]) 
            G_psss[it] = self.pss_1wo_2ll(self.x, omegas[it])

        G_comps = G_plants * G_psss
            
        axes[0].plot(omegas/(2*np.pi), np.angle(G_plants, deg=True), label='Plant')
        axes[1].plot(omegas/(2*np.pi), np.angle(G_plants, deg=True), label='Plant')

        axes[0].plot(omegas/(2*np.pi), np.abs(G_psss), label='PSS')
        axes[1].plot(omegas/(2*np.pi), np.angle(G_psss, deg=True), label='PSS')

        axes[0].plot(omegas/(2*np.pi), np.abs(G_comps), label='Plant+PSS')
        axes[1].plot(omegas/(2*np.pi), np.angle(G_comps, deg=True), label='Plant+PSS')


        for it, item in enumerate(self.freqs):
            G_plants = self.plant_eval(2*np.pi*self.freqs[it]) 
            G_psss = self.pss_1wo_2ll(self.x, 2*np.pi*self.freqs[it])
            G_comps = G_plants * G_psss

            axes[0].plot(self.freqs[it], np.abs(G_comps), 'o')
            axes[1].plot(self.freqs[it], np.angle(G_comps, deg=True), 'o')

        T_1,T_2 = self.results['T_1'],self.results['T_2']
        T_3,T_4 = self.results['T_3'],self.results['T_4']

        s = 1j*omegas
        G_12 = (T_1*s+1)/(T_2*s+1)
        G_34 = (T_3*s+1)/(T_4*s+1)

        axes[1].plot(omegas/(2*np.pi), np.angle(G_12, deg=True), '.')
        axes[1].plot(omegas/(2*np.pi), np.angle(G_34, deg=True), '.')
       
        # omegas,ms,angs = self.bode(self.G_pod)
        # axes[0].plot(omegas/(2*np.pi), ms, label='POD')
        # axes[1].plot(omegas/(2*np.pi), np.rad2deg(angs), label='POD')

        # omegas,ms,angs = self.bode(self.G)
        # axes[0].plot(omegas/(2*np.pi), ms, label='Comp.')
        # axes[1].plot(omegas/(2*np.pi), np.rad2deg(angs), label='Comp.')

        for ax in axes:
            ax.grid()
            ax.legend()

        fig.tight_layout()
        fig.savefig(fig_file)

def pade_plant(delay, order = 3):

    num, den = ct.pade(delay, order)

    sys = ct.tf2ss(num, den)

    A = sys.A
    B = sys.B
    C = sys.C
    D = sys.D

    return A,B,C,D



if __name__ == '__main__':


    A = np.array([-0.2])
    B = np.array([ 1])
    C = np.array([ 1])
    D = np.array([ 0])
    # Designer
    p = designer()
    p.set_freqs([0.35,1.5]) # Hz

    p.set_plant_ss(A,B,C,D)
    p.plant_angles_eval()

    p.run()

    p.report()