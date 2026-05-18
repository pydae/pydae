# Parámetros base de la red
import numpy as np 

class Kmodel:

    def __init__(self):

        self.K_1 = 0.0

    def k_model_eval(self, model, bus_name, X_th, inf_bus_name=None, gen_name=None):

        if not gen_name: gen_name = bus_name

        if inf_bus_name:
            V_inf = model.get_value(f'V_{inf_bus_name}')
        else:
            V_inf = 1.0
        

        # 1. Fetch Machine Parameters
        X_d  = model.get_value(f'X_d_{gen_name}')
        X1d  = model.get_value(f'X1d_{gen_name}')
        X_q  = model.get_value(f'X_q_{gen_name}') # Necesario para K1, K4, K5
        T1d0 = model.get_value(f'T1d0_{gen_name}')

        # 2. Fetch Steady-State Operating Point
        delta_0    = model.get_value(f'delta_{gen_name}')
        Eq_prime_0 = model.get_value(f'e1q_{gen_name}') # Tensión transitoria eje d (E'q)

        # Terminal voltage and angle
        V_0     = model.get_value(f'V_{bus_name}')
        theta_0 = model.get_value(f'theta_{bus_name}')

        # 3. Compute auxiliary steady-state variables
        # Tensiones en el eje d-q
        v_d0 = V_0 * np.sin(delta_0 - theta_0)
        v_q0 = V_0 * np.cos(delta_0 - theta_0)

        # Corrientes en el eje d-q considerando el bus infinito y X_th
        i_d0 = (Eq_prime_0 - V_inf * np.cos(delta_0)) / (X1d + X_th)
        i_q0 = (V_inf * np.sin(delta_0)) / (X_q + X_th)

        # 4. Compute the K constants (Heffron-Phillips)
        # K1: Coeficiente de par sincronizante
        K_1 = (Eq_prime_0 * V_inf * np.cos(delta_0)) / (X1d + X_th) + \
            ((X1d - X_q) * (V_inf**2) * np.cos(2 * delta_0)) / ((X1d + X_th) * (X_q + X_th))

        # K2: Sensibilidad de la potencia activa respecto al flujo
        K_2 = (V_inf * np.sin(delta_0)) / (X1d + X_th)

        # K3: Factor de impedancia
        K_3 = (X1d + X_th) / (X_d + X_th)

        # K4: Efecto desmagnetizador
        K_4 = (V_inf * (X_d - X1d) * np.sin(delta_0)) / (X1d + X_th)

        # K5: Sensibilidad de la tensión terminal al ángulo del rotor
        K_5 = (v_d0 / V_0) * X_q * (V_inf * np.cos(delta_0) / (X_q + X_th)) - \
            (v_q0 / V_0) * X1d * (V_inf * np.sin(delta_0) / (X1d + X_th))

        # K6: Sensibilidad de la tensión terminal al flujo de campo (E'q)
        K_6 = (v_q0 / V_0) * (X_th / (X1d + X_th))

        self.K_1 = K_1
        self.K_2 = K_2
        self.K_3 = K_3
        self.K_4 = K_4             
        self.K_5 = K_5
        self.K_6 = K_6

        self.T1d0 = T1d0
        


    def G_vf_vt_eval(self, omega):

        s = 1j*omega

        K_1 = self.K_1
        K_2 = self.K_2
        K_3 = self.K_3
        K_4 = self.K_4             
        K_5 = self.K_5
        K_6 = self.K_6
        T1d0 = self.T1d0

        return K_3 * K_6 / (1 + s * K_3 * T1d0)

    def G_vref_vt_eval(self, omega, T_avr, K_avr):

        s = 1j*omega

        G_vf_vt = self.G_vf_vt_eval(omega)
        G_avr = K_avr/(T_avr*s + 1)

        G_vref_vt = G_vf_vt*G_avr/(1 + G_vf_vt*G_avr)
           
        return G_vref_vt
    
    def G_vref_pe_eval(self, omega, T_avr, K_avr):

        s = 1j*omega

        G_vref_vt = self.G_vref_vt_eval(omega, T_avr, K_avr)

        K_2 = self.K_2
        K_6 = self.K_6

        G_vt_pe = K_2/K_6
 

        G_vref_pe = G_vref_vt*G_vt_pe
           
        return G_vref_pe

    def report_k_model(self):

        K_1 = self.K_1
        K_2 = self.K_2
        K_3 = self.K_3
        K_4 = self.K_4             
        K_5 = self.K_5
        K_6 = self.K_6
        T1d0 = self.T1d0


        # Imprimir resultados
        Tau = K_3 * T1d0
        print("-" * 40)
        print("CONSTANTES DE HEFFRON-PHILLIPS")
        print("-" * 40)
        print(f"K_1 = {K_1:.4f}")
        print(f"K_2 = {K_2:.4f}")
        print(f"K_3 = {K_3:.4f}")
        print(f"K_4 = {K_4:.4f}")
        print(f"K_5 = {K_5:.4f}")
        print(f"K_6 = {K_6:.4f}")
        print("-" * 40)

    def report_tf(self, omega):

        


