import numpy as np

def eval_dae(model):

    sin = np.sin
    cos = np.cos
    Dt = 0.001
    delta = model.get_value("delta")
    omega = model.get_value("omega")
    e1q = model.get_value("e1q")
    e1d = model.get_value("e1d")
    x = np.array([delta, omega, e1q, e1d])


    i_d = model.get_value("i_d")
    i_q = model.get_value("i_q")
    p_m = model.get_value("p_m")
    q_g = model.get_value("q_g")
    y = np.array([i_d, i_q, p_m, q_g])


    v_f = 1.5
    p_g = 0.5
    V = 1.0
    theta = 0.0

    S_n = 100000000.0
    X_d = 1.8
    X1d = 0.3
    T1d0 = 8.0
    X_q = 1.7
    X1q = 0.55
    T1q0 = 0.4
    R_a = 0.01
    X_l = 0.2
    H = 5.0
    D = 1.0
    Omega_b = 314.1592653589793

    v_d = V*sin(delta - theta)
    v_q = V*cos(delta - theta)
    p_e = i_d*(R_a*i_d + V*sin(delta - theta)) + i_q*(R_a*i_q + V*cos(delta - theta))

    f = np.array([[Omega_b*(omega - 1.0)], [(-D*(omega - 1.0) - i_d*(R_a*i_d + V*sin(delta - theta)) - i_q*(R_a*i_q + V*cos(delta - theta)) + p_m)/(2*H)], [(-e1q - i_d*(-X1d + X_d) + v_f)/T1d0], [(-e1d + i_q*(-X1q + X_q))/T1q0]])

    g = np.array([[R_a*i_q + V*cos(delta - theta) + X1d*i_d - e1q], [R_a*i_d + V*sin(delta - theta) - X1q*i_q - e1d], [V*i_d*sin(delta - theta) + V*i_q*cos(delta - theta) - p_g], [V*i_d*cos(delta - theta) - V*i_q*sin(delta - theta) - q_g]])

    h = np.array([[p_e], [v_f], [p_m]])

    F_x = np.array([[0, Omega_b, 0, 0], [(-V*i_d*cos(delta - theta) + V*i_q*sin(delta - theta))/(2*H), -D/(2*H), 0, 0], [0, 0, -1/T1d0, 0], [0, 0, 0, -1/T1q0]])

    F_y_ini = np.array([[0, 0, 0, 0], [(-2*R_a*i_d - V*sin(delta - theta))/(2*H), (-2*R_a*i_q - V*cos(delta - theta))/(2*H), 1/(2*H), 0], [(X1d - X_d)/T1d0, 0, 0, 0], [0, (-X1q + X_q)/T1q0, 0, 0]])

    F_u_ini = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])

    G_x = np.array([[-V*sin(delta - theta), 0, -1, 0], [V*cos(delta - theta), 0, 0, -1], [V*i_d*cos(delta - theta) - V*i_q*sin(delta - theta), 0, 0, 0], [-V*i_d*sin(delta - theta) - V*i_q*cos(delta - theta), 0, 0, 0]])

    G_y_ini = np.array([[X1d, R_a, 0, 0], [R_a, -X1q, 0, 0], [V*sin(delta - theta), V*cos(delta - theta), 0, 0], [V*cos(delta - theta), -V*sin(delta - theta), 0, -1]])

    G_u_ini = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])

    H_x = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])

    H_y = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])

    H_u = np.array([[0, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0]])

    jac_ini = np.array([[0, Omega_b, 0, 0, 0, 0, 0, 0], [(-V*i_d*cos(delta - theta) + V*i_q*sin(delta - theta))/(2*H), -D/(2*H), 0, 0, (-2*R_a*i_d - V*sin(delta - theta))/(2*H), (-2*R_a*i_q - V*cos(delta - theta))/(2*H), 1/(2*H), 0], [0, 0, -1/T1d0, 0, (X1d - X_d)/T1d0, 0, 0, 0], [0, 0, 0, -1/T1q0, 0, (-X1q + X_q)/T1q0, 0, 0], [-V*sin(delta - theta), 0, -1, 0, X1d, R_a, 0, 0], [V*cos(delta - theta), 0, 0, -1, R_a, -X1q, 0, 0], [V*i_d*cos(delta - theta) - V*i_q*sin(delta - theta), 0, 0, 0, V*sin(delta - theta), V*cos(delta - theta), 0, 0], [-V*i_d*sin(delta - theta) - V*i_q*cos(delta - theta), 0, 0, 0, V*cos(delta - theta), -V*sin(delta - theta), 0, -1]])

    jac_run = np.array([[0, Omega_b, 0, 0, 0, 0, 0, 0], [(-V*i_d*cos(delta - theta) + V*i_q*sin(delta - theta))/(2*H), -D/(2*H), 0, 0, (-2*R_a*i_d - V*sin(delta - theta))/(2*H), (-2*R_a*i_q - V*cos(delta - theta))/(2*H), 0, 0], [0, 0, -1/T1d0, 0, (X1d - X_d)/T1d0, 0, 0, 0], [0, 0, 0, -1/T1q0, 0, (-X1q + X_q)/T1q0, 0, 0], [-V*sin(delta - theta), 0, -1, 0, X1d, R_a, 0, 0], [V*cos(delta - theta), 0, 0, -1, R_a, -X1q, 0, 0], [V*i_d*cos(delta - theta) - V*i_q*sin(delta - theta), 0, 0, 0, V*sin(delta - theta), V*cos(delta - theta), -1, 0], [-V*i_d*sin(delta - theta) - V*i_q*cos(delta - theta), 0, 0, 0, V*cos(delta - theta), -V*sin(delta - theta), 0, -1]])

    jac_trap = np.array([[1, -0.5*Dt*Omega_b, 0, 0, 0, 0, 0, 0], [-0.25*Dt*(-V*i_d*cos(delta - theta) + V*i_q*sin(delta - theta))/H, 0.25*D*Dt/H + 1, 0, 0, -0.25*Dt*(-2*R_a*i_d - V*sin(delta - theta))/H, -0.25*Dt*(-2*R_a*i_q - V*cos(delta - theta))/H, 0, 0], [0, 0, 0.5*Dt/T1d0 + 1, 0, -0.5*Dt*(X1d - X_d)/T1d0, 0, 0, 0], [0, 0, 0, 0.5*Dt/T1q0 + 1, 0, -0.5*Dt*(-X1q + X_q)/T1q0, 0, 0], [-V*sin(delta - theta), 0, -1, 0, X1d, R_a, 0, 0], [V*cos(delta - theta), 0, 0, -1, R_a, -X1q, 0, 0], [V*i_d*cos(delta - theta) - V*i_q*sin(delta - theta), 0, 0, 0, V*sin(delta - theta), V*cos(delta - theta), -1, 0], [-V*i_d*sin(delta - theta) - V*i_q*cos(delta - theta), 0, 0, 0, V*cos(delta - theta), -V*sin(delta - theta), 0, -1]])

    expected_dict = {'f':f,'g':g,'h':h}
    expected_dict.update({'F_x':F_x,'F_y_ini':F_y_ini,'F_u_ini':F_u_ini,'G_x':G_x})
    expected_dict.update({'G_y':G_y_ini,'G_u_ini':G_u_ini,'H_x':H_x,'H_y':H_y,'H_u':H_u})
    expected_dict.update({'jac_ini':jac_ini,'jac_run':jac_run,'jac_trap':jac_trap})

    return expected_dict
