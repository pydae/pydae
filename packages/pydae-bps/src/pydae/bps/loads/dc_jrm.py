# -*- coding: utf-8 -*-
r"""
Data-centre load model for transient stability analysis.

A. Jiménez-Ruiz and F. Milano, "Data Center Model for Transient Stability
Analysis of Power Systems".

The facility is a UPS between the grid bus and an internal bus that feeds

* the IT load (servers), split into CPU and GPU (AI accelerator) power, each
  smoothed by a first-order lag that represents the CPU/GPU scaling driver
  (eqs. 1-9);
* the cooling load, a third-order squirrel-cage induction motor
  (Milano, *Power System Modelling and Scripting*, eqs. 15.60-15.66);
* the miscellaneous load, a ZIP load evaluated at the internal voltage (eq. 10).

The grid sees (eqs. 11-15)

.. math::

    p_{grid} = k_{conn}\,\left((1+\beta)\,p_{DC} + k_{chg}\,P_{chg}\right), \qquad
    q_{grid} = k_{conn}\,q_{DC}

where ``k_conn`` is the connected fraction of the facility (0 = emergency mode,
1 = normal mode, in between = some UPS segments reconnected) and ``k_chg``
switches the UPS charging mode. Both are inputs driven by the discrete
:class:`DcJrmController`, which implements the disconnection / delayed and
staggered reconnection / disturbance-counting logic of section III-F, and the
GPU pulse train, CPU bursts and Ornstein-Uhlenbeck IT noise.

Simplifications with respect to the paper: the motor is third order (no stator
flux dynamics), and when only part of the facility is connected a single
aggregated motor sees a weighted mix of grid and UPS voltage.
"""

import numpy as np


def dc_jrm(grid, name, bus_name, data_dict):
    """Build the data-centre DAE and return its (p_W, q_var) grid injection."""

    bk = grid.backend

    # network
    V = bk.symbols(f"V_{bus_name}")
    theta = bk.symbols(f"theta_{bus_name}")

    # parameters
    S_n = bk.symbols(f"S_n_{name}")              # facility base (VA)
    S_m = bk.symbols(f"S_m_{name}")              # cooling motor base (VA)
    p_cpu_0 = bk.symbols(f"p_cpu_0_{name}")
    p_cpu_100 = bk.symbols(f"p_cpu_100_{name}")
    p_gpu_0 = bk.symbols(f"p_gpu_0_{name}")
    p_gpu_100 = bk.symbols(f"p_gpu_100_{name}")
    T_cpu = bk.symbols(f"T_cpu_{name}")
    T_gpu = bk.symbols(f"T_gpu_{name}")
    p_zip_0 = bk.symbols(f"p_zip_0_{name}")
    q_zip_0 = bk.symbols(f"q_zip_0_{name}")
    a_p, b_p, c_p = [bk.symbols(f"{s}_{name}") for s in ('a_p', 'b_p', 'c_p')]
    a_q, b_q, c_q = [bk.symbols(f"{s}_{name}") for s in ('a_q', 'b_q', 'c_q')]
    beta = bk.symbols(f"beta_{name}")
    K_online = bk.symbols(f"K_online_{name}")
    P_chg = bk.symbols(f"P_chg_{name}")
    e_max = bk.symbols(f"e_max_{name}")
    K_e = bk.symbols(f"K_e_{name}")
    r_s = bk.symbols(f"r_s_{name}")
    x_s = bk.symbols(f"x_s_{name}")
    x_r = bk.symbols(f"x_r_{name}")
    x_mu = bk.symbols(f"x_mu_{name}")
    r_r = bk.symbols(f"r_r_{name}")
    H_m = bk.symbols(f"H_m_{name}")
    Omega_b = bk.symbols(f"Omega_b_{name}")

    # inputs
    u_cpu = bk.symbols(f"u_cpu_{name}")
    u_gpu = bk.symbols(f"u_gpu_{name}")
    p_burst = bk.symbols(f"p_burst_{name}")
    eta_it = bk.symbols(f"eta_it_{name}")
    k_conn = bk.symbols(f"k_conn_{name}")
    k_chg = bk.symbols(f"k_chg_{name}")
    v_ups = bk.symbols(f"v_ups_{name}")
    theta_ups = bk.symbols(f"theta_ups_{name}")
    K_hold = bk.symbols(f"K_hold_{name}")

    # dynamic states
    p_cpu = bk.symbols(f"p_cpu_{name}")
    p_gpu = bk.symbols(f"p_gpu_{name}")
    e = bk.symbols(f"e_{name}")
    e1d = bk.symbols(f"e1d_{name}")
    e1q = bk.symbols(f"e1q_{name}")
    sigma = bk.symbols(f"sigma_{name}")

    # algebraic states
    v_i = bk.symbols(f"v_i_{name}")
    theta_i = bk.symbols(f"theta_i_{name}")
    i_d = bk.symbols(f"i_d_{name}")
    i_q = bk.symbols(f"i_q_{name}")
    p_cool = bk.symbols(f"p_cool_{name}")
    q_cool = bk.symbols(f"q_cool_{name}")
    p_dc = bk.symbols(f"p_dc_{name}")
    q_dc = bk.symbols(f"q_dc_{name}")
    p_grid = bk.symbols(f"p_grid_{name}")
    q_grid = bk.symbols(f"q_grid_{name}")
    tau_m = bk.symbols(f"tau_m_{name}")          # y_ini, u_run
    p_cool_ref = bk.symbols(f"p_cool_ref_{name}")  # u_ini, y_run

    # auxiliar
    x_0 = x_s + x_mu
    x_1 = x_s + x_r*x_mu/(x_r + x_mu)
    T_10 = (x_r + x_mu)/(Omega_b*r_r)
    v_d = -v_i*bk.sin(theta_i)
    v_q = v_i*bk.cos(theta_i)
    tau_e = e1d*i_d + e1q*i_q
    p_cpu_raw = p_cpu_0 + (p_cpu_100 - p_cpu_0)*u_cpu + p_burst   # eq. (1)
    p_gpu_raw = p_gpu_0 + (p_gpu_100 - p_gpu_0)*u_gpu             # eq. (4)
    p_it = p_cpu + p_gpu + eta_it                                 # eq. (8)
    p_zip = p_zip_0*(a_p + b_p*v_i + c_p*v_i**2)                  # eq. (10)
    q_zip = q_zip_0*(a_q + b_q*v_i + c_q*v_i**2)
    K_isl = K_online + (1 - K_online)*(1 - k_conn)  # share of loads fed by the UPS inverter

    # dynamic equations
    dp_cpu = (p_cpu_raw - p_cpu)/T_cpu                            # eq. (6)
    dp_gpu = (p_gpu_raw - p_gpu)/T_gpu                            # eq. (7)
    de = p_grid - (1 + beta)*p_dc + K_e*(e_max - e)               # eq. (15), K_e: regularization
    de1d = Omega_b*sigma*e1q - (e1d + (x_0 - x_1)*i_q)/T_10       # Milano (15.62)
    de1q = -Omega_b*sigma*e1d - (e1q - (x_0 - x_1)*i_d)/T_10
    dsigma = (tau_m - tau_e)/(2*H_m)                              # Milano (15.65)

    # algebraic equations
    eq_v_i = -v_i + V + K_isl*(v_ups - V)
    eq_theta_i = -theta_i + theta + K_hold*(theta_ups - theta)
    eq_i_d = v_d - e1d - r_s*i_d + x_1*i_q                        # Milano (15.63)
    eq_i_q = v_q - e1q - r_s*i_q - x_1*i_d
    eq_p_cool = -p_cool + (v_d*i_d + v_q*i_q)*S_m/S_n
    eq_q_cool = -q_cool + (v_q*i_d - v_d*i_q)*S_m/S_n
    eq_p_dc = -p_dc + p_cool + p_zip + p_it                       # eq. (11)
    eq_q_dc = -q_dc + q_cool + q_zip
    eq_p_grid = -p_grid + k_conn*((1 + beta)*p_dc + k_chg*P_chg)  # eqs. (12)-(14)
    eq_q_grid = -q_grid + k_conn*q_dc
    eq_p_cool_ref = -p_cool_ref + p_cool

    # dae
    grid.dae['f'] += [dp_cpu, dp_gpu, de, de1d, de1q, dsigma]
    grid.dae['x'] += [p_cpu, p_gpu, e, e1d, e1q, sigma]
    grid.dae['g'] += [eq_v_i, eq_theta_i, eq_i_d, eq_i_q, eq_p_cool, eq_q_cool,
                      eq_p_dc, eq_q_dc, eq_p_grid, eq_q_grid, eq_p_cool_ref]
    grid.dae['y_ini'] += [v_i, theta_i, i_d, i_q, p_cool, q_cool,
                          p_dc, q_dc, p_grid, q_grid, tau_m]
    grid.dae['y_run'] += [v_i, theta_i, i_d, i_q, p_cool, q_cool,
                          p_dc, q_dc, p_grid, q_grid, p_cool_ref]

    # data (powers in MW, converted to pu on S_n)
    S_n_N = data_dict['S_mva']*1e6
    S_m_N = data_dict.get('S_m_mva', data_dict['p_cool_mw']/0.8)*1e6

    def pu(mw):
        return mw*1e6/S_n_N

    p_cpu_0_N = pu(data_dict.get('p_cpu_0_mw', 0.0))
    p_cpu_100_N = pu(data_dict.get('p_cpu_100_mw', 0.0))
    p_gpu_0_N = pu(data_dict.get('p_gpu_0_mw', 0.0))
    p_gpu_100_N = pu(data_dict.get('p_gpu_100_mw', 0.0))
    u_cpu_N = data_dict.get('u_cpu', 1.0)
    u_gpu_N = data_dict.get('u_gpu', 1.0)
    p_cool_N = pu(data_dict['p_cool_mw'])
    e_max_N = data_dict.get('e_max_s', 300.0)   # seconds of full facility load

    grid.dae['params_dict'].update({
        str(S_n): S_n_N, str(S_m): S_m_N,
        str(p_cpu_0): p_cpu_0_N, str(p_cpu_100): p_cpu_100_N,
        str(p_gpu_0): p_gpu_0_N, str(p_gpu_100): p_gpu_100_N,
        str(T_cpu): data_dict.get('T_cpu', 0.05), str(T_gpu): data_dict.get('T_gpu', 0.05),
        str(p_zip_0): pu(data_dict.get('p_zip_mw', 0.0)),
        str(q_zip_0): pu(data_dict.get('q_zip_mvar', 0.0)),
        str(a_p): data_dict.get('a_p', 0.0), str(b_p): data_dict.get('b_p', 0.0),
        str(c_p): data_dict.get('c_p', 1.0),
        str(a_q): data_dict.get('a_q', 0.0), str(b_q): data_dict.get('b_q', 0.0),
        str(c_q): data_dict.get('c_q', 1.0),
        str(beta): data_dict.get('beta', 0.0),
        str(K_online): 1.0 if data_dict.get('topology', 'offline') == 'online' else 0.0,
        str(P_chg): pu(data_dict.get('p_chg_mw', 0.0)),
        str(e_max): e_max_N, str(K_e): data_dict.get('K_e', 1e-3),
        str(r_s): data_dict.get('r_s', 0.031), str(x_s): data_dict.get('x_s', 0.1),
        str(x_r): data_dict.get('x_r', 0.18), str(x_mu): data_dict.get('x_mu', 3.2),
        str(r_r): data_dict.get('r_r', 0.018), str(H_m): data_dict.get('H_m', 0.5),
        str(Omega_b): 2*np.pi*data_dict.get('F_n', 50.0),
        })

    u_common = {str(u_cpu): u_cpu_N, str(u_gpu): u_gpu_N, str(p_burst): 0.0,
                str(eta_it): 0.0, str(k_conn): 1.0, str(k_chg): 0.0,
                str(v_ups): 1.0, str(theta_ups): 0.0, str(K_hold): 0.0}
    grid.dae['u_ini_dict'].update(u_common)
    grid.dae['u_ini_dict'].update({str(p_cool_ref): p_cool_N})
    grid.dae['u_run_dict'].update(u_common)
    grid.dae['u_run_dict'].update({str(tau_m): p_cool_N*S_n_N/S_m_N})

    p_m_0 = p_cool_N*S_n_N/S_m_N
    grid.dae['xy_0_dict'].update({
        str(p_cpu): p_cpu_0_N + (p_cpu_100_N - p_cpu_0_N)*u_cpu_N,
        str(p_gpu): p_gpu_0_N + (p_gpu_100_N - p_gpu_0_N)*u_gpu_N,
        str(e): e_max_N, str(e1d): 0.1*p_m_0, str(e1q): 0.9, str(sigma): 0.01,
        str(v_i): 1.0, str(theta_i): 0.0, str(i_d): 0.5*p_m_0, str(i_q): p_m_0,
        str(p_cool): p_cool_N, str(q_cool): 0.5*p_cool_N,
        str(p_dc): 1.0, str(q_dc): 0.1, str(p_grid): 1.0, str(q_grid): 0.1,
        str(tau_m): p_m_0,
        })

    # outputs
    grid.dae['h_dict'].update({
        f"p_it_{name}": p_it,
        f"soc_{name}": e/e_max,
        f"slip_{name}": sigma,
        f"p_grid_mw_{name}": p_grid*S_n/1e6,
        })

    # grid injection (negative: consumption)
    p_W = -p_grid*S_n
    q_var = -q_grid*S_n

    return p_W, q_var


class DcJrmController:
    r"""Discrete part of the data-centre model, run once per solver step.

    UPS logic (section III-F):

    * **Disconnection** with no delay when :math:`|V-1| > \Delta v_{max}` or
      :math:`|f-F_n| > \Delta f_{max}` (eq. 16).
    * **Delayed, staggered reconnection**: the facility is split into
      ``N_seg`` UPS segments; segment ``k`` reconnects ``T_rec + k*dT_rec``
      seconds after the grid is back in range (``N_seg=1`` is the single-UPS
      case that can produce flapping, Fig. 10; ``N_seg=10``, ``T_rec=5``,
      ``dT_rec=1`` is Fig. 11).
    * **Disturbance counting**: after ``N_count`` disturbances within
      ``T_count`` seconds, reconnection is locked out.
    * **Charging mode** while the UPS energy is below ``e_max``.

    Workload (section II): GPU pulse train (eq. 5), periodic CPU bursts
    (eq. 2) and Ornstein-Uhlenbeck IT noise (eq. 9).

    ``step(model)`` returns the change in ``k_conn`` so it can also be passed as
    the ``on_step`` callback of :class:`pydae.bps.utils.faults.FaultManager`.
    """

    def __init__(self, name, bus, cfg=None, seed=0, level="low"):
        cfg = cfg or {}
        self.name, self.bus, self.level = name, bus, level
        self.F_n = cfg.get('F_n', 50.0)
        self.Dv_max = cfg.get('Dv_max', 0.1)
        self.Df_max = cfg.get('Df_max', 0.3)
        self.N_seg = cfg.get('N_seg', 1)
        self.T_rec = cfg.get('T_rec', 30.0)
        self.dT_rec = cfg.get('dT_rec', 0.0)
        self.N_count = cfg.get('N_count', 3)
        self.T_count = cfg.get('T_count', 60.0)
        self.angle_comp = cfg.get('angle_comp', True)
        self.v_hold = cfg.get('v_hold', 'pre')       # 'pre' (pre-fault) or a value in pu
        self.gpu = cfg.get('gpu')                    # {u_min, u_max, T, tau, t_0}
        self.burst = cfg.get('burst')                # {p, T, tau, t_0} (p in pu of S_n)
        self.ou = cfg.get('ou')                      # {a, b}
        self.rng = np.random.default_rng(seed)

        self.connected = [True]*self.N_seg
        self.t_ok = None                             # time the grid came back in range
        self.out_prev = False
        self.onsets = []
        self.locked = False
        self.V_pre, self.theta_pre = 1.0, 0.0
        self.eta, self.t_last = 0.0, None

    def _t(self, model):
        # ctypes/cffi Model keeps the current time in t_start (end of last run)
        t = getattr(model, 't', None)
        return getattr(model, 't_start', 0.0) if t is None else t

    def step(self, model):
        n, t = self.name, self._t(model)
        V = model.get_value(f'V_{self.bus}')
        f = self.F_n*model.get_value('omega_coi')
        out = abs(V - 1.0) > self.Dv_max or abs(f - self.F_n) > self.Df_max

        k_old = sum(self.connected)/self.N_seg

        if out:
            if not self.out_prev:
                self.onsets = [t_o for t_o in self.onsets if t - t_o < self.T_count] + [t]
                if len(self.onsets) >= self.N_count:
                    self.locked = True
            self.connected = [False]*self.N_seg
            self.t_ok = None
        else:
            if all(self.connected):
                self.V_pre = V
                self.theta_pre = model.get_value(f'theta_{self.bus}')
            elif not self.locked:
                if self.t_ok is None:
                    self.t_ok = t
                for k in range(self.N_seg):
                    if t - self.t_ok >= self.T_rec + k*self.dT_rec:
                        self.connected[k] = True
        self.out_prev = out

        k_conn = sum(self.connected)/self.N_seg
        islanded = k_conn < 1.0
        soc = model.get_value(f'soc_{n}')
        v_ups = self.V_pre if self.v_hold == 'pre' else float(self.v_hold)

        model.set_value(f'k_conn_{n}', k_conn)
        model.set_value(f'k_chg_{n}', 1.0 if soc < 0.999 else 0.0)
        if islanded:
            model.set_value(f'v_ups_{n}', v_ups)
            model.set_value(f'theta_ups_{n}', self.theta_pre)
            model.set_value(f'K_hold_{n}', 0.0 if self.angle_comp else 1.0)
        else:
            model.set_value(f'K_hold_{n}', 0.0)

        self._workload(model, t)
        return abs(k_conn - k_old)

    def _workload(self, model, t):
        n = self.name
        if self.gpu:
            g = self.gpu
            phase = (t - g.get('t_0', 0.0)) % g['T']
            on = t >= g.get('t_0', 0.0) and phase < g['tau']
            model.set_value(f'u_gpu_{n}', g['u_max'] if on else g['u_min'])
        if self.burst:
            b = self.burst
            phase = (t - b.get('t_0', 0.0)) % b['T']
            on = t >= b.get('t_0', 0.0) and phase < b['tau']
            model.set_value(f'p_burst_{n}', b['p'] if on else 0.0)
        if self.ou:
            dt = 0.0 if self.t_last is None else t - self.t_last
            if dt > 0.0:
                a, b = self.ou['a'], self.ou['b']
                self.eta += -a*self.eta*dt + b*np.sqrt(dt)*self.rng.standard_normal()
                model.set_value(f'eta_it_{n}', self.eta)
        self.t_last = t


def test():
    import pytest
    from pydae.bps import BpsBuilder
    from pydae.core import Builder, Model

    grid = BpsBuilder('dc_jrm.hjson')
    grid.uz_jacs = False
    grid.construct('temp_dc_jrm')
    Builder(grid.sys_dict, target='cffi', sparse=False).build()

    model = Model('temp_dc_jrm')
    model.ini({}, 'temp_dc_jrm_xy_0.json')
    model.report_x()
    model.report_y()

    assert model.get_value('p_cool_dc') == pytest.approx(60/400, rel=1e-6)
    assert model.get_value('soc_dc') == pytest.approx(1.0, rel=1e-6)
    assert 0.0 < model.get_value('sigma_dc') < 0.1


if __name__ == '__main__':
    test()
