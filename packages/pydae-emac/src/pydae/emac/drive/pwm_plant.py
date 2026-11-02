# -*- coding: utf-8 -*-
"""The machine with its PWM modulator and inverter, as a plant a Python loop
drives one control period at a time.

A real drive splits the work in two, and so does this plant:

* the **processor** (your Python loop) runs every control period T_s: it
  reads the measurements, runs the controller and writes three modulation
  values -- the duties d_a, d_b, d_c in [0, 1];
* the **PWM modulator** (inside the plant) holds those duties for the
  period and compares them with a triangular carrier.  Each period is one
  ramp of the carrier, rising and falling in turn (double update: f_sw =
  1 / (2 T_s)), so each inverter leg switches exactly once, where its duty
  and the carrier cross.  The phase voltages are therefore four constant
  pieces per period, and the machine is integrated through them.

::

    plant = PwmPlant(tables, R, J, V_dc, T_s, x0=...)
    for k in range(n):
        m = plant.measurements()                     # i_abc, theta_m (carrier peak or valley)
        d = my_controller(m)                         # Clarke + Park, control, modulation values
        plant.step(d)                                # modulator + inverter + machine, T_s
    r = plant.results()

The machine is the switched drive's: fluxes as states, currents and torque
from the FEA tables with the spatial ripple as a Fourier series in the rotor
angle (:func:`~pydae.emac.drive.drive_model.build_machine`).  The modulator
is the same code ``build_control_period`` uses
(:func:`~pydae.emac.drive.drive_model.pwm_pieces`), so a Python controller
that does what ``PIController`` does reproduces ``simulate()``.

Options: ``mode="averaged"`` holds the period-mean voltage instead of
switching; ``t_dead`` adds the inverter's dead time (the current at the
period start decides which diode holds each leg during it).

**Speed.**  ``step(d)`` holds the speed (the load torque equals the motor
torque, as in ``build_control_period``); ``step(d, T_load=...)`` lets the
rotor respond to a load torque instead.

One period is one compiled CasADi function evaluated through a
``FunctionBuffer`` on preallocated arrays, so a call costs microseconds.
"""
import casadi as ca
import numpy as np
from pydae.emac.drive.drive_model import build_machine, build_window, pwm_pieces


class PwmPlant:
    """Machine + inverter + PWM carrier, advanced one control period per call.

    Parameters
    ----------
    tables : dict
        From :func:`pydae.emac.luts.build_flux_tables`.
    R, J : float
        Phase resistance [Ohm] and rotor inertia [kg m^2].
    V_dc : float
        DC-link voltage [V].
    T_s : float
        Control period [s]: one carrier ramp.
    mode : {"switched", "averaged"}
    t_dead : float
        Dead time [s].
    x0 : array (4,)
        Initial state (lam_d, lam_q, omega_m, theta_m).
    integrator : {"rk4", "cvodes"}
        One RK4 step per constant-voltage piece, or adaptive CVODES.
    log : bool
        Keep every switching-instant sample for :meth:`results`.
    """

    def __init__(self, tables, R, J, V_dc, T_s, mode="switched", t_dead=0.0, x0=None,
                 integrator="rk4", log=True):
        self.p, self.T_s, self.V_dc, self.log = tables["p"], float(T_s), float(V_dc), log
        ode, self.currents = build_machine(tables, R, J)
        window = build_window(ode, integrator)

        x = ca.MX.sym("x", 4)
        d = ca.MX.sym("d", 3)
        rising = ca.MX.sym("rising")
        T_l = ca.MX.sym("T_l")
        hold = ca.MX.sym("hold")                    # 1: load = motor torque (speed held)
        i_d, i_q, T_e = ca.vertsplit(self.currents(x))
        ang = self.p * x[3] + ca.DM([0, -2 * np.pi / 3, 2 * np.pi / 3])
        i_abc = i_d * ca.cos(ang) - i_q * ca.sin(ang)
        T_load = hold * T_e + (1 - hold) * T_l
        pieces, dts = pwm_pieces(d, rising, i_abc, T_load, self.V_dc, self.T_s, mode, t_dead)
        x_end, X = window(x, ca.horzcat(*pieces))
        Y = self.currents.map(4)(X)                 # i_d, i_q, T_e at each piece's end
        f = ca.Function("pwm_period", [x, d, rising, T_l, hold], [x_end, X, Y, ca.vertcat(*dts)])

        self._in = [np.zeros(4), np.zeros(3), np.zeros(1), np.zeros(1), np.zeros(1)]
        # CasADi writes column-major into flat buffers; the matrices are
        # Fortran-ordered views on them (views, not copies, so they see the writes)
        flat = [np.zeros(4), np.zeros(16), np.zeros(12), np.zeros(4)]
        self._out = [flat[0], flat[1].reshape((4, 4), order="F"),
                     flat[2].reshape((3, 4), order="F"), flat[3]]
        self._buf, self._eval = f.buffer()
        for k, a in enumerate(self._in):
            self._buf.set_arg(k, memoryview(a))
        for k, a in enumerate(flat):
            self._buf.set_res(k, memoryview(a))

        self.t, self.k = 0.0, 0
        self.x = np.zeros(4) if x0 is None else np.asarray(x0, float).copy()
        self._meas = np.array(self.currents(self.x)).ravel()
        self._t_log, self._x_log, self._y_log = [], [], []

    # ---- what the sensors give, at the current instant (a carrier peak or valley)
    def measurements(self):
        """The sensors' readings, as a drive's processor gets them:

        ``i_abc``    the three phase currents [A] (current sensors)
        ``theta_m``  the rotor's mechanical angle [rad] (encoder / resolver)
        ``omega_m``  the mechanical speed [rad/s]

        No dq quantity: the processor makes those itself, with Clarke + Park
        on the measured angle (:func:`pydae.emac.drive.modulation.abc_to_dq`).
        The measurements are ideal (no noise, offset, quantisation or delay).
        """
        return {"i_abc": self.i_abc, "theta_m": float(self.x[3]), "omega_m": float(self.x[2])}

    @property
    def i_abc(self):
        """Phase currents [A]."""
        ang = self.theta_e + np.array([0, -2 * np.pi / 3, 2 * np.pi / 3])
        return self._meas[0] * np.cos(ang) - self._meas[1] * np.sin(ang)

    # ---- the plant's internal quantities: for logging and checks, NOT measurable
    @property
    def i_dq(self):
        """dq currents [A] in the table's frame (internal)."""
        return self._meas[:2].copy()

    @property
    def T_e(self):
        """Electromagnetic torque [N.m] (internal)."""
        return float(self._meas[2])

    @property
    def theta_e(self):
        """Electrical rotor angle [rad]."""
        return self.p * self.x[3]

    @property
    def omega_m(self):
        """Mechanical speed [rad/s]."""
        return float(self.x[2])

    @property
    def omega_e(self):
        """Electrical speed [rad/s]."""
        return self.p * self.x[2]

    # ---- one control period ---------------------------------------------------
    def step(self, duties, T_load=None):
        """Hold ``duties`` (3, in [0, 1]) on this period's carrier ramp and
        advance the machine T_s.  ``T_load`` None holds the speed."""
        self._in[0][:] = self.x
        self._in[1][:] = duties
        self._in[2][0] = 1.0 if self.k % 2 == 0 else 0.0     # rising, falling, rising, ...
        self._in[3][0] = 0.0 if T_load is None else T_load
        self._in[4][0] = 1.0 if T_load is None else 0.0
        self._eval()
        x_end, X, Y, dts = self._out
        if self.log:
            self._t_log.append(self.t + np.cumsum(dts))
            self._x_log.append(X.copy())
            self._y_log.append(Y.copy())
        self.x = x_end.copy()
        self._meas = Y[:, -1].copy()                # = the state at the period's end
        self.k += 1
        self.t = self.k * self.T_s

    def results(self):
        """Every switching-instant sample so far (zero-length pieces dropped):
        t, x (4 x N), y = (i_d, i_q, T_e) (3 x N), theta_e, i_a."""
        t = np.concatenate(self._t_log)
        dt = np.diff(np.r_[0.0, t])
        keep = dt > 1e-12 * self.T_s
        x = np.hstack(self._x_log)[:, keep]
        y = np.hstack(self._y_log)[:, keep]
        th = self.p * x[3]
        return dict(t=t[keep], x=x, y=y, theta_e=th, i_a=y[0] * np.cos(th) - y[1] * np.sin(th))
