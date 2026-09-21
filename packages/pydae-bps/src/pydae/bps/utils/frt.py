# -*- coding: utf-8 -*-
r"""
Fault ride-through (FRT) control for grid-following converters.

The discrete grid-code ride-through relay is the *particular* behaviour of a
grid-following converter (as opposed to the generic fault continuation in
:mod:`pydae.bps.utils.faults`). It is **shared** by every converter that exposes
the LVRT interface -- ``lvrt_ext_<name>`` / ``i_sr_ref_<name>`` /
``i_sa_ref_<name>`` -- which today means both :mod:`pydae.bps.pvs.pv_pq_ss`
inverters and :mod:`pydae.bps.vscs.bess_pq_ss` VSCs. Rather than duplicate the
relay in each model, it lives here once and both models re-export it, so
``pv_pq_ss`` and ``bess_pq_ss`` present the same ``vrt`` / :class:`VrtController`
/ :func:`iq_po123` API.

The same :func:`iq_po123` curve is used two ways by two owners: relaxed to a
fixed point by :class:`~pydae.bps.utils.faults.FaultManager` during the fault
switch, and rate-limited by :func:`vrt` during the time-domain integration.
"""


def iq_po123(V, v_dead=0.85, v_full=0.5, iq_max=1.0):
    """Reactive-current-vs-voltage characteristic (Spanish REE PO 12.3 style).

    Piecewise-linear grid-code FRT curve::

        V >= v_dead            : i_q = 0            (dead band near nominal)
        v_full < V < v_dead    : i_q = iq_max * (v_dead - V) / (v_dead - v_full)
        V <= v_full            : i_q = iq_max       (saturated, deep dip)

    Defaults give the commonly used curve: dead band above 0.85 pu, full
    reactive current at/below 0.5 pu, slope ~2.86 pu/pu in between. Adjust
    v_dead / v_full / iq_max to the exact PO 12.3 figure for your plant.
    """
    if V >= v_dead:
        return 0.0
    if V <= v_full:
        return iq_max
    return iq_max * (v_dead - V) / (v_dead - v_full)


def _ramp(cur, tgt, rate):
    return min(tgt, cur + rate) if tgt > cur else max(tgt, cur - rate)


def vrt(model, target, level, iq_level, v_enter=0.85, v_exit=0.9,
        rate=0.2, iq_rate=0.2, iq_curve=iq_po123, verbose=False):
    """Discrete grid-code fault-ride-through relay driving the PO 12.3 reactive
    current curve, with hysteresis and rate-limited commutation.

    Call once per time step. Each converter *enters* LVRT when its own terminal
    voltage drops to ``v_enter`` (the curve dead-band edge, 0.85 pu) and
    *returns* to PQ once it recovers to ``v_exit`` (0.9 pu); in between it holds
    its target mode (latched, no chatter). While in LVRT the reactive current
    reference tracks the voltage-dependent curve ``iq_curve(V)`` -- more support
    for deeper dips, self-limiting near the dead band so it cannot overvolt.

    Both the mode (``lvrt_ext``) and the reactive reference (``i_sr_ref``) are
    rate-limited toward their targets so the commutation and the current change
    span several steps -- an abrupt step would jump the terminal voltage across
    the relay threshold in one shot, which a fixed-step integrator cannot cross.

    ``target`` (latched mode), ``level`` (ramped lvrt_ext) and ``iq_level``
    (ramped i_sr_ref) are dicts keyed by converter bus, updated in place; seed
    them from the state FaultManager leaves after apply()/clear() so the relay
    and the fault continuation stay consistent. Returns the number of converters
    in LVRT.
    """
    n_lvrt = 0
    for bus in target:
        V = model.get_value(f'V_{bus}')
        if V <= v_enter:               # dipped -> target LVRT
            target[bus] = 1.0
        elif V >= v_exit:              # recovered -> target PQ
            target[bus] = 0.0
        iq_tgt = iq_curve(V) if target[bus] else 0.0
        level[bus] = _ramp(level[bus], target[bus], rate)
        iq_level[bus] = _ramp(iq_level[bus], iq_tgt, iq_rate)
        model.set_value(f'lvrt_ext_{bus}', level[bus])
        model.set_value(f'i_sr_ref_{bus}', iq_level[bus])
        model.set_value(f'i_sa_ref_{bus}', 0.0)
        n_lvrt += level[bus] > 0.5
    if verbose:
        print(f"  vrt: {n_lvrt}/{len(target)} converters in LVRT")
    return n_lvrt


class VrtController:
    """Stateful per-timestep FRT relay for the discrete orchestration layer.

    Wraps :func:`vrt` with the latched/ramped state it needs (``target`` mode,
    rate-limited ``lvrt_level`` = ``lvrt_ext``, ``iq_level`` = ``i_sr_ref``) so a
    :class:`~pydae.bps.discrete.DiscreteOrchestrator` can drive it generically:
    it just calls :meth:`step` once per solver step, without knowing anything
    about grid codes. This is the *particular* converter behaviour; the
    orchestrator is the *generic* layer that invokes it.

    ``devices`` is the list of converter bus/instance names (every unit exposing
    the ``lvrt_ext_<name>`` interface -- pv_pq_ss inverters and bess_pq_ss VSCs
    share it, so one controller handles them all). The relay tunables mirror
    :func:`vrt`.

    ``level`` is the latency tier the FRT relay runs at (see
    :class:`pydae.bps.discrete.DiscreteOrchestrator`); it is ``"low"`` -- a local
    fast control that must react every solver step. ``period`` (seconds), if
    given, overrides the tier for an explicit cadence.
    """
    def __init__(self, devices, v_enter=0.85, v_exit=0.9, rate=0.2,
                 iq_rate=0.2, iq_curve=iq_po123, verbose=False,
                 level="low", period=None):
        self.devices = list(devices)
        self.v_enter = v_enter
        self.v_exit = v_exit
        self.rate = rate
        self.iq_rate = iq_rate
        self.iq_curve = iq_curve
        self.verbose = verbose
        self.level = level        # latency tier (read by the orchestrator)
        self.period = period      # explicit cadence override (seconds), or None
        # Per-bus discrete relay state (distinct from the `level` tier above):
        self.target = {bus: 0.0 for bus in self.devices}     # latched mode
        self.lvrt_level = {bus: 0.0 for bus in self.devices}  # ramped lvrt_ext
        self.iq_level = {bus: 0.0 for bus in self.devices}    # ramped i_sr_ref

    def sync(self, model):
        """Adopt the converter state currently set on the model.

        Call after a :class:`~pydae.bps.utils.faults.FaultManager` apply()/clear()
        (or any external write to ``lvrt_ext``/``i_sr_ref``) so the relay does
        not fight the continuation's operating point on the next step.
        """
        for bus in self.devices:
            self.lvrt_level[bus] = model.get_value(f'lvrt_ext_{bus}')
            self.target[bus] = round(self.lvrt_level[bus])
            self.iq_level[bus] = model.get_value(f'i_sr_ref_{bus}')

    def step(self, model):
        """Run the relay once; returns the number of converters in LVRT."""
        return vrt(model, self.target, self.lvrt_level, self.iq_level,
                   v_enter=self.v_enter, v_exit=self.v_exit, rate=self.rate,
                   iq_rate=self.iq_rate, iq_curve=self.iq_curve,
                   verbose=self.verbose)




# ---------------------------------------------------------------------- #
# Orden TED/749/2020 (BOE-A-2020-8965) fast current injection
# ---------------------------------------------------------------------- #
def iq_ted749(V, V_pre=1.0, iq_pre=0.0, K_1=2.0, i_lim_up=0.9, i_lim_lo=-0.9):
    r"""Reactive-current law of Orden TED/749/2020, Anexo, apartado 2.3.1.

    Implements figure 8 a)+b) for the positive sequence: a **continuous
    proportional** control on the voltage *error*, incremental to the
    pre-disturbance reactive current, with the total saturated::

        dU1 = V_pre - V                        (apartado 2.3.1 a)
        dI1 = K_1 * dU1                        (figura 8 a,  2 <= K_1 <= 6)
        I1  = clip(iq_pre + dI1, i_lim_lo, i_lim_up)   (figura 8 b)

    ``i_lim_up`` is parameterisable in [0, 1] pu and ``i_lim_lo`` in [-1, 0] pu;
    the order's defaults are +-0.9 pu. Note this is a *line through the origin*,
    not a dead-band curve: it is symmetric (it absorbs on a swell exactly as it
    injects on a dip) and at ``V == V_pre`` it returns the pre-disturbance
    reactive current, not zero.

    That difference from :func:`iq_po123` is the whole point. An absolute
    dead-band curve returns exactly zero over a finite band of voltages, and a
    grid-following converter commanded to inject exactly zero current has an
    unobservable terminal angle -- a structurally singular algebraic Jacobian.
    The TED/749 law never does that, so it is well posed at every voltage.
    """
    return min(max(iq_pre + K_1 * (V_pre - V), i_lim_lo), i_lim_up)


class Ted749Controller:
    r"""Fast current injection per Orden TED/749/2020 (BOE-A-2020-8965).

    The discrete relay for a *modulo de parque electrico* (MPE) of type B, C or
    D. Same role and same interface as :class:`VrtController` -- one per-step
    controller driving every converter that exposes ``lvrt_ext_<name>`` /
    ``i_sr_ref_<name>`` / ``i_sa_ref_<name>`` -- but the Spanish order's law
    rather than the older PO 12.3 curve. Select it with
    ``make_discrete_controller(model, law="ted749")`` (i.e.
    ``DiscreteOrchestrator.from_model(model, law="ted749")``).

    **Activation** (apartado 2.3.1 c): the control arms when the terminal
    voltage leaves ``[V_min, V_max]`` (the order's defaults for connection below
    110 kV are 0.85 / 1.15 pu) *or* on an abrupt change greater than
    ``dv_abrupt`` (0.1 pu, parameterisable 0.05-0.15).

    **Deactivation** (apartado 2.3.1 d): it disarms only once the voltage is
    back in range *and* ``t_hold`` seconds (5 s) have elapsed since the onset --
    a **time** latch, not a voltage hysteresis band. ``t_max`` (30 s) is the
    thermal ceiling of apartado 2.3.1 d. Re-arming while active restarts the
    5 s count (apartado 2.3.1 e).

    **Reactive current**: :func:`iq_ted749`, incremental to the reactive current
    the converter carried just before the disturbance.

    **Active current** (apartado 3.3.1): the active current is *not* zeroed.
    It is whatever the apparent-current capability leaves free,
    ``sqrt(i_max^2 - I1^2)``, capped at the pre-disturbance active current --
    the order requires active power "en funcion de la tension existente en cada
    momento", recovering as soon as the voltage reaches 0.85 pu. With the
    reactive current back at its pre-fault value the cap binds immediately, so
    active power returns on voltage recovery without extra logic.

    Both references are rate-limited at ``1/t_r`` pu/s (apartado 2.3.1: 90 % of
    the required response within ti+tr <= 50 ms), using the wall of *simulation*
    time between calls, so the relay behaves the same at any ``Dt`` or
    real-time chunk size.

    ``level`` is the latency tier (``"low"`` -- a local fast control, every step).
    """
    def __init__(self, devices, K_1=2.0, i_lim_up=0.9, i_lim_lo=-0.9,
                 V_min=0.85, V_max=1.15, dv_abrupt=0.1, t_hold=5.0,
                 t_max=30.0, i_max=1.0, t_r=0.03, verbose=False,
                 level="low", period=None):
        self.devices = list(devices)
        self.K_1 = K_1
        self.i_lim_up = i_lim_up
        self.i_lim_lo = i_lim_lo
        self.V_min = V_min
        self.V_max = V_max
        self.dv_abrupt = dv_abrupt
        self.t_hold = t_hold
        self.t_max = t_max
        self.i_max = i_max
        self.t_r = t_r
        self.verbose = verbose
        self.level = level
        self.period = period
        # Per-converter relay state.
        self.active = {b: False for b in self.devices}   # control armed
        self.t_onset = {b: None for b in self.devices}   # disturbance onset
        self.v_pre = {b: 1.0 for b in self.devices}      # U1i
        self.iq_pre = {b: 0.0 for b in self.devices}     # I1i
        self.ia_pre = {b: 0.0 for b in self.devices}     # active current, pre
        self.lvrt_level = {b: 0.0 for b in self.devices}  # ramped lvrt_ext
        self.iq_level = {b: 0.0 for b in self.devices}    # ramped i_sr_ref
        self.ia_level = {b: 0.0 for b in self.devices}    # ramped i_sa_ref
        self._prev = {}      # bus -> (V, iq, ia) one call back (undisturbed)
        self._t_prev = None

    # ------------------------------------------------------------------ #
    def _measure(self, model, bus):
        """Terminal voltage and the active/reactive current it carries.

        ``p_s = i_sa * V`` and ``q_s = i_sr * V`` identically in the converter's
        voltage-oriented frame, so the pre-disturbance references are read off
        the injected powers without needing extra outputs.
        """
        V = model.get_value(f'V_{bus}')
        if V <= 0.0:
            return V, 0.0, 0.0
        return (V, model.get_value(f'q_s_{bus}') / V,
                model.get_value(f'p_s_{bus}') / V)

    def sync(self, model):
        """Adopt the converter references currently set on the model.

        Call after a :class:`~pydae.bps.utils.faults.FaultManager` apply()/clear()
        so the relay resumes from the continuation's operating point instead of
        stepping away from it.
        """
        for bus in self.devices:
            self.lvrt_level[bus] = model.get_value(f'lvrt_ext_{bus}')
            self.iq_level[bus] = model.get_value(f'i_sr_ref_{bus}')
            self.ia_level[bus] = model.get_value(f'i_sa_ref_{bus}')
            if self.lvrt_level[bus] > 0.5 and not self.active[bus]:
                # The continuation armed the control; adopt it, with the onset
                # clock started now so the 5 s latch is measured from here.
                # Latch the same pre-disturbance operating point step() would
                # have used -- the last undisturbed sample. Without this the
                # relay would resume against the (1, 0, 0) defaults and command
                # zero active current, stepping straight off the operating point
                # the continuation just reached.
                base = self._prev.get(bus)
                if base is not None:
                    self.v_pre[bus], self.iq_pre[bus], self.ia_pre[bus] = base
                self.active[bus] = True
                self.t_onset[bus] = getattr(model, 't', None)

    def step(self, model):
        """Run the relay once; returns the number of converters injecting."""
        t = getattr(model, 't', None)
        dt = None if (t is None or self._t_prev is None) else t - self._t_prev
        step_max = None if not dt or dt <= 0.0 else dt / self.t_r

        n_active = 0
        for bus in self.devices:
            V, iq, ia = self._measure(model, bus)
            prev = self._prev.get(bus)
            in_range = self.V_min <= V <= self.V_max
            abrupt = prev is not None and abs(V - prev[0]) > self.dv_abrupt

            if not self.active[bus]:
                if not in_range or abrupt:
                    # Arm, latching the operating point from *before* the jump.
                    base = prev if prev is not None else (V, iq, ia)
                    self.v_pre[bus], self.iq_pre[bus], self.ia_pre[bus] = base
                    self.active[bus] = True
                    self.t_onset[bus] = t
            else:
                if abrupt:                       # apartado 2.3.1 e: restart
                    self.t_onset[bus] = t
                elapsed = None if (t is None or self.t_onset[bus] is None) \
                    else t - self.t_onset[bus]
                if elapsed is not None and (
                        (in_range and elapsed >= self.t_hold)
                        or elapsed >= self.t_max):
                    self.active[bus] = False

            if self.active[bus]:
                mode_tgt = 1.0
                iq_tgt = iq_ted749(V, self.v_pre[bus], self.iq_pre[bus],
                                   K_1=self.K_1, i_lim_up=self.i_lim_up,
                                   i_lim_lo=self.i_lim_lo)
                headroom = max(self.i_max ** 2 - iq_tgt ** 2, 0.0) ** 0.5
                ia_tgt = min(self.ia_pre[bus], headroom)
                n_active += 1
            else:
                mode_tgt = iq_tgt = ia_tgt = 0.0
                # Undisturbed: keep tracking the operating point to latch on.
                self._prev[bus] = (V, iq, ia)

            if step_max is None:
                self.lvrt_level[bus], self.iq_level[bus], self.ia_level[bus] = (
                    mode_tgt, iq_tgt, ia_tgt)
            else:
                self.lvrt_level[bus] = _ramp(self.lvrt_level[bus], mode_tgt, step_max)
                self.iq_level[bus] = _ramp(self.iq_level[bus], iq_tgt, step_max)
                self.ia_level[bus] = _ramp(self.ia_level[bus], ia_tgt, step_max)

            model.set_value(f'lvrt_ext_{bus}', self.lvrt_level[bus])
            model.set_value(f'i_sr_ref_{bus}', self.iq_level[bus])
            model.set_value(f'i_sa_ref_{bus}', self.ia_level[bus])

        self._t_prev = t
        if self.verbose:
            print(f"  ted749: {n_active}/{len(self.devices)} converters injecting")
        return n_active


def make_discrete_controller(model, law="po123", **kwargs):
    """Discrete-controller factory discovered by :class:`DiscreteOrchestrator`.

    Inspects ``model`` for the grid-following LVRT interface and returns one
    controller covering every converter exposing it (pv_pq_ss and bess_pq_ss
    alike), or ``None`` when the system has none (so the orchestrator skips it).

    ``law`` picks which grid code the relay follows:

    * ``"po123"`` (default) -- :class:`VrtController`, the PO 12.3 style
      absolute ``iq(V)`` curve with a latched voltage hysteresis band.
    * ``"ted749"`` -- :class:`Ted749Controller`, Orden TED/749/2020 apartado
      2.3.1: continuous proportional control on the voltage error, incremental
      to the pre-disturbance current, with a 5 s time latch.

    ``kwargs`` are forwarded to the selected controller (relay tunables, plus
    ``level`` / ``period`` to place the relay in a non-default latency tier).
    """
    devices = [n[len('lvrt_ext_'):] for n in getattr(model, 'u_run_names', [])
               if n.startswith('lvrt_ext_')]
    if not devices:
        return None
    key = str(law).lower()
    if key in ('ted749', 'ted', 'ted/749/2020', 'boe-a-2020-8965'):
        return Ted749Controller(devices, **kwargs)
    if key in ('po123', 'po12.3', 'vrt'):
        return VrtController(devices, **kwargs)
    raise ValueError(
        f"unknown FRT law {law!r}; expected 'po123' or 'ted749'")
