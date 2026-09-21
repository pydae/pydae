# -*- coding: utf-8 -*-
r"""
Discrete control orchestration for pydae power-system models.

A pydae DAE model captures the *continuous* dynamics. Many grid-following
elements also need a small **discrete** control part that runs once per solver
step -- e.g. the grid-code fault-ride-through relay
:func:`pydae.bps.pvs.pv_pq_ss.vrt`, which reads the terminal voltages and writes
``lvrt_ext`` / ``i_sr_ref`` before the next integration step.

The DAE solver (offline ``model.run`` loop, or the real-time API background
thread) should not have to know *which* elements have such a part or *how* to
call them. That knowledge is the model's; the solver only needs a single,
generic hook it can call every step.

:class:`DiscreteOrchestrator` is that hook. It is **generic** -- it holds a list
of discrete controllers and, on :meth:`step`, calls each one. The *particular*
behaviour lives with each model as a controller object (any object with a
``step(model)`` method, optionally ``sync(model)``), built by a module-level
``make_discrete_controller(model)`` factory that returns ``None`` when the model
has no such element. See :class:`pydae.bps.pvs.pv_pq_ss.VrtController`.

Latency tiers
-------------
Not every discrete part runs at the same rate. A local fast control (the FRT
relay ``vrt``) runs every solver step; a centralized controller (a plant PPC)
might run only every 200 ms. Each controller declares a **latency tier** and the
orchestrator runs it at that cadence. A controller's period (seconds) is
resolved in precedence order:

1. an explicit ``controller.period`` (seconds), else
2. ``tiers[controller.level]`` for a named ``controller.level``, else
3. ``0.0`` -- **low**, i.e. every step (the default when neither attribute is
   set, so a plain controller keeps running each step).

The default tiers are :data:`DEFAULT_TIERS` = ``{"low": 0.0, "mid": 0.05,
"high": 0.2}`` (seconds); pass ``tiers=`` to override. Cadence is expressed in
**seconds** so it ports across the offline loop (``model.Dt``) and the real-time
loop (``chunk_sec``), whose step sizes differ. Gating needs a clock: pass the
current simulation time to :meth:`step` as ``step(model, t)`` (offline: the loop
time or ``model.t``; real-time: ``simulator.t_sim``). Tiers gate *cadence*, not
*order* -- due controllers still run in registration order.

Discovery is explicit and side-effect-safe: the orchestrator probes a small list
of known discrete-capable model modules (``DEFAULT_DISCRETE_MODULES``) for that
factory, plus any factory registered via :func:`register_discrete`. It does not
blind-scan packages (the model dirs hold dev/test scripts that must not import).

Typical use::

    from pydae.bps.discrete import DiscreteOrchestrator

    discrete = DiscreteOrchestrator.from_model(model)   # generic, auto-discovers
    for t in times:
        discrete.step(model)          # run every element's discrete part
        model.run(t, {}, method='trapezoidal')

For the real-time API, attach it and the background loop calls it each chunk::

    app.state.model    = model
    app.state.discrete = DiscreteOrchestrator.from_model(model)
"""
import importlib
from typing import Any, Callable, Dict, List, Optional

# Named latency tiers -> period in seconds. "low" == every step; "mid" ~ fast
# measurement / feeder control; "high" ~ centralized controller (e.g. plant PPC).
DEFAULT_TIERS: Dict[str, float] = {"low": 0.0, "mid": 0.05, "high": 0.2}

# Floating-point tolerance when comparing the sim clock against a due time.
_EPS = 1e-9

# Modules known to expose a ``make_discrete_controller(model)`` factory.
# One entry can cover several element types that share an interface: the shared
# FRT factory in pydae.bps.utils.frt discovers every converter exposing
# ``lvrt_ext_<name>``, which both pv_pq_ss inverters and bess_pq_ss VSCs do, so a
# single controller drives them all (pv_pq_ss and bess_pq_ss re-export it). Add a
# module here (or use register_discrete) to extend coverage.
DEFAULT_DISCRETE_MODULES: List[str] = [
    "pydae.bps.utils.frt",
    "pydae.bps.ppcs.sppc",
]

# Factories registered at import time by model modules (optional extension point
# complementing DEFAULT_DISCRETE_MODULES). Deduplicated against it by identity.
_REGISTRY: List[Callable] = []


def register_discrete(factory: Callable) -> Callable:
    """Register a ``make_discrete_controller(model) -> controller | None`` factory.

    Usable as a decorator. Lets a model module opt into discrete orchestration
    without editing :data:`DEFAULT_DISCRETE_MODULES`. Returns *factory* so it can
    decorate.
    """
    if factory not in _REGISTRY:
        _REGISTRY.append(factory)
    return factory


def _collect_factories(modules: List[str]) -> List[Callable]:
    """Import each module path and collect its ``make_discrete_controller``.

    Missing modules or a missing factory are skipped silently -- the list is a
    best-effort catalogue, not a hard dependency.
    """
    factories: List[Callable] = []
    for modpath in modules:
        try:
            mod = importlib.import_module(modpath)
        except Exception:
            continue
        fac = getattr(mod, "make_discrete_controller", None)
        if callable(fac):
            factories.append(fac)
    return factories


class DiscreteOrchestrator:
    """Generic per-step driver for the discrete control parts of a model.

    Holds a list of *controllers* (any object with ``step(model)``; optionally
    ``sync(model)``). :meth:`step` calls each one, in registration order, once
    per solver step. It knows nothing about grid codes or converters -- the
    behaviour is entirely in the controllers, built by model-owned factories.
    """
    def __init__(self, controllers: Optional[List[Any]] = None,
                 tiers: Optional[Dict[str, float]] = None):
        self.controllers: List[Any] = list(controllers) if controllers else []
        # Named tier -> period (seconds). Copied so callers can't mutate ours.
        self.tiers: Dict[str, float] = dict(tiers) if tiers else dict(DEFAULT_TIERS)
        # Next due sim-time per controller (keyed by id); unset => run next step.
        self._next_due: Dict[int, float] = {}

    @classmethod
    def from_model(cls, model: Any, modules: Optional[List[str]] = None,
                   extra_factories: Optional[List[Callable]] = None,
                   tiers: Optional[Dict[str, float]] = None,
                   **factory_kwargs) -> "DiscreteOrchestrator":
        """Discover and build every discrete controller applicable to ``model``.

        Probes ``modules`` (default :data:`DEFAULT_DISCRETE_MODULES`) and the
        :func:`register_discrete` registry for ``make_discrete_controller``
        factories, calls each with ``model`` (and ``factory_kwargs``), and keeps
        the non-``None`` results. ``extra_factories`` appends caller-supplied
        factories. Factories are deduplicated by identity so a module listed in
        both ``modules`` and the registry runs once. ``tiers`` overrides the
        default latency-tier periods (see :data:`DEFAULT_TIERS`).
        """
        probe = modules if modules is not None else DEFAULT_DISCRETE_MODULES
        factories = _collect_factories(probe) + list(_REGISTRY)
        if extra_factories:
            factories += list(extra_factories)

        controllers: List[Any] = []
        seen: set = set()
        for fac in factories:
            if id(fac) in seen:
                continue
            seen.add(id(fac))
            controller = fac(model, **factory_kwargs)
            if controller is not None:
                controllers.append(controller)
        return cls(controllers, tiers=tiers)

    def add(self, controller: Any) -> None:
        """Append a controller built or configured by the caller."""
        self.controllers.append(controller)

    def sync(self, model: Any) -> None:
        """Re-seed every controller's internal state from the model.

        Call after an out-of-band write to the model inputs (e.g. a
        :class:`~pydae.bps.utils.faults.FaultManager` apply()/clear()) so the
        controllers do not fight the new operating point. Controllers without a
        ``sync`` method are skipped.
        """
        for controller in self.controllers:
            sync = getattr(controller, "sync", None)
            if callable(sync):
                sync(model)

    def reset_schedule(self) -> None:
        """Clear the tier timers so every controller fires on the next step.

        Useful after a discontinuity (a fault apply/clear, a mode change) when
        the slow tiers should re-evaluate immediately rather than wait out the
        rest of their period.
        """
        self._next_due.clear()

    def _period_of(self, controller: Any) -> float:
        """Resolve a controller's cadence in seconds (see module docstring)."""
        period = getattr(controller, "period", None)
        if period is not None:
            return float(period)
        level = getattr(controller, "level", None)
        if level is not None and level in self.tiers:
            return float(self.tiers[level])
        return 0.0

    def step(self, model: Any, t: Optional[float] = None) -> list:
        """Run the due controllers' discrete parts; return a per-controller list.

        ``t`` is the current simulation time (seconds), the clock the latency
        tiers gate on. If omitted it falls back to ``model.t``; if that is also
        unavailable the tiers cannot be gated and *every* controller runs (the
        ungated legacy behaviour). The returned list is aligned with
        :attr:`controllers`, holding each run controller's result and ``None``
        for those skipped this step.
        """
        if t is None:
            t = getattr(model, "t", None)

        results: list = []
        for controller in self.controllers:
            period = self._period_of(controller)
            if t is None or period <= 0.0:
                # No clock, or a "low" controller -> run every step.
                results.append(controller.step(model))
                continue
            due = self._next_due.get(id(controller))
            if due is None or t >= due - _EPS:
                results.append(controller.step(model))
                # Advance the schedule; clamp forward if we have fallen behind so
                # a long gap does not trigger a burst of catch-up runs.
                base = due if due is not None else t
                nxt = base + period
                self._next_due[id(controller)] = nxt if nxt > t else t + period
            else:
                results.append(None)
        return results

    def __len__(self) -> int:
        return len(self.controllers)

    def __bool__(self) -> bool:
        return bool(self.controllers)
