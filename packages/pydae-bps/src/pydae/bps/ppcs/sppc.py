# -*- coding: utf-8 -*-
r"""
Simple open-loop plant power controller (SPPC).

A minimal, **open-loop** plant-level controller: it exposes two plant setpoints
``p_poi_ref`` / ``q_poi_ref`` and, through a *discrete* control part (like the
converter ride-through relay :func:`pydae.bps.utils.frt.vrt`), pushes them to the
active/reactive-power setpoints (``p_s_ppc`` / ``q_s_ppc``) of a configured list
of PV and BESS converters. There is **no feedback** — it does not read the POI
power; it simply dispatches the reference.

Because a real plant controller is a *centralized, slow* loop, the discrete part
runs on the **high** latency tier of
:class:`pydae.bps.discrete.DiscreteOrchestrator` (~200 ms), unlike the local
per-step ``vrt`` relay.

Declared as a ``ppcs`` entry in the HJSON::

    ppcs: [{
        type: "sppc",
        name: "POI",
        pvs:  ["LV0101", "LV0102"],   // pv_pq_ss converter names
        bess: ["BESS01", "BESS02"],   // bess_pq_ss converter names
        p_poi_ref: 1.0,               // initial plant active setpoint (pu)
        q_poi_ref: 0.0,               // initial plant reactive setpoint (pu)
        share: false,                 // false: broadcast; true: split by count
        reg_bus: "POI",               // bus whose voltage gates the freeze (default: name)
        freeze_v: 0.8                 // freeze dispatch below this voltage (pu)
    }]

**Low-voltage freeze.** While the regulated bus voltage is below ``freeze_v``
(default 0.8 pu) the discrete dispatch is *frozen* -- the converter setpoints are
held so the plant controller does not fight the local ride-through relay during a
fault -- and it resumes automatically once the voltage recovers.

**Dispatch law.** With ``share=false`` (default) every converter's setpoint is
set to the plant reference (``p_s_ppc = p_poi_ref``): since each ``p_s_ppc`` is
per-unit of that converter's own rating, the plant then produces ``p_poi_ref``
per-unit of total capacity. With ``share=true`` the reference is split equally,
``p_s_ppc = p_poi_ref / N``.

The builder only *creates the inputs* and records the converter list in
``sys_dict['sppc']``; the dispatch itself is done by :class:`SppcController`
(discovered by the discrete orchestrator), so it also works over the real-time
API where ``p_poi_ref`` / ``q_poi_ref`` are written via ``POST /setpoints``.
"""


def descriptions():
    """Single source of truth for SPPC inputs (open loop: no states/params)."""
    d = []
    d += [{"type": "Input", "tex": "p_{poi,ref}", "data": "p_poi_ref",
           "model": "p_poi_ref", "default": 1.0,
           "description": "Plant active-power setpoint (dispatched to converters)",
           "units": "pu"}]
    d += [{"type": "Input", "tex": "q_{poi,ref}", "data": "q_poi_ref",
           "model": "q_poi_ref", "default": 0.0,
           "description": "Plant reactive-power setpoint (dispatched to converters)",
           "units": "pu"}]
    return d


def sppc(dae, data, name, backend):
    """Add the SPPC *name* to the DAE system.

    Registers the plant setpoints ``p_poi_ref`` / ``q_poi_ref`` (names overridable
    via ``p_ref_name`` / ``q_ref_name``) as runtime inputs and records the
    converter dispatch list in ``dae['sppc'][name]`` for :class:`SppcController`.
    Adds no equations or states — the dispatch is discrete/open-loop.
    """
    p_ref_name = data.get('p_ref_name', 'p_poi_ref')
    q_ref_name = data.get('q_ref_name', 'q_poi_ref')
    p_poi_ref = backend.symbols(p_ref_name)
    q_poi_ref = backend.symbols(q_ref_name)

    p_ref_0 = data.get('p_poi_ref', 1.0)
    q_ref_0 = data.get('q_poi_ref', 0.0)
    for d, val in ((p_ref_name, p_ref_0), (q_ref_name, q_ref_0)):
        dae['u_ini_dict'].update({d: val})
        dae['u_run_dict'].update({d: val})

    # Expose the setpoints as outputs too (observable via GET /measurements) and,
    # importantly, anchor the input symbols in the expression graph.
    dae['h_dict'].update({p_ref_name: p_poi_ref, q_ref_name: q_poi_ref})

    pvs = list(data.get('pvs', []))
    bess = list(data.get('bess', []))
    dae.setdefault('sppc', {})[name] = {
        'p_ref': p_ref_name, 'q_ref': q_ref_name,
        'pvs': pvs, 'bess': bess,
        'share': bool(data.get('share', False)),
        # Freeze the dispatch while the regulated bus voltage is depressed, so
        # the plant controller does not fight the local FRT relay during a fault.
        'reg_bus': data.get('reg_bus', name),
        'freeze_v': data.get('freeze_v', 0.8),
    }


# =============================================================================
# Discrete control part (runs on the "high" latency tier)
# =============================================================================
class SppcController:
    """Open-loop plant dispatch, run once per (high-tier) discrete step.

    Reads each plant's ``p_poi_ref`` / ``q_poi_ref`` and writes every configured
    converter's ``p_s_ppc`` / ``q_s_ppc`` (broadcast, or split by count when
    ``share``). Duck-typed for :class:`pydae.bps.discrete.DiscreteOrchestrator`
    (``step(model)`` + a ``level`` tier); it is the *particular* plant behaviour,
    the orchestrator the generic driver. ``level`` defaults to ``"high"`` so the
    centralized dispatch runs at ~200 ms, not every solver step.
    """
    def __init__(self, plants, level="high", period=None):
        self.plants = plants          # list of {p_ref, q_ref, converters, share}
        self.level = level            # latency tier read by the orchestrator
        self.period = period          # explicit cadence override (s), or None

    def step(self, model):
        """Dispatch the plant references; returns the number of converters set.

        While the regulated bus voltage is below ``freeze_v`` (default 0.8 pu) the
        dispatch is **frozen** -- the converter setpoints are left untouched so the
        plant controller does not fight the local ride-through during a fault --
        and it resumes automatically once the voltage recovers.
        """
        n = 0
        for pl in self.plants:
            if pl['reg_bus'] and model.get_value(f"V_{pl['reg_bus']}") < pl['freeze_v']:
                continue                       # frozen: hold setpoints (low voltage)
            p = model.get_value(pl['p_ref'])
            q = model.get_value(pl['q_ref'])
            k = len(pl['converters']) or 1
            p_i = p / k if pl['share'] else p
            q_i = q / k if pl['share'] else q
            for conv in pl['converters']:
                model.set_value(f'p_s_ppc_{conv}', p_i)
                model.set_value(f'q_s_ppc_{conv}', q_i)
                n += 1
        return n


def make_discrete_controller(model, level="high", **_ignored):
    """Discrete-controller factory discovered by :class:`DiscreteOrchestrator`.

    Builds one :class:`SppcController` covering every SPPC declared in
    ``model.sys_dict['sppc']``, or ``None`` when the system has none.
    """
    cfg = getattr(model, 'sys_dict', {}).get('sppc', {})
    plants = []
    for _name, c in cfg.items():
        converters = list(c.get('pvs', [])) + list(c.get('bess', []))
        if not converters:
            continue
        plants.append({'p_ref': c['p_ref'], 'q_ref': c['q_ref'],
                       'converters': converters, 'share': bool(c.get('share', False)),
                       'reg_bus': c.get('reg_bus'), 'freeze_v': c.get('freeze_v', 0.8)})
    if not plants:
        return None
    return SppcController(plants, level=level)
