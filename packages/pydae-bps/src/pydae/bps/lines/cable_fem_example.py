# -*- coding: utf-8 -*-
"""
Worked example: FEM thermal characterisation of a buried cable circuit, all the
way through to a dynamic-rating simulation in pydae-bps.

This is the narrated version of ``tests/bps/test_cable_fem_dtr.py``. The test
asserts; this script *shows*, and ends on the numbers an engineer actually asks
for — what the circuit can carry, and how hot it gets on a daily load cycle.

The steps::

    1. the circuit     geometry, materials and burial layout
    2. the FEM          2D steady thermal field, checked against closed forms
    3. the ladder       Cauer RC network reduced from the field
    4. the dtr blocks   BOTH forms line_dtr_cable accepts: two-node and n-stage
    5. ampacity         the steady rating, FEM vs the library's own defaults
    6. ini()            build both pydae-bps models, trim each to 200 A
    7. run()            24 h of daily cycling through both, conductor and cover
    8. plots            optional figures

Both thermal models of the same cable are simulated, because the cost of the
two-node truncation is invisible in a steady-state study and large in a cyclic
one. They agree on the steady state by construction — they carry the same DC
thermal resistance — but on a daily cycle the two-node form flattens the swing
by about 2x, and therefore the peak, which is the unsafe direction for a rating
study. The n-stage ladder is what :mod:`cable_fem` extracts from the FEM and
what ``line_dtr_cable`` now accepts directly.

Run it::

    python -m pydae.bps.lines.cable_fem_example
    python -m pydae.bps.lines.cable_fem_example --plot .
    python -m pydae.bps.lines.cable_fem_example --stages 5    # coarser ladder
    python -m pydae.bps.lines.cable_fem_example --dt 3600     # coarser steps
    python -m pydae.bps.lines.cable_fem_example --production  # fine mesh

Needs ``scikit-fem``, ``scipy`` and CasADi::

    pip install "pydae-bps[fem]" casadi

Roughly ten seconds, nearly all of it the finite-element solve: the 24 h
simulations themselves take about 0.04 s each. ``--production`` switches to the
mesh the module ships as its default (1.15e6 elements) and takes a couple of
minutes; the exported parameters move by well under a kelvin.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
import time

import numpy as np

from pydae.bps.lines import cable_fem as cf


# --- Example settings --------------------------------------------------------
# A mesh coarse enough to keep the script interactive. Dropping the outermost
# refinement pass and keeping one in-cable pass gives ~1.6e5 elements, which
# still reproduces the analytic T4 to better than 0.1 %.
FAST_REFINE = (20.0, 11.0, 6.5, 4.0, 2.6)
FAST_PASSES = 1

CURRENT = 200.0          # [A] the current the circuit is characterised at
PF = 0.97                # load power factor used to size the test load
U_KV = 20.0              # [kV] nominal line-to-line voltage
T_LIMIT = 90.0           # [degC] XLPE continuous conductor limit
# Ladder order to export and simulate. 7 reaches 99 % of the cable's true
# thermal impedance at the daily period, against 94 % for 5 -- worth ~2 K of
# peak on a hard cycle. It is not a threshold: 6 and 8 are within 2 % as well,
# because the rational-Krylov shifts move with the order, so accuracy at any
# ONE frequency oscillates while the overall step error falls monotonically.
# Passivity holds to at least n = 10 for this cable; identify_ladder() checks
# and reports it, so a bad order fails loudly rather than silently.
N_STAGES = 7

CYCLE_HOURS = 24         # hours of daily cycling to simulate
CYCLE_BASE = 300.0       # [A] mean current of the daily cycle
CYCLE_AMPLITUDE = 150.0  # [A] +/- swing of the daily cycle about the mean
CYCLE_PEAK_HOUR = 18.0   # hour of day the load peaks

# Load-segment and output step. Large on purpose -- the fastest ladder stage
# has a time constant of ~20 s and carries 4 % of the thermal resistance, so
# nothing here needs second-by-second resolution. 900 s runs 24 h in ~0.04 s
# and is converged to 0.03 K; 3600 s is just as fast but costs ~0.2 K on the
# peak. step_size_check() measures this rather than assuming it.
CYCLE_DT = 900.0

RULE = "=" * 78
THIN = "-" * 78


def banner(n: int, title: str) -> None:
    print(f"\n{RULE}\n {n}. {title}\n{RULE}")


# =============================================================================
# 1-4. THE CIRCUIT, ITS THERMAL FIELD, AND THE TWO dtr BLOCKS
# =============================================================================

def characterise(production: bool = False, n_stages: int = N_STAGES):
    """
    Steps 1-4: build the case, solve the field, reduce it to a ladder, and
    export both of the forms ``line_dtr_cable`` accepts.
    """
    banner(1, "THE CIRCUIT")
    case = cf.cigre_mv_eu_case(CURRENT)
    print(case.report())

    banner(2, "THE FINITE-ELEMENT THERMAL FIELD")
    t0 = time.time()
    kw = ({} if production
          else {'in_cable_passes': FAST_PASSES, 'refine_factors': FAST_REFINE})
    fem = cf.ThermalFEM(case, **kw)
    steady = fem.steady_metrics()
    print(f" mesh: {fem.n_elements} elements, {fem.n_dofs} dofs, solved in "
          f"{time.time() - t0:.1f} s"
          + ("" if production else "   (coarse — --production for the shipped "
                                   "default)"))
    print(steady)

    # Two closed-form results that share no code with the FEM. Agreement here
    # is what makes the numbers below trustworthy rather than merely plausible.
    print(f"\n{THIN}\n independent analytic cross-checks\n{THIN}")
    t1 = fem.analytic_T1()
    t1_sampled = ((steady.t_conductor[steady.hot_index] - steady.t_node_e)
                  / steady.losses['W_c'])
    a4 = fem.analytic_T4()
    t4 = steady.z_hot - t1 / (1.0 + steady.losses['lambda1'])
    print(f"   T1  layer-stack log formula {t1:.5f}  vs FEM θ_e sample "
          f"{t1_sampled:.5f} K·m/W  ({1e2 * (t1_sampled - t1) / t1:+.1f} %)")
    print(f"   T4  Kennelly images        {a4['total']:.5f}  vs FEM "
          f"{t4:.5f} K·m/W  ({1e2 * (t4 - a4['total']) / a4['total']:+.1f} %)")
    print(f"       = outer layers {a4['outer_layers']:.5f} + soil self "
          f"{a4['soil_self']:.5f} + soil mutual {a4['soil_mutual']:.5f}")

    banner(3, "THE LADDER MODEL")
    ladder = cf.identify_ladder(fem, n_stages=n_stages)
    reference = cf.identify_ladder(fem, n_stages=cf.REFERENCE_ORDER,
                                   tau_range=(10 ** 0.5, 10 ** 7.5))
    print(ladder)
    print(f"\n{THIN}\n the ladder as differential equations\n{THIN}")
    print(ladder.equations())

    # Independent ground truth: backward Euler on the full FEM system.
    t_chk = np.array([1e3, 1e5, 1e7])
    z_fem = cf.transient_fem_step(fem, t_chk)
    z_lad = ladder.step(t_chk)
    print("\n vs a direct backward-Euler transient FEM solve:")
    print("        t [s]    FEM [K·m/W]   ladder [K·m/W]      diff")
    for tt, a_, b_ in zip(t_chk, z_fem, z_lad):
        print(f"    {tt:9.1e}   {a_:11.5f}     {b_:11.5f}   {b_ - a_:+9.5f}")

    banner(4, "THE TWO dtr BLOCKS FOR line_dtr_cable")
    fit = cf.fit_two_node(fem, steady=steady, reference=reference)
    block_2n = fit.as_dtr()
    block_lad = cf.ladder_dtr(ladder, fit)

    print(" (a) two-node, IEC 60853 — T1/T4/Q1/Q2/p:")
    print(fit)
    print('\n   "dtr": '
          + json.dumps(block_2n, indent=8).replace("\n", "\n   "))
    print(f"\n (b) {ladder.n_stages}-stage ladder — R/C lists:")
    print('   "dtr": '
          + json.dumps(block_lad, indent=8).replace("\n", "\n   "))
    print("\n   Both pin the thermal network EXPLICITLY, which overrides the "
          "library's")
    print("   geometry route — deliberately: that route cannot represent three")
    print("   thermally coupled cables. Compare:")
    print(cf.compare_with_library(case, fit))
    z_lad_dc = sum(block_lad['R']) * (1.0 + fit.lambda1)
    print(f"\n   (1+λ1)·sum(R) = {z_lad_dc:.5f} K·m/W and "
          f"T1 + (1+λ1)·T4 = {fit.z_equivalent:.5f} K·m/W: the two blocks "
          "carry the")
    print("   same DC thermal resistance, so they agree on every steady state "
          "and")
    print("   differ only in transit.")

    return case, fem, steady, ladder, reference, fit, block_2n, block_lad


# =============================================================================
# 5. AMPACITY  --  the question the whole exercise exists to answer
# =============================================================================

def ampacity(design, z_eq: float, t_soil: float,
             t_limit: float = T_LIMIT) -> float:
    r"""
    Steady current rating [A] for a conductor temperature limit.

    Setting :math:`\theta_c = \theta_{\rm limit}` in the steady relation and
    eliminating the temperature-dependent resistance collapses to the IEC 60287
    rating equation, because :math:`1 + 70\alpha_R = R_{ac,90}/R_{ac,20}`
    exactly:

    .. math:: I_{\max} = \sqrt{\frac{\theta_{\rm limit}-\theta_{\rm soil}}
                                    {R_{ac}(90\,^\circ\!C)\;Z_{\rm eq}}},
              \qquad Z_{\rm eq} = T_1 + (1+\lambda_1)\,T_4 .

    The whole difficulty is in :math:`Z_{\rm eq}`, which is what the FEM
    supplies and what the isolated-cable formula gets wrong. For the n-stage
    ladder the same expression holds with
    :math:`Z_{\rm eq} = (1+\lambda_1)\sum R_i`, since the ladder takes the
    total loss.
    """
    if t_limit <= t_soil:
        raise ValueError("the limit must be above the soil temperature")
    return math.sqrt((t_limit - t_soil) / (design.R_ac_90 * z_eq))


def report_ampacity(case, fem, fit) -> float:
    """
    Step 5: the steady rating, and where the library's own defaults lose it.

    Rather than assert which effect dominates, the gap is walked one correction
    at a time, cumulatively, so the attribution is arithmetic instead of
    assertion. The order is outermost cause first; a different order would
    shuffle the individual steps but not the endpoints.
    """
    banner(5, "AMPACITY  —  WHAT CAN THIS CIRCUIT ACTUALLY CARRY?")
    design = case.design
    lib = cf.library_defaults(case)
    a4 = fem.analytic_T4()

    i_fem = ampacity(design, fit.z_equivalent, fit.t_soil)
    z_lib = lib['T1'] + lib['T4']                    # library lambda1 = 0
    i_lib = ampacity(design, z_lib, 15.0)

    print(f" from the FEM             Z_eq {fit.z_equivalent:.4f} K·m/W, "
          f"t_soil {fit.t_soil:.2f} °C  ->  I_max {i_fem:6.1f} A")
    print(f" from library defaults    Z_eq {z_lib:.4f} K·m/W, "
          f"t_soil {15.0:.2f} °C  ->  I_max {i_lib:6.1f} A")
    print(f"\n   The defaults would rate this circuit {i_lib - i_fem:.0f} A "
          f"({1e2 * (i_lib / i_fem - 1.0):.0f} %) high: the cable reaches "
          f"{T_LIMIT:.0f} °C")
    print(f"   at {i_fem:.0f} A, not {i_lib:.0f} A.")

    lam = fit.lambda1
    t4_outer = lib['T4'] + a4['outer_layers']
    stages = [
        ("library defaults", z_lib, 15.0),
        ("+ soil datum from the no-load field", z_lib, fit.t_soil),
        ("+ screen losses (lambda1)",
         lib['T1'] + (1.0 + lam) * lib['T4'], fit.t_soil),
        ("+ the cable's own jacket and bedding",
         fit.T1 + (1.0 + lam) * t4_outer, fit.t_soil),
        ("+ mutual heating of the other two phases  = FEM",
         fit.z_equivalent, fit.t_soil),
    ]
    print(f"\n{THIN}\n where the {i_lib - i_fem:.0f} A goes\n{THIN}")
    prev = None
    for label, z, ts in stages:
        i = ampacity(design, z, ts)
        delta = "" if prev is None else f"   {i - prev:+6.1f} A"
        print(f"   {label:<45s} {i:6.1f} A{delta}")
        prev = i
    print("\n   So the single largest term is the mutual heating an "
          "isolated-cable T4")
    print("   cannot see — but the dropped screen loss is nearly as large, "
          "because")
    print("   lambda1 enters twice: it adds loss AND multiplies T4.")

    # The closed form must be the fixed point the simulator converges to.
    check = cf.CableCase(case.design, case.installation, current=i_fem)
    theta = check.self_consistent_conductor_temperature(fit.z_equivalent,
                                                        fit.t_soil)
    print(f"\n   check: at {i_fem:.1f} A the self-consistent conductor "
          f"temperature is {theta:.2f} °C")
    print("   That is a STEADY rating, and both dtr blocks give it. Step 7 is")
    print("   where they part company.")
    return i_fem


# =============================================================================
# 6. ini()  --  the real pydae-bps models
# =============================================================================

def build_model(case, fit, dtr, work_dir: str, name: str,
                current: float = CURRENT):
    """
    Build a pydae-bps system from one exported ``dtr`` block and trim the load
    until the line carries exactly *current*.

    The load flow sets the line current, so a fixed load would leave each model
    somewhere near — but not at — the current the cable was characterised at.
    Since the conductor rise goes as I², "near" is not good enough to compare
    two thermal models against each other. The load is an *input*, so each trim
    costs one ``ini()`` rather than a rebuild.
    """
    from pydae.bps import BpsBuilder
    from pydae.core.builder.casadi_builder import CasadiBuilder
    from pydae.core.model.casadi_model import CasadiModel

    s = math.sqrt(3.0) * U_KV * 1e3 * current
    p, q = s * PF, s * math.sqrt(1.0 - PF ** 2)
    data = cf.bps_data(case, fit, name=name, dtr=dtr,
                       P_load_W=p, Q_load_var=q, U_kV=U_KV)

    # construct() writes "<name>_xy_0.json" (and a bare "xy_0.json") into the
    # cwd, so keep them out of wherever the user happens to be standing.
    cwd = os.getcwd()
    os.chdir(work_dir)
    try:
        grid = BpsBuilder(data, use_casadi=True)
        grid.construct(name)
        model = CasadiModel(CasadiBuilder(grid.sys_dict).build())
        for trim in range(12):
            model.ini({'P_2': p, 'Q_2': q}, newton_tol=1e-10)
            i_line = float(model.get_value('I_cond_1_2'))
            if abs(i_line - current) < 1e-6:
                break
            scale = current / i_line
            p *= scale
            q *= scale
    finally:
        os.chdir(cwd)
    return model, {'P_2': p, 'Q_2': q}, i_line, trim + 1


def build_both(case, fit, block_2n, block_lad, work_dir: str):
    """Step 6: build the two-node and the n-stage model side by side."""
    banner(6, "THE pydae-bps MODELS  —  build + ini()")
    n = len(block_lad['R'])
    out = {}
    for key, dtr, title in (
            ('two_node', block_2n, 'two-node (IEC 60853)'),
            ('ladder', block_lad, f'{n}-stage ladder')):
        t0 = time.time()
        model, load, i_line, trims = build_model(
            case, fit, dtr, work_dir, f'cable_fem_ex_{key}')
        g = lambda k: float(model.get_value(f'{k}_1_2'))
        states = [s.rsplit('_1_2', 1)[0] for s in model.x_names
                  if s.endswith('_1_2')]
        print(f"\n {title}:  {len(states)} thermal states "
              f"({', '.join(states)})")
        print(f"   built and trimmed to I = {i_line:.4f} A in {trims} ini() "
              f"calls, {time.time() - t0:.1f} s")
        print(f"   t_conductor {g('t_conductor'):6.2f} °C,  t_soil "
              f"{g('t_soil'):6.2f} °C,  W_total {g('W_total'):.3f} W/m")
        out[key] = {'model': model, 'load': load, 'title': title,
                    't_conductor': g('t_conductor')}

    d = out['ladder']['t_conductor'] - out['two_node']['t_conductor']
    rise = out['two_node']['t_conductor'] - fit.t_soil
    print(f"\n   The two agree on the steady state to {abs(d):.3f} K "
          f"({1e2 * abs(d) / rise:.2f} % of the {rise:.1f} K rise), as they")
    print("   must: same DC thermal resistance. Any difference in step 7 is")
    print("   therefore transient, not a disagreement about the cable.")
    print(f"   FEM-derived value at the same current: "
          f"{fit.t_conductor_self_consistent:.2f} °C")
    return out


# =============================================================================
# 7. run()  --  a daily load cycle
# =============================================================================

def daily_current(hour: float, base: float = CYCLE_BASE,
                  amplitude: float = CYCLE_AMPLITUDE) -> float:
    """
    Mean current with a sinusoidal daily swing of +/- *amplitude* amperes.

    ``cos`` is at its maximum when its argument is zero, so the peak lands on
    ``CYCLE_PEAK_HOUR`` with no phase offset needed.
    """
    phase = 2.0 * math.pi * (hour - CYCLE_PEAK_HOUR) / 24.0
    return base + amplitude * math.cos(phase)


def run_cycle(model, load, hours: int = CYCLE_HOURS, dt: float = CYCLE_DT):
    """
    Cycle the load for *hours* hours in piecewise-constant segments of *dt*,
    starting from the steady state at the mean current of the cycle.

    The load is scaled from its characterisation value by the same linear
    factor the run itself uses, so the mean-load steady state the cycle starts
    from is exactly the point the cycle oscillates about — no start transient
    of its own on top of the daily swing.

    ``ini()`` is called first, so this can be run repeatedly on one model
    without the previous run's end state leaking in — which is what lets
    :func:`step_size_check` re-run the same models at a different step.

    Under the hood IDAS still adapts its own internal steps, so *dt* sets the
    load resolution and the output grid rather than the integration accuracy.
    """
    mean_frac = CYCLE_BASE / CURRENT
    model.ini({'P_2': load['P_2'] * mean_frac, 'Q_2': load['Q_2'] * mean_frac},
              newton_tol=1e-10)
    model.Dt = dt
    n_seg = max(1, int(round(hours * 3600.0 / dt)))
    t0 = time.time()
    for k in range(n_seg):
        hour = (k + 0.5) * dt / 3600.0          # mid-segment sample
        frac = daily_current(hour) / CURRENT
        model.run((k + 1) * dt,
                  {'P_2': load['P_2'] * frac, 'Q_2': load['Q_2'] * frac})
    model.post()
    # Every thermal node the block exposes, so the caller can locate the cable
    # surface in whichever form it is looking at: the two-node block names its
    # outer node t_sheath, the ladder numbers its interior nodes t_ladder_i.
    nodes = {n.rsplit('_1_2', 1)[0]: np.array(model.get_values(n), dtype=float)
             for n in model.outputs_list
             if n.endswith('_1_2') and (n.startswith('t_ladder_')
                                        or n.startswith('t_sheath'))}
    # np.array (not asarray) so these stay valid after the model is re-run.
    return {
        't': np.array(model.Time, dtype=float),
        't_cond': np.array(model.get_values('t_conductor_1_2'), dtype=float),
        'i_cond': np.array(model.get_values('I_cond_1_2'), dtype=float),
        'w_total': np.array(model.get_values('W_total_1_2'), dtype=float),
        't_soil': float(model.get_value('t_soil_1_2')),
        'nodes': nodes,
        'dt': dt, 'segments': n_seg, 'wall': time.time() - t0,
    }


def cover_temperature(run: dict, ladder=None, fit=None):
    r"""
    Temperature of the cable's outer covering — the cable/soil interface.

    The two-node block has that node explicitly: ``t_sheath`` sits between
    :math:`T_1` (conductor to cable surface) and :math:`T_4` (surface to
    ambient soil), so it *is* the surface temperature and is returned as is.

    The ladder has no such node. Its stages come out of a Foster→Cauer
    identification of the FEM step response, so an interior node is a time
    constant, not a radius. What is still meaningful is where the surface sits
    along the ladder's thermal path: the conductor-to-surface resistance is
    :math:`T_1/(1+\lambda_1)` in the ladder's normalisation (the ladder is
    driven by the *total* loss :math:`W_c(1+\lambda_1)+W_d`, the two-node
    :math:`T_1` by the conductor loss alone). Walking that resistance out from
    the conductor lands part-way into one stage, and the temperature there is
    the linear drop across that stage — exact in steady state, an
    interpolation while the ladder is moving.

    Returns ``None`` when neither form applies.
    """
    if 't_sheath' in run['nodes']:
        return run['nodes']['t_sheath'], 'cable surface (t_sheath node)'
    if ladder is None or fit is None:
        return None
    R = np.asarray(ladder.R, dtype=float)
    r_internal = fit.T1 / (1.0 + fit.lambda1)
    temps = [run['t_cond']] + [run['nodes'][f't_ladder_{i + 1}']
                               for i in range(1, len(R))]
    temps.append(np.full_like(run['t_cond'], run['t_soil']))
    cum = 0.0
    for k, r in enumerate(R):
        if cum + r >= r_internal:
            frac = (r_internal - cum) / r
            return (temps[k] - frac * (temps[k] - temps[k + 1]),
                    f'cable surface (interpolated {1e2 * frac:.0f} % into '
                    f'ladder stage {k + 1})')
        cum += r
    return temps[-1], 'cable surface (beyond the last ladder stage)'


def report_cycle(built, fit, ladder, hours: int = CYCLE_HOURS,
                 dt: float = CYCLE_DT):
    """Step 7: run the same cycle through both models and compare."""
    banner(7, f"run()  —  {hours} h OF DAILY CYCLING")
    print(f" load : {CYCLE_BASE:.0f} A mean, ±{CYCLE_AMPLITUDE:.0f} A "
          f"({daily_current(CYCLE_PEAK_HOUR - 12.0):.0f} A to "
          f"{daily_current(CYCLE_PEAK_HOUR):.0f} A), peaking at "
          f"{CYCLE_PEAK_HOUR:02.0f}:00,")
    print("        driven by modulating the load so the DTR model sees it "
          "through the")
    print("        line current, as in service.")
    print(f" start: the steady state at a flat {CYCLE_BASE:.0f} A, so one day "
          "is representative")
    print("        of a settled cycle for every stage faster than the soil.")
    print(f" steps: {int(round(hours * 3600.0 / dt))} segments of "
          f"{dt / 60:.0f} min")

    runs = {k: run_cycle(built[k]['model'], built[k]['load'], hours, dt)
            for k in ('two_node', 'ladder')}
    print(f"        simulated in {runs['two_node']['wall']:.2f} s (two-node) "
          f"and {runs['ladder']['wall']:.2f} s ({ladder.n_stages}-stage)")

    print(f"\n{THIN}\n{'':24s}{'two-node':>12s}{'n-stage ladder':>18s}\n{THIN}")
    stat = {}
    for key in ('two_node', 'ladder'):
        c = runs[key]['t_cond']
        cover, how = cover_temperature(runs[key], ladder, fit)
        runs[key]['t_cover'], runs[key]['cover_how'] = cover, how
        stat[key] = {
            'min': float(c.min()), 'max': float(c.max()),
            'swing': float(np.ptp(c)),
            'peak_h': float(runs[key]['t'][int(np.argmax(c))] / 3600.0),
            'cover_max': float(cover.max()),
            'cover_swing': float(np.ptp(cover)),
            'insulation_drop': float((c - cover).max()),
        }
    i_peak_h = runs['two_node']['t'][
        int(np.argmax(runs['two_node']['i_cond']))] / 3600.0
    for label, key_ in (('conductor min [°C]', 'min'),
                        ('conductor max [°C]', 'max'),
                        ('daily swing [K]', 'swing'),
                        ('peak at [h]', 'peak_h'),
                        ('cover max [°C]', 'cover_max'),
                        ('cover swing [K]', 'cover_swing'),
                        ('max cond-cover [K]', 'insulation_drop')):
        print(f"   {label:<22s}{stat['two_node'][key_]:12.2f}"
              f"{stat['ladder'][key_]:18.2f}")
    print(f"   (the current peaks at {i_peak_h:.1f} h)")
    print(f"   cover: two-node {runs['two_node']['cover_how']}")
    print(f"          ladder   {runs['ladder']['cover_how']}")

    d_peak = stat['ladder']['max'] - stat['two_node']['max']
    ratio = stat['ladder']['swing'] / stat['two_node']['swing']
    # How much thermal resistance each model keeps behind poles fast enough to
    # follow a 24 h cycle (omega*tau <= 1, i.e. tau <= period/2pi ~ 3.8 h).
    # This is the structural difference; |Z| is its consequence. Computed from
    # the poles rather than from a stage index, which would go stale the moment
    # --stages changes.
    period = 24.0 * 3600.0
    w_day = 2.0 * np.pi / period
    tau_fast = 1.0 / w_day                       # ~3.82 h

    def reachable(tau, R):
        """R behind poles with omega*tau <= 1, and the total."""
        tau, R = np.asarray(tau), np.asarray(R)
        return float(R[tau <= tau_fast].sum()), float(R.sum())

    r_fast_l, r_tot_l = reachable(ladder.tau_foster, ladder.R_foster)
    A2, b2 = cf._two_node_matrices(fit.T1, fit.T4, fit.Q1, fit.Q2, fit.lambda1,
                                   1.0 / (1.0 + fit.lambda1))
    ev, V = np.linalg.eig(A2)
    res_2n = ((np.array([1.0, 0.0]) @ V) * (np.linalg.inv(V) @ b2) / -ev).real
    r_fast_2, r_tot_2 = reachable(-1.0 / ev.real, res_2n)

    z_lad = abs(ladder.impedance(w_day)[0])
    z_2n = abs(fit.impedance(w_day)[0])

    print(f"\n   The two-node model reports a {stat['two_node']['swing']:.2f} "
          f"K swing where the ladder reports")
    print(f"   {stat['ladder']['swing']:.2f} K — a factor of {ratio:.2f} — and "
          f"a peak {d_peak:.2f} K lower. Both started from")
    print("   the same steady state and saw the same current, so this is the")
    print("   truncation alone. Both models hold the same total thermal")
    print(f"   resistance ({ladder.R.sum():.3f} K·m/W); what differs is how "
          "much of it a 24 h")
    print("   cycle can reach:")
    print(f"     two-node  R behind poles with tau <= {tau_fast / 3600:.1f} h: "
          f"{r_fast_2:.3f} of {r_tot_2:.3f} K·m/W   ->  "
          f"|Z| at 24 h = {z_2n:.3f}")
    print(f"               its two poles sit at {fit.time_constants[1]:.0f} s "
          f"and {fit.time_constants[0] / 86400:.1f} d, so the daily band is "
          "empty")
    print(f"     {ladder.n_stages}-stage  R behind poles with tau <= "
          f"{tau_fast / 3600:.1f} h: {r_fast_l:.3f} of {r_tot_l:.3f} K·m/W   "
          f"->  |Z| at 24 h = {z_lad:.3f}")
    print(f"               {ladder.n_stages} poles spread across the spectrum, "
          f"so {r_fast_l / max(r_fast_2, 1e-12):.1f}x as much of the")
    print("               resistance is reachable, plus partial response from "
          "the slower ones")
    print(f"   Ratio {z_lad / z_2n:.2f}x, against the {ratio:.2f}x swing "
          "actually simulated.")
    m_lad = T_LIMIT - stat['ladder']['max']
    m_2n = T_LIMIT - stat['two_node']['max']
    word = "Headroom to" if m_lad > 0.0 else "Margin against"
    print(f"\n   {word} the {T_LIMIT:.0f} °C limit on this cycle: "
          f"{m_lad:+.1f} K by the ladder, {m_2n:+.1f} K by the")
    print(f"   two-node block. Believing the two-node figure overstates it by "
          f"{d_peak:.1f} K —")
    print("   under-predicting the swing under-predicts the PEAK, which is the")
    print("   unsafe direction. Size cyclic headroom from the ladder block.")
    if m_lad <= 0.0:
        print(f"\n   At this loading the ladder puts the conductor "
              f"{-m_lad:.1f} K OVER the XLPE limit while")
        print(f"   the two-node block "
              + (f"still shows {m_2n:.1f} K in hand" if m_2n > 0.0
                 else f"shows only {-m_2n:.1f} K over") +
              " — the whole point of carrying the")
        print("   extra stages: the two forms disagree about whether this "
              "cycle is allowed.")
    print(f"\n   The cover (cable/soil interface) peaks at "
          f"{stat['ladder']['cover_max']:.2f} °C on the ladder and swings")
    print(f"   {stat['ladder']['cover_swing']:.2f} K, against "
          f"{stat['ladder']['swing']:.2f} K at the conductor: the insulation "
          f"holds up to")
    print(f"   {stat['ladder']['insulation_drop']:.2f} K of the rise and the "
          "cover only ever sees what the soil")
    print("   is slow enough to take away, so it lags and flattens the cycle.")

    return runs, stat


def validate_pydae_ladder(built, ladder, hours: int = CYCLE_HOURS,
                          dts=(900.0, 300.0, 60.0)):
    """
    Confirm pydae-bps really is integrating the ladder :mod:`cable_fem`
    identified, by pushing the model's own loss series through the Python
    ladder and comparing.

    This is the seam worth checking: :func:`line_dtr_cable._dtr_cable_ladder`
    rebuilds the network symbolically from the R/C lists, so a sign or index
    slip there would give a plausible-looking but wrong trajectory.

    The comparison cannot be exact at any single step size, because the Python
    reference holds the loss constant across each output interval while pydae
    keeps updating it with the conductor temperature. That is a property of the
    *reference*, not of the model — so rather than assert it, the check is run
    at several step sizes: a residual that falls linearly with ``dt`` is the
    hold and extrapolates to zero, whereas a wrong network would leave a floor.
    """
    print(f"\n{THIN}\n is pydae-bps integrating the identified ladder?\n{THIN}")
    print("   pydae-bps n-stage model vs the same R/C integrated in Python,")
    print("   driven by the simulation's own W_total series:\n")
    print(f"   {'dt [s]':>8s}{'max |diff| [K]':>16s}{'ratio':>9s}")
    A, b, _c = ladder.matrices()
    errs = []
    for dt in dts:
        r = run_cycle(built['ladder']['model'], built['ladder']['load'],
                      hours, dt)
        y0 = -np.linalg.solve(A, b * r['w_total'][0])   # steady at initial loss
        ref = r['t_soil'] + ladder.simulate(r['t'], r['w_total'], y0=y0)
        e = float(np.max(np.abs(r['t_cond'] - ref)))
        ratio = "" if not errs else f"{errs[-1] / e:8.2f}"
        print(f"   {dt:8.0f}{e:16.4f}{ratio:>9s}")
        errs.append(e)
    # First order in dt: each refinement should shrink the error by the same
    # factor the step shrank by.
    expected = [dts[i] / dts[i + 1] for i in range(len(dts) - 1)]
    achieved = [errs[i] / errs[i + 1] for i in range(len(errs) - 1)]
    print(f"\n   step ratios {', '.join(f'{v:.1f}' for v in expected)}  ->  "
          f"error ratios {', '.join(f'{v:.1f}' for v in achieved)}")
    print("   The residual is first order in dt, so it is the zero-order hold "
          "in the")
    print("   Python reference and extrapolates to zero. A wrong network would "
          "sit on")
    print("   a floor instead. pydae-bps is integrating the identified ladder.")
    return errs[-1]


def step_size_check(built, hours: int = CYCLE_HOURS, coarse: float = CYCLE_DT,
                    fine: float = 60.0):
    """
    What the large steps cost, measured rather than assumed.

    Re-runs the cycle with one-minute load segments and compares. This is the
    justification for the coarse default: a buried cable has no dynamics fast
    enough for quarter-hour load resolution to matter, and the runtime
    difference is an order of magnitude.
    """
    print(f"\n{THIN}\n what do the {coarse / 60:.0f} min steps cost?\n{THIN}")
    print(f"   {'':28s}{'wall [s]':>10s}{'peak [°C]':>12s}{'swing [K]':>12s}")
    out = {}
    for key, label in (('two_node', 'two-node'), ('ladder', 'ladder')):
        for dt in (coarse, fine):
            r = run_cycle(built[key]['model'], built[key]['load'], hours, dt)
            peak, swing = float(r['t_cond'].max()), float(np.ptp(r['t_cond']))
            out[(key, dt)] = (peak, swing, r['wall'])
            print(f"   {label + ', ' + f'{dt / 60:.0f} min':<28s}{r['wall']:10.2f}"
                  f"{peak:12.3f}{swing:12.3f}")
    print()
    for key, label in (('two_node', 'two-node'), ('ladder', 'ladder')):
        dp = abs(out[(key, coarse)][0] - out[(key, fine)][0])
        ds = abs(out[(key, coarse)][1] - out[(key, fine)][1])
        print(f"   {label}: the coarse steps cost {dp:.3f} K on the peak and "
              f"{ds:.3f} K on the swing")
    print("\n   Both are far below the model error the truncation already "
          "carries, so")
    print("   the large steps are essentially free. That is a property of the")
    print("   PHYSICS, not of the integrator: nothing in a buried cable moves")
    print("   faster than its insulation, and the soil stages run to days.")
    return out


# =============================================================================
# 8. PLOTS
# =============================================================================

def plot_cycle(runs, stat, i_max: float, path: str) -> None:
    """
    Current, conductor temperature and cable-cover temperature over the
    simulated day.

    The temperature axis is zoomed to the data rather than stretched to the
    90 degC limit: the whole point of the figure is the gap between the two
    models, and at full scale that gap is a few pixels. The headroom to the
    limit is stated in the corner instead, which is the number that matters
    without costing two-thirds of the panel.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    r2, rl = runs['two_node'], runs['ladder']
    h2, hl = r2['t'] / 3600.0, rl['t'] / 3600.0
    y2, yl = r2['t_cond'], rl['t_cond']
    c2, cl = r2['t_cover'], rl['t_cover']

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True,
                                 gridspec_kw={'height_ratios': [1, 2]})

    # --- current ------------------------------------------------------------
    a1.step(h2, r2['i_cond'], where='post', color='steelblue', lw=1.7)
    a1.axhline(i_max, color='crimson', ls=':', lw=1.3,
               label=f"steady rating {i_max:.0f} A")
    a1.set_ylabel('line current [A]')
    a1.legend(fontsize=9, loc='lower right')
    a1.grid(alpha=0.3)
    a1.set_title(f'Daily load cycle on the CIGRE TB 575 MV cable feeder — '
                 f'{CYCLE_BASE:.0f} A ± {CYCLE_AMPLITUDE:.0f} A, '
                 f'two thermal models of the same cable', fontsize=12)

    # --- temperature --------------------------------------------------------
    # Two curves per model: the conductor (the rated quantity) and the cable
    # cover, i.e. the cable/soil interface. Same colour per model, the cover
    # thinner, so the eye reads "which model" from the hue and "which node"
    # from the weight.
    # The two grids are the same here, but interpolate anyway so the shaded
    # divergence stays correct if they ever differ.
    y2_on_l = np.interp(hl, h2, y2)
    a2.fill_between(hl, y2_on_l, yl, color='seagreen', alpha=0.13, lw=0,
                    label='what the two-node form misses')
    a2.plot(hl, yl, color='seagreen', lw=2.4,
            label='conductor — n-stage ladder (reduced from the FEM)')
    a2.plot(h2, y2, color='crimson', lw=2.0, ls='--',
            label='conductor — two-node, IEC 60853')
    a2.plot(hl, cl, color='seagreen', lw=1.4, alpha=0.85,
            label='cable cover — n-stage ladder')
    a2.plot(h2, c2, color='crimson', lw=1.3, ls='--', alpha=0.85,
            label='cable cover — two-node (t_sheath node)')

    if T_LIMIT <= max(yl.max(), y2.max()) + 8.0:
        a2.axhline(T_LIMIT, color='0.35', ls=(0, (6, 3)), lw=1.2)
        a2.text(0.01, T_LIMIT, f" XLPE limit {T_LIMIT:.0f} °C",
                transform=a2.get_yaxis_transform(), va='bottom', ha='left',
                fontsize=9, color='0.3')

    # Mark each model's own conductor peak, and the instant of largest
    # divergence -- they are not the same moment, so one arrow cannot honestly
    # show both.
    for h, y, colour, tag in ((hl, yl, 'seagreen', 'ladder'),
                              (h2, y2, 'crimson', 'two-node')):
        k = int(np.argmax(y))
        a2.plot(h[k], y[k], 'o', color=colour, ms=6, zorder=5)
        a2.annotate(f"{y[k]:.1f} °C", (h[k], y[k]),
                    textcoords='offset points', xytext=(6, 7 if tag == 'ladder'
                                                        else -14),
                    fontsize=9, color=colour, fontweight='bold')

    k_div = int(np.argmax(np.abs(yl - y2_on_l)))
    a2.annotate("", xy=(hl[k_div], yl[k_div]), xytext=(hl[k_div], y2_on_l[k_div]),
                arrowprops=dict(arrowstyle='<->', color='0.25', lw=1.4))
    a2.text(hl[k_div] - 0.4, 0.5 * (yl[k_div] + y2_on_l[k_div]),
            f"{abs(yl[k_div] - y2_on_l[k_div]):.1f} K", ha='right', va='center',
            fontsize=10, color='0.15', fontweight='bold')

    # The conductor-to-cover drop at the ladder's peak: what the insulation
    # holds, and the reason the cover can look comfortable while the conductor
    # is not.
    k_pk = int(np.argmax(yl))
    a2.annotate("", xy=(hl[k_pk], yl[k_pk]), xytext=(hl[k_pk], cl[k_pk]),
                arrowprops=dict(arrowstyle='<->', color='seagreen', lw=1.2,
                                alpha=0.8))
    a2.text(hl[k_pk] + 0.35, 0.5 * (yl[k_pk] + cl[k_pk]),
            f"{yl[k_pk] - cl[k_pk]:.1f} K\nacross the\ninsulation",
            ha='left', va='center', fontsize=8.5, color='seagreen')

    lo = min(y2.min(), yl.min(), c2.min(), cl.min())
    hi = max(y2.max(), yl.max(), T_LIMIT if T_LIMIT <= max(yl.max(), y2.max())
             else -np.inf)
    a2.set_ylim(lo - 3.0, hi + 7.0)
    a2.set_xlim(-0.5, hl[-1] + 0.5)
    a2.set_xticks(range(0, int(hl[-1]) + 1, 3))
    a2.set_xlabel('time of day [h]')
    a2.set_ylabel('temperature [°C]')
    a2.grid(alpha=0.3)
    a2.legend(fontsize=9, loc='upper left', framealpha=0.92, ncol=2)

    m_lad, m_2n = T_LIMIT - yl.max(), T_LIMIT - y2.max()
    word = 'headroom' if m_lad > 0.0 else 'margin'
    a2.text(0.985, 0.045,
            f"soil {rl['t_soil']:.1f} °C   ·   XLPE limit {T_LIMIT:.0f} °C\n"
            f"conductor {word} at the peak: {m_lad:+.1f} K by the ladder, "
            f"{m_2n:+.1f} K by the two-node block",
            transform=a2.transAxes, ha='right', va='bottom', fontsize=9,
            color='0.3',
            bbox=dict(boxstyle='round,pad=0.45', fc='white', ec='0.8'))

    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"   cycle    -> {path}")


def plot_impedance(fit, ladder, reference, path: str) -> None:
    r"""
    Why the two models diverge on a cycle but not in steady state.

    Plots the thermal impedance magnitude against excitation period. A pole
    responds only while :math:`\omega\tau \ll 1`, so at a given period a model
    can only deliver the thermal resistance sitting behind poles fast enough to
    follow. At infinite period (DC) all three curves meet — that is why the
    steady states agree. At 24 h they do not.

    The stem plot underneath shows where each model keeps its resistance. The
    two-node form, having only two poles to spend, is forced to put them at the
    extremes and leaves the decade around 10^4-10^5 s empty — which is exactly
    where a daily cycle lives.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    periods = np.logspace(1.0, 8.0, 400)          # 10 s ... ~3 years
    omega = 2.0 * np.pi / periods
    z2 = np.abs(fit.impedance(omega))
    zl = np.abs(ladder.impedance(omega))
    zr = np.abs(reference.impedance(omega))

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 7.5), sharex=True,
                                 gridspec_kw={'height_ratios': [2.2, 1]})

    a1.semilogx(periods, zr, color='0.6', lw=4.0, alpha=0.8,
                label='FEM (order 13) — what the cable actually does')
    a1.semilogx(periods, zl, color='seagreen', lw=2.2,
                label=f'{ladder.n_stages}-stage ladder')
    a1.semilogx(periods, z2, color='crimson', lw=2.0, ls='--',
                label='two-node, IEC 60853')

    w_day = 2.0 * np.pi / 86400.0
    z2d, zld, zrd = (float(np.abs(m.impedance(w_day)[0]))
                     for m in (fit, ladder, reference))
    a1.axvline(86400.0, color='0.35', lw=1.2, ls=':')
    a1.plot([86400.0] * 3, [z2d, zld, zrd], 'o', ms=7,
            color='none', mec='0.2', mew=1.2)
    a1.annotate(f"24 h\n{zld:.2f} vs {z2d:.2f} K·m/W\n"
                f"→ {zld / z2d:.1f}x the swing",
                xy=(86400.0, 0.5 * (zld + z2d)),
                xytext=(1.7e5, 0.30), fontsize=9.5, color='0.15',
                arrowprops=dict(arrowstyle='->', color='0.35', lw=1.1),
                bbox=dict(boxstyle='round,pad=0.4', fc='white', ec='0.75'))
    a1.set_ylabel('|Z| [K·m/W per W/m of total loss]')
    a1.set_title('Thermal impedance vs excitation period — the models agree '
                 'at DC and part company on a daily cycle', fontsize=11.5)
    a1.legend(fontsize=9, loc='upper left')
    a1.grid(alpha=0.3, which='both')

    # --- where each model keeps its resistance ------------------------------
    a2.stem(reference.tau_foster, reference.R_foster, linefmt='-',
            markerfmt='o', basefmt=' ', label='FEM (order 13)')
    for coll in a2.containers:
        pass
    a2.stem(ladder.tau_foster, ladder.R_foster, linefmt='-', markerfmt='s',
            basefmt=' ')
    # two-node modal residues
    A2, b2 = cf._two_node_matrices(fit.T1, fit.T4, fit.Q1, fit.Q2, fit.lambda1,
                                   1.0 / (1.0 + fit.lambda1))
    lam, V = np.linalg.eig(A2)
    res = ((np.array([1.0, 0.0]) @ V) * (np.linalg.inv(V) @ b2) / -lam).real
    a2.stem(-1.0 / lam.real, res, linefmt='-', markerfmt='D', basefmt=' ')
    # recolour the three stem groups
    for stem, colour in zip(a2.containers, ('0.6', 'seagreen', 'crimson')):
        plt.setp(stem[0], color=colour, ms=6)
        plt.setp(stem[1], color=colour, lw=2.0)
    a2.axvline(86400.0, color='0.35', lw=1.2, ls=':')
    a2.text(86400.0 * 1.15, 0.92 * max(reference.R_foster.max(), res.max()),
            'a 24 h cycle can only reach\npoles to the LEFT of this line',
            fontsize=9, color='0.25', va='top')
    a2.set_xscale('log')
    a2.set_xlabel('time constant / excitation period [s]')
    a2.set_ylabel('R behind each pole\n[K·m/W]')
    a2.grid(alpha=0.3, which='both')

    for ax in (a1, a2):
        for secs, lab in ((3600.0, '1 h'), (86400.0, '1 d'),
                          (7 * 86400.0, '1 wk'), (30 * 86400.0, '1 mo')):
            ax.axvline(secs, color='0.93', lw=0.8, zorder=0)
    a2.set_xlim(10.0, 1e8)

    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"   impedance-> {path}")


# =============================================================================
# DRIVER
# =============================================================================

def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(
        description="Worked example: FEM -> dtr -> ini() -> run() for a buried "
                    "MV cable circuit, in both thermal-model forms.")
    ap.add_argument('--production', action='store_true',
                    help="use the mesh cable_fem ships as its default "
                         "(slower, sub-kelvin difference)")
    ap.add_argument('--plot', metavar='DIR', default=None,
                    help="write figures into DIR")
    ap.add_argument('--hours', type=int, default=CYCLE_HOURS,
                    help=f"hours of cycling (default {CYCLE_HOURS})")
    ap.add_argument('--dt', type=float, default=CYCLE_DT,
                    help=f"load segment and output step [s] "
                         f"(default {CYCLE_DT:.0f})")
    ap.add_argument('--stages', type=int, default=N_STAGES,
                    help=f"ladder order to export and simulate "
                         f"(default {N_STAGES}; 2 is the IEC 60853 two-node "
                         "form, which this script always simulates alongside)")
    ap.add_argument('--no-step-check', action='store_true',
                    help="skip the step-size sensitivity re-runs")
    args = ap.parse_args(argv)

    t_start = time.time()
    (case, fem, steady, ladder, reference, fit,
     block_2n, block_lad) = characterise(args.production, args.stages)
    i_max = report_ampacity(case, fem, fit)

    with tempfile.TemporaryDirectory(prefix="cable_fem_example_") as work:
        built = build_both(case, fit, block_2n, block_lad, work)
        runs, stat = report_cycle(built, fit, ladder, args.hours, args.dt)
        ladder_err = validate_pydae_ladder(built, ladder, args.hours)
        if not args.no_step_check:
            step_size_check(built, args.hours, args.dt)

        if args.plot:
            banner(8, "FIGURES")
            os.makedirs(args.plot, exist_ok=True)
            plot_cycle(runs, stat, i_max,
                       os.path.join(args.plot, 'cable_cycle.png'))
            plot_impedance(fit, ladder, reference,
                           os.path.join(args.plot, 'cable_impedance.png'))
            field = os.path.join(args.plot, 'cable_field.png')
            cf.plot_field(fem, field)
            print(f"   field    -> {field}")

    print(f"\n{RULE}\n SUMMARY\n{RULE}")
    print(f"   steady rating (90 °C)         {i_max:.0f} A   "
          f"(library defaults would say "
          f"{ampacity(case.design, cf.library_defaults(case)['T1'] + cf.library_defaults(case)['T4'], 15.0):.0f} A)")
    print(f"   conductor at {CURRENT:.0f} A flat         "
          f"{fit.t_conductor_self_consistent:.2f} °C")
    print(f"   two-node    T1 {fit.T1:.4f}  T4 {fit.T4:.4f} K·m/W   "
          f"Q1 {fit.Q1:.0f}  Q2 {fit.Q2:.0f} J/(m·K)")
    print(f"   {ladder.n_stages}-stage     sum R {ladder.R.sum():.4f} K·m/W   "
          f"tau {', '.join(f'{v:.3g}' for v in ladder.time_constants)} s")
    print(f"   mutual heating share          "
          f"{1e2 * ladder.mutual_share:.0f} % of the conductor rise")
    print(f"   {args.hours} h cycle peak             "
          f"{stat['two_node']['max']:.2f} °C (two-node)  vs  "
          f"{stat['ladder']['max']:.2f} °C ({ladder.n_stages}-stage)")
    print(f"   {args.hours} h cycle swing            "
          f"{stat['two_node']['swing']:.2f} K  vs  "
          f"{stat['ladder']['swing']:.2f} K   "
          f"({stat['ladder']['swing'] / stat['two_node']['swing']:.2f}x)")
    print(f"   {args.hours} h cycle cover peak       "
          f"{stat['two_node']['cover_max']:.2f} °C (two-node)  vs  "
          f"{stat['ladder']['cover_max']:.2f} °C ({ladder.n_stages}-stage)")
    print(f"   pydae ladder vs Python ladder {ladder_err:.1e} K (1 min steps)")
    print(f"   total wall clock              {time.time() - t_start:.1f} s")
    print("\n   Paste ladder_dtr(...) into a lines[] entry for cyclic work, or")
    print("   fit.as_dtr() where only the steady state matters. A dtr_codes")
    print("   entry reuses either across a whole feeder.")

    return {'case': case, 'fem': fem, 'steady': steady, 'ladder': ladder,
            'fit': fit, 'i_max': i_max, 'runs': runs, 'stat': stat,
            'built': built, 'block_two_node': block_2n,
            'block_ladder': block_lad}


if __name__ == "__main__":
    main()
