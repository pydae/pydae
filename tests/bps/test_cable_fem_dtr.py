# tests/bps/test_cable_fem_dtr.py
"""
End-to-end test of the FEM -> DTR -> ini() -> run() workflow for underground
cables.

The chain under test, all four links in one file:

    1. pydae.bps.lines.cable_fem   builds the cable geometry, materials and
                                   buried layout, and solves the 2D thermal
                                   field with scikit-fem;
    2.                             reduces the field to a Cauer RC ladder and
                                   then to the two-node IEC 60853 form;
    3. pydae.bps.lines.line_dtr_cable   receives those parameters as a "dtr"
                                   block on a lines[] entry;
    4. pydae.bps + CasadiModel     builds the system, ini()s it and run()s a
                                   transient.

What makes this more than a smoke test is that every link is checked against
something that does NOT come from the link itself:

  * the FEM steady field against the Kennelly image method and the analytic
    layer-stack log formula (no mesh, no assembly involved);
  * the Cauer ladder against a direct backward-Euler transient FEM solve
    (shares no machinery with the Krylov/Cauer path);
  * the exported block against the parameter contract that
    ``line_dtr_cable.descriptions()`` publishes, so renaming a parameter on
    either side fails here rather than silently producing a cold cable;
  * the ini() steady state against the closed-form two-node relation;
  * the run() trajectory against an independent scipy integration of the same
    two ODEs.

The CasADi backend is used throughout, so no C toolchain is needed.

Run selectively::

    uv run pytest -m bps                                   # all bps tests
    uv run pytest tests/bps/test_cable_fem_dtr.py -v
    uv run pytest tests/bps/test_cable_fem_dtr.py -m "not slow"
    uv run pytest tests/bps/test_cable_fem_dtr.py -k ini
"""

import json
import math
import os

import numpy as np
import pytest


# The FEM stages need scikit-fem/scipy and the simulation stages need CasADi;
# neither is a hard dependency of pydae-bps (cable_fem imports them lazily), so
# the whole module skips rather than fails when they are absent.
pytest.importorskip("skfem", reason='cable_fem needs scikit-fem: '
                                    'pip install "pydae-bps[fem]"')
pytest.importorskip("scipy", reason="cable_fem needs scipy")
pytest.importorskip("casadi", reason="the CasADi backend needs casadi")

from pydae.bps.lines import cable_fem as cf            # noqa: E402
from pydae.bps.lines import line_dtr_cable             # noqa: E402


# =============================================================================
# Test-grade settings
# =============================================================================
# The production mesh (1.15e6 elements) takes ~15 s per solve, and the ladder
# reduction factorises the stiffness matrix a dozen times over. Dropping the
# outermost refinement pass and keeping a single in-cable pass gives ~1.6e5
# elements, which solves in ~1.5 s and still reproduces the analytic T4 to
# better than 0.1 %.
#
# The in-cable pass is not optional here: the theta_e cross-check samples a
# 0.6 mm-wide annulus, and without it that sample is 9 % low -- which is
# exactly the numerical fragility that makes cable_fem take T1 analytically in
# the first place.
#
# Every assertion below is either mesh-INDEPENDENT (an exact algebraic
# identity) or carries a tolerance sized for this mesh. The tight numbers live
# in cable_fem.test() and in the production run.
REFINE_FACTORS = (20.0, 11.0, 6.5, 4.0, 2.6)
IN_CABLE_PASSES = 1

#: Current the circuit is characterised and simulated at [A].
TEST_CURRENT = 200.0

#: Power factor used to size the test load.
TEST_PF = 0.97

#: Daily-cycle settings. 24 h at quarter-hour steps: one day is representative
#: of a settled cycle for every stage faster than the soil, and the run costs
#: about 0.05 s.
CYCLE_HOURS = 24
CYCLE_DT = 900.0
CYCLE_SWING = 0.30
CYCLE_PEAK_HOUR = 18.0

#: ``line_dtr_cable``'s own default isolated-cable T4 (dc = 12.4 mm,
#: De = 34.2 mm, L_depth = 0.8 m, rho_soil = 1.0 K.m/W) -- what the 'geometry'
#: export mode falls back to when the block pins no dimensions. Taken from the
#: library rather than retyped, so it cannot drift.
DEFAULT_ISOLATED_T4 = line_dtr_cable._cable_thermal_network({})['T4']


# =============================================================================
# Fixtures
# =============================================================================

class Characterisation:
    """Everything stages 1-3 produce, computed once for the whole module."""

    def __init__(self):
        self.case = cf.cigre_mv_eu_case(TEST_CURRENT)
        self.fem = cf.ThermalFEM(self.case, in_cable_passes=IN_CABLE_PASSES,
                                 refine_factors=REFINE_FACTORS)
        self.steady = self.fem.steady_metrics()
        self.ladder = cf.identify_ladder(self.fem, n_stages=5)
        self.reference = cf.identify_ladder(
            self.fem, n_stages=cf.REFERENCE_ORDER,
            tau_range=(10 ** 0.5, 10 ** 7.5))
        self.fit = cf.fit_two_node(self.fem, steady=self.steady,
                                   reference=self.reference)


@pytest.fixture(scope="module")
def char():
    """The FEM characterisation. A few seconds, shared by every test below."""
    return Characterisation()


def _build_and_tune(fit, case, work_dir, dtr_mode="fitted", name=None,
                    current=TEST_CURRENT, dtr=None):
    """
    Build the pydae-bps model from an exported ``dtr`` block and trim the load
    until the line current is *exactly* the current the cable was
    characterised at.

    Trimming matters: the conductor rise goes as I-squared, so comparing the
    model with the FEM at whatever current happens to fall out of the load flow
    would confound a parameter-transport error with a 2 % current mismatch. The
    load is an *input*, so each trim costs one ini() and not a rebuild.

    :returns: ``(model, tuned_load)`` with ``tuned_load`` the ``{'P_2','Q_2'}``
        dict that puts the line on *current*.
    """
    from pydae.bps import BpsBuilder
    from pydae.core.builder.casadi_builder import CasadiBuilder
    from pydae.core.model.casadi_model import CasadiModel

    name = name or f"cable_fem_dtr_{dtr_mode}"
    s_target = math.sqrt(3.0) * 20e3 * current
    p = s_target * TEST_PF
    q = s_target * math.sqrt(1.0 - TEST_PF ** 2)
    data = cf.bps_data(case, fit, name=name, mode=dtr_mode, dtr=dtr,
                       P_load_W=p, Q_load_var=q)

    # construct() drops a "<name>_xy_0.json" in the cwd, so keep it in the
    # test's temporary directory rather than the repository root.
    cwd = os.getcwd()
    os.chdir(work_dir)
    try:
        grid = BpsBuilder(data, use_casadi=True)
        grid.construct(name)
        # Tight integrator tolerances. This file compares pydae's trajectory
        # against an independent scipy integration of the same ODEs, and at
        # IDAS's defaults (reltol 1e-6, abstol 1e-5) the accumulated
        # integration error over 24 h is ~7e-3 K -- enough that the comparison
        # would measure the integrator rather than the equations. The models
        # here are tiny, so the tighter tolerance is free.
        model = CasadiModel(CasadiBuilder(grid.sys_dict).build(),
                            integrator_reltol=1e-10, integrator_abstol=1e-10)
        i_line = float('nan')
        for _ in range(12):
            model.ini({'P_2': p, 'Q_2': q}, newton_tol=1e-10)
            i_line = float(model.get_value('I_cond_1_2'))
            if abs(i_line - current) < 1e-6:
                break
            scale = current / i_line       # the current tracks the load
            p *= scale
            q *= scale
    finally:
        os.chdir(cwd)

    assert abs(i_line - current) < 1e-4, (
        f"could not trim the load to {current} A; stalled at {i_line:.4f} A")
    return model, {'P_2': p, 'Q_2': q}


@pytest.fixture(scope="module")
def built(char, tmp_path_factory):
    """The fitted-block model, built once and trimmed to TEST_CURRENT."""
    work = tmp_path_factory.mktemp("cable_fem_fitted")
    model, load = _build_and_tune(char.fit, char.case, work, "fitted")
    return {'model': model, 'load': load, 'dir': work}


@pytest.fixture
def model(built):
    """
    The built model, re-initialised for each test that simulates.

    ``ini()`` resets ``t``, the ``Time/X/Y/Z`` storage and the run inputs, so
    every simulation test starts from the same converged steady state instead
    of wherever the previous test's ``run()`` left the cable. Sharing the
    *build* is what makes this affordable; sharing the *state* would make the
    tests order-dependent.
    """
    m = built['model']
    m.ini(dict(built['load']), newton_tol=1e-10)
    return m


@pytest.fixture(scope="module")
def built_ladder(char, tmp_path_factory):
    """The n-stage model, built once from the same characterisation."""
    work = tmp_path_factory.mktemp("cable_fem_ladder")
    block = cf.ladder_dtr(char.ladder, char.fit)
    model, load = _build_and_tune(char.fit, char.case, work, "ladder",
                                  name="cable_fem_dtr_ladder", dtr=block)
    return {'model': model, 'load': load, 'block': block, 'dir': work}


@pytest.fixture
def ladder_model(built_ladder):
    """The n-stage model, re-initialised for each test that simulates."""
    m = built_ladder['model']
    m.ini(dict(built_ladder['load']), newton_tol=1e-10)
    return m


def _cycle(model, load, hours=CYCLE_HOURS, dt=CYCLE_DT):
    """
    Drive a daily load cycle through a model and return the conductor series.

    Large steps on purpose: the fastest ladder stage has a ~20 s time constant
    and carries 4 % of the thermal resistance, so quarter-hour load resolution
    costs a few hundredths of a kelvin (measured in the example's
    ``step_size_check``) and runs 24 h in well under a tenth of a second.
    """
    model.ini(dict(load), newton_tol=1e-10)
    model.Dt = dt
    n_seg = int(round(hours * 3600.0 / dt))
    for k in range(n_seg):
        hour = (k + 0.5) * dt / 3600.0
        frac = 1.0 + CYCLE_SWING * math.cos(
            2.0 * math.pi * (hour - CYCLE_PEAK_HOUR) / 24.0)
        model.run((k + 1) * dt,
                  {'P_2': load['P_2'] * frac, 'Q_2': load['Q_2'] * frac})
    model.post()
    return np.array(model.get_values('t_conductor_1_2'), dtype=float)


def _dtr_values(model):
    """Read the DTR parameters back out of the built model."""
    keys = ('T1', 'T4', 'Q1', 'Q2', 'p_vw', 'lambda1', 'Wd', 'R_ac_20',
            'alpha_R', 'K_dtr', 't_soil')
    return {k: float(model.get_value(f'{k}_1_2')) for k in keys}


# =============================================================================
# 1. GEOMETRY AND MATERIALS  (no FEM -- fast)
# =============================================================================

@pytest.mark.bps
def test_geometry_and_loading_are_the_benchmark_cable():
    """
    Stage 1: the default case really is the CIGRE TB 575 cable, and the loss
    split is the one a layered thermal model needs.
    """
    case = cf.cigre_mv_eu_case(TEST_CURRENT)
    d = case.design

    # Table 6.11 dimensions must be recoverable from the resolved layer stack.
    assert 2.0 * d.r_outer == pytest.approx(34.2e-3, abs=5e-5)
    assert 2.0 * d.r_conductor == pytest.approx(12.4e-3, abs=1e-6)
    assert math.pi * d.r_conductor ** 2 * 1e6 == pytest.approx(120.0, abs=1.5)

    # Flat formation at a = 0.7 m, b = 0.3 m (Fig. 6.6 / Table 6.9).
    c = case.centers
    assert c.shape == (3, 2)
    assert np.allclose(c[:, 1], -0.7)
    assert np.linalg.norm(c[1] - c[0]) == pytest.approx(0.3)
    assert case.hot_index == 1, "the centre cable is the rating point"

    # Only the CONDUCTOR loss crosses T1, so the split has to be explicit.
    w = case.losses()
    assert w['W_c'] == pytest.approx(TEST_CURRENT ** 2 * 0.338e-3)
    assert w['W_total'] == pytest.approx(TEST_CURRENT ** 2 * 0.501e-3)
    assert w['lambda1'] == pytest.approx(0.501 / 0.338 - 1.0)
    assert w['W_s'] > 0.0, "screen losses must not be dropped"


# =============================================================================
# 2. FEM  ->  checked against closed-form results
# =============================================================================

@pytest.mark.bps
@pytest.mark.slow
def test_fem_steady_field_matches_closed_form(char):
    """
    Stage 2: the steady field conserves energy, orders itself physically, and
    agrees with two independent analytic results that share no code with the
    FEM -- the layer-stack log formula for T1 and the Kennelly image method for
    T4. Passing this validates the mesh, the material mapping, the boundary
    conditions and the loss placement at once.
    """
    st, fem, case = char.steady, char.fem, char.case
    hot = st.hot_index

    assert abs(st.energy_balance) < 1e-8, "global energy balance broken"
    assert abs(st.source_error) < 2e-3, "the loss bands are under-resolved"

    # Physical ordering, outward and across the row.
    assert st.t_conductor[hot] > st.t_node_e > st.t_soil
    others = [i for i in range(case.n_cables) if i != hot]
    for i in others:
        assert st.t_conductor[hot] > st.t_conductor[i]
    assert st.t_conductor[others[0]] == pytest.approx(
        st.t_conductor[others[1]], abs=0.05), "the mesh is not symmetric"

    # t_soil is NOT the deep-earth boundary value: it carries the
    # surface-convection gradient down to burial depth. That is exactly why
    # line_dtr_cable's fixed 15 degC default reads cold.
    assert case.installation.t_deep < st.t_soil < case.installation.t_air

    # T1: analytic log formula vs sampling the FEM at the theta_e radius. The
    # sample is the mesh-sensitive one -- it averages over a 0.6 mm annulus --
    # so the tolerance is loose on purpose and tightens with refinement
    # (-4.5 % on this mesh, -1.3 % on the production mesh).
    t1_analytic = fem.analytic_T1()
    t1_sampled = (st.t_conductor[hot] - st.t_node_e) / st.losses['W_c']
    assert t1_sampled == pytest.approx(t1_analytic, rel=0.06)

    # T4: Kennelly images plus the outer layer stack.
    a4 = fem.analytic_T4()
    t4_fem = st.z_hot - t1_analytic / (1.0 + st.losses['lambda1'])
    assert t4_fem == pytest.approx(a4['total'], rel=0.05)
    assert a4['soil_mutual'] > 0.3 * a4['soil_self'], (
        "mutual heating should be a large fraction of the self term at "
        "b = 0.3 m spacing")


@pytest.mark.bps
@pytest.mark.slow
def test_kennelly_diagonal_is_the_library_formula(char):
    """
    The claim the whole module rests on: ``line_dtr_cable``'s geometry-derived
    ``T4`` is the Kennelly result for an ISOLATED cable, i.e. the *diagonal* of
    the mutual matrix, and the off-diagonal heating is what it omits.
    """
    case = char.case
    lib = cf.library_defaults(case)
    Z = case.kennelly_matrix()
    hot = case.hot_index

    assert np.allclose(Z, Z.T), "the mutual matrix must be symmetric"
    assert Z[hot, hot] == pytest.approx(lib['T4'], rel=1e-3)
    assert Z[hot].sum() - Z[hot, hot] > 0.0, "no mutual heating found"
    assert char.fit.T4 > lib['T4'], (
        "the fitted T4 must exceed the isolated-cable value -- it carries the "
        "mutual heating of the other two phases")


# =============================================================================
# 3. LADDER  ->  checked against a direct transient FEM solve
# =============================================================================

@pytest.mark.bps
@pytest.mark.slow
def test_ladder_is_passive_and_reproduces_the_fem_transient(char):
    """
    Stage 3: the identified Cauer ladder is realisable with strictly positive
    elements, is the same system as the Foster form it was extracted from, has
    an exact DC resistance, and tracks a backward-Euler transient FEM solve
    that shares none of its machinery.
    """
    lad, fem = char.ladder, char.fem

    assert lad.passive, "no passive ladder was realised for the hot spot"
    assert np.all(lad.R > 0.0) and np.all(lad.C > 0.0)

    # sum(R) is the DC thermal resistance, straight out of (A, b, c).
    A, b, c = lad.matrices()
    assert float(-c @ np.linalg.solve(A, b)) == pytest.approx(lad.R.sum(),
                                                              abs=1e-8)
    assert lad.dc_resistance == pytest.approx(lad.R.sum(), rel=1e-9)

    # The circuit evaluation and the state space must be the same system.
    t = np.logspace(0.0, 8.0, 60)
    assert np.max(np.abs(lad.step(t) - lad.foster_step(t))) < 1e-6

    # Independent ground truth.
    t_chk = np.array([1e3, 1e5, 1e7])
    z_fem = cf.transient_fem_step(fem, t_chk)
    assert np.max(np.abs(lad.step(t_chk) - z_fem)) < 0.02

    # Time constants must span the insulation-to-soil range, and the balanced
    # collapse must put the centre cable above its neighbours.
    tau = lad.time_constants
    assert tau[0] > 1e5 and tau[-1] < 1e4
    assert lad.z_balanced > lad.z_outer > 0.0
    assert 0.1 < lad.mutual_share < 0.6


# =============================================================================
# 4a. THE EXPORTED BLOCK  ->  checked against line_dtr_cable's own contract
# =============================================================================

@pytest.mark.bps
@pytest.mark.slow
def test_dtr_block_matches_the_line_dtr_cable_contract(char):
    """
    The exported block must speak exactly the vocabulary
    ``line_dtr_cable.descriptions()`` publishes -- no invented keys, and
    nothing the model needs left out.

    This is the seam where a silent failure would hurt most: an unrecognised
    key is simply ignored when the ``dtr`` data is resolved, so a renamed
    parameter would leave the cable running on library defaults and reading
    ~20 K cold, with no error raised anywhere.
    """
    fit = char.fit
    dtr = fit.as_dtr()

    settable = {d['data'] for d in line_dtr_cable.descriptions() if d['data']}

    # Nothing invented.
    emitted = set(dtr) - {'model'}
    assert emitted <= settable, (
        "the block emits keys line_dtr_cable does not read: "
        f"{sorted(emitted - settable)}")

    # The two-node block does not carry R/C -- those select the other form.
    assert not {'R', 'C'} & emitted

    # Nothing missing: the two exported blocks plus the geometry mapping must
    # between them cover every settable name. 'soil_shell' is the documented
    # exception -- it is only a shape factor for the ANALYTIC Q2, which both
    # exported forms supersede.
    ladder_block = cf.ladder_dtr(char.ladder, fit)
    covered = (emitted | set(ladder_block) | set(cf.geometry_dtr(char.case)))
    covered -= {'model'}
    assert settable - covered == {'soil_shell'}, (
        f"uncovered line_dtr_cable parameters: {sorted(settable - covered)}")

    # The two forms are alternatives, so their network parameters must be
    # disjoint -- that disjointness is how line_dtr_cable dispatches.
    assert set(ladder_block) - {'model'} <= settable
    assert {'R', 'C'} <= set(ladder_block)
    assert not {'T1', 'T4', 'Q1', 'Q2', 'p'} & set(ladder_block)

    # The block must be routable and serialisable exactly as written.
    assert dtr['model'] == 'cable'
    assert json.loads(json.dumps(dtr)) == dtr

    # Physically admissible values.
    assert dtr['T1'] > 0.0 and dtr['T4'] > 0.0
    assert dtr['Q1'] > 0.0 and dtr['Q2'] > dtr['Q1']
    assert 0.0 < dtr['p'] <= 0.5, "Van Wormer factor out of range"
    assert dtr['lambda1'] > 0.0
    assert dtr['t_soil'] > 0.0

    # 'geometry' mode must pin nothing, so the library derives the network.
    geo = fit.as_dtr(mode='geometry')
    assert not {'T1', 'T4', 'Q1', 'Q2', 'p'} & set(geo)
    assert geo['model'] == 'cable'


@pytest.mark.bps
@pytest.mark.slow
def test_two_node_fit_keeps_the_steady_state_exact(char):
    """
    ``T1`` and ``T4`` are pinned by the FEM steady state, never fitted, so the
    two-node steady conductor temperature must reproduce the FEM to round-off.
    Only the transient pays for the truncation.
    """
    fit, st = char.fit, char.steady
    hot = st.hot_index

    assert fit.steady_conductor(st.losses['W_c']) == pytest.approx(
        st.t_conductor[hot], abs=1e-6)
    assert (fit.T1 / (1.0 + fit.lambda1) + fit.T4) == pytest.approx(
        st.z_hot, abs=1e-9)

    # The truncation cost is real and large -- roughly 20 % of the rise -- and
    # is reported rather than asserted away. This bound only catches a fit
    # that has actually broken.
    rise = st.t_conductor[hot] - st.t_soil
    assert 0.0 < fit.err_kelvin < 0.45 * rise

    # Two well-separated time constants: insulation-ish and soil-ish.
    tau = fit.time_constants
    assert tau[0] > 1e4 > tau[1] > 0.0


# =============================================================================
# 4b. ini()  ->  checked against the closed-form two-node steady state
# =============================================================================

@pytest.mark.bps
@pytest.mark.model
@pytest.mark.slow
def test_ini_reproduces_the_fem_steady_state(char, model):
    """
    The pydae-bps model, built from the exported block and trimmed to the
    characterisation current, must land on the FEM conductor temperature.

    Three separate things are checked, because they fail for different
    reasons: parameter *transport* (did the block arrive intact?), internal
    *consistency* (does the model solve its own two-node relation?), and
    *meaning* (does that relation agree with the FEM?).
    """
    fit = char.fit
    dtr = fit.as_dtr()
    p = _dtr_values(model)

    # --- transport: every exported number arrived, under the model's name ---
    for block_key, model_key in (('T1', 'T1'), ('T4', 'T4'), ('Q1', 'Q1'),
                                 ('Q2', 'Q2'), ('p', 'p_vw'),
                                 ('lambda1', 'lambda1'), ('Wd', 'Wd'),
                                 ('R_ac_20', 'R_ac_20'),
                                 ('alpha_R', 'alpha_R'), ('K_dtr', 'K_dtr'),
                                 ('t_soil', 't_soil')):
        assert p[model_key] == pytest.approx(dtr[block_key], rel=1e-12), (
            f"{block_key} did not survive the trip into the model")

    # --- consistency: the model solves its own two-node steady relation ----
    t_c = float(model.get_value('t_conductor_1_2'))
    t_e = float(model.get_value('t_sheath_1_2'))
    w_c = float(model.get_value('W_c_1_2'))
    expect_c = p['t_soil'] + w_c * p['T1'] + w_c * (1.0 + p['lambda1']) * p['T4']
    expect_e = p['t_soil'] + w_c * (1.0 + p['lambda1']) * p['T4']
    assert t_c == pytest.approx(expect_c, abs=1e-6)
    assert t_e == pytest.approx(expect_e, abs=1e-6)
    assert t_c > t_e > p['t_soil'], "temperatures not monotonic outward"

    # --- the losses are the ones the cable was characterised with ----------
    i_cond = float(model.get_value('I_cond_1_2'))
    assert i_cond == pytest.approx(TEST_CURRENT, abs=1e-4)
    r_ac = float(model.get_value('R_ac_1_2'))
    assert r_ac == pytest.approx(
        p['R_ac_20'] * (1.0 + p['alpha_R'] * (t_c - 20.0)), rel=1e-10)
    assert w_c == pytest.approx(i_cond ** 2 * r_ac, rel=1e-10)
    assert float(model.get_value('W_total_1_2')) == pytest.approx(
        w_c * (1.0 + p['lambda1']) + p['Wd'], rel=1e-10)

    # --- meaning: the model agrees with the FEM at the same current --------
    # The FEM-derived value re-evaluates R_ac at the converged temperature --
    # the same fixed point the model solves. What is left is the rounding of
    # the block as written (T1/T4 to 6 decimals, t_soil to 2, lambda1 to 4).
    assert t_c == pytest.approx(fit.t_conductor_self_consistent, abs=0.02), (
        f"model {t_c:.4f} degC vs FEM-derived "
        f"{fit.t_conductor_self_consistent:.4f} degC")


@pytest.mark.bps
@pytest.mark.model
@pytest.mark.slow
def test_geometry_mode_block_builds_and_reads_cold(char, tmp_path):
    """
    The 'geometry' export mode must also build and initialise -- and it must
    come out substantially COOLER than the fitted block, because the library
    then derives ``T4`` for an isolated cable at its own default burial depth.

    This is the module's central claim, measured end to end through the real
    simulator rather than argued about on paper. (Observed gap on the
    test-grade mesh: ~11 K; the assertion leaves margin.)
    """
    geo_model, _load = _build_and_tune(char.fit, char.case, tmp_path,
                                       dtr_mode="geometry",
                                       name="cable_fem_dtr_geometry")
    t_c_geo = float(geo_model.get_value('t_conductor_1_2'))
    t_soil = float(geo_model.get_value('t_soil_1_2'))
    t_c_fit = char.fit.t_conductor_self_consistent

    assert float(geo_model.get_value('I_cond_1_2')) == pytest.approx(
        TEST_CURRENT, abs=1e-4)
    assert t_c_geo > t_soil, "the cable should still warm up"
    assert t_c_geo < t_c_fit - 8.0, (
        f"the geometry route gives {t_c_geo:.2f} degC vs the FEM-derived "
        f"{t_c_fit:.2f} degC; a gap of this size is the whole reason this "
        "module exists")

    # Geometry mode leaves line_dtr_cable to derive the network from its OWN
    # default dimensions -- it is not quietly being fed cable_fem's layer stack.
    assert float(geo_model.get_value('T4_1_2')) == pytest.approx(
        DEFAULT_ISOLATED_T4, rel=1e-9)
    assert DEFAULT_ISOLATED_T4 < char.fit.T4

    # lambda1 and t_soil DO come across even in geometry mode, so the gap
    # above is attributable to the thermal network alone.
    assert float(geo_model.get_value('lambda1_1_2')) == pytest.approx(
        char.fit.as_dtr(mode='geometry')['lambda1'])
    assert t_soil == pytest.approx(char.fit.as_dtr(mode='geometry')['t_soil'])


# =============================================================================
# 4c. run()  ->  checked against an independent integration of the same ODEs
# =============================================================================

@pytest.mark.bps
@pytest.mark.model
@pytest.mark.slow
def test_run_matches_an_independent_two_node_integration(model):
    """
    The transient: step the soil temperature and integrate the same two ODEs
    with scipy, from the model's own parameters and its own converged initial
    state. Agreement proves the equations pydae-bps is integrating are the ones
    the FEM was reduced to -- not merely that *something* warms up.

    The conductor current is NOT constant through the run, and the reference
    has to be driven with the model's own ``I_cond`` series because of it.
    pydae-bps runs the cable DTR closed-loop: ``add_line_dtr`` writes the line
    admittance as R_pu_20*(1 + alpha_R*(t_conductor - 20)) into the algebraic
    equations, so as the conductor heats, the series resistance rises, the
    voltage drop shifts and the current moves with it -- here by 0.06 A over
    24 h. Holding the current at its initial value instead leaves a 7e-3 K
    residual that has nothing to do with the thermal model.
    """
    from scipy.integrate import solve_ivp

    p = _dtr_values(model)
    t_c0 = float(model.get_value('t_conductor_1_2'))
    t_e0 = float(model.get_value('t_sheath_1_2'))
    t_soil_new = p['t_soil'] + 10.0
    t_end = CYCLE_HOURS * 3600.0

    # Run first, so the reference can be driven with the current the model
    # actually settled on at each instant.
    model.Dt = CYCLE_DT
    model.run(t_end, {'t_soil_1_2': t_soil_new})
    model.post()

    t = np.asarray(model.Time, dtype=float)
    y_model = np.asarray(model.get_values('t_conductor_1_2'), dtype=float)
    i_series = np.asarray(model.get_values('I_cond_1_2'), dtype=float)
    assert abs(i_series[-1] - i_series[0]) > 1e-3, (
        "the current did not move at all -- the temperature-dependent line "
        "admittance appears to have been bypassed")

    def rhs(tt, y):
        t_c, t_e = y
        i_cond = np.interp(tt, t, i_series)
        w_c = i_cond ** 2 * p['R_ac_20'] * (1.0 + p['alpha_R'] * (t_c - 20.0))
        return [(w_c + p['p_vw'] * p['Wd'] - (t_c - t_e) / p['T1']) / p['Q1'],
                ((t_c - t_e) / p['T1'] + (1.0 - p['p_vw']) * p['Wd']
                 + p['lambda1'] * w_c
                 - (t_e - t_soil_new) / p['T4']) / p['Q2']]

    sol = solve_ivp(rhs, (0.0, t_end), [t_c0, t_e0], rtol=1e-11, atol=1e-11,
                    dense_output=True)
    assert sol.success
    y_ref = sol.sol(t)[0]

    assert len(t) > 50, "the run produced almost no output points"
    err = float(np.max(np.abs(y_model - y_ref)))
    assert err < 1e-3, (
        f"pydae-bps and the independent integration disagree by {err:.2e} K")

    # The soil step must move the conductor, monotonically, and by less than
    # the step itself over 24 h: the external node holds ~1e5 J/(m.K), so it
    # cannot follow instantly.
    assert y_model[-1] > t_c0
    assert np.all(np.diff(y_model) > -1e-9), "the response is not monotonic"
    assert y_model[-1] - t_c0 < 10.0


@pytest.mark.bps
@pytest.mark.model
@pytest.mark.slow
def test_run_soil_step_reaches_the_analytic_steady_state(model):
    """
    Run long enough for the external node to saturate, and the conductor must
    land on the closed-form steady state at the new soil temperature.

    The fixed point is not a simple offset: ``R_ac`` rises with temperature, so
    the loss grows as the cable warms and the conductor settles above
    ``theta_soil + W_c(20 degC) * Z_eq``. Solving

        theta_c = theta_soil + I^2 R_ac_20 (1 + alpha (theta_c - 20)) Z_eq

    for theta_c and checking it at BOTH ends of the step makes this a test of
    the resistance-temperature feedback term rather than of the offset.

    Two details decide whether this comparison is exact or merely close, and
    both are worth spelling out because each is worth a few tenths of a kelvin:

    * the closed form is evaluated with the parameters *the model holds*, not
      the unrounded ones the fit produced -- the exported block rounds
      ``R_ac_20`` to four significant figures;
    * the conductor current is re-read after the run, because over 60 days the
      slack source drifts it by ~0.07 % and the rise goes as I-squared.
    """
    p = _dtr_values(model)
    z_eq = p['T1'] + (1.0 + p['lambda1']) * p['T4']
    t_soil_new = p['t_soil'] + 10.0

    def fixed_point(t_soil, current):
        """Self-consistent conductor temperature, from the model's own numbers."""
        loop = current ** 2 * p['R_ac_20'] * z_eq
        assert loop * p['alpha_R'] < 1.0, "thermal runaway: no steady state"
        return ((t_soil + loop * (1.0 - 20.0 * p['alpha_R']))
                / (1.0 - loop * p['alpha_R']))

    t_c0 = float(model.get_value('t_conductor_1_2'))
    i_0 = float(model.get_value('I_cond_1_2'))
    assert t_c0 == pytest.approx(fixed_point(p['t_soil'], i_0), abs=1e-6), (
        "ini() did not land on the analytic fixed point")

    # 60 days is set by the physics -- the slow two-node pole is ~1.6 d, so
    # this is ~37 time constants. The STEP can be enormous: one output point
    # per day still lands on the fixed point, and costs 60 steps instead of
    # 1440.
    model.Dt = 86400.0
    model.run(60.0 * 86400.0, {'t_soil_1_2': t_soil_new})
    model.post()
    t_c = float(model.get_value('t_conductor_1_2'))
    i_1 = float(model.get_value('I_cond_1_2'))

    assert t_c == pytest.approx(fixed_point(t_soil_new, i_1), abs=1e-6), (
        f"after 60 days the conductor sits at {t_c:.6f} degC; the analytic "
        f"steady state at {i_1:.4f} A is "
        f"{fixed_point(t_soil_new, i_1):.6f} degC")

    # The feedback must AMPLIFY the step: more than 1 K of conductor rise per
    # kelvin of soil rise. The loop gain says by how much.
    loop = i_1 ** 2 * p['R_ac_20'] * z_eq * p['alpha_R']
    assert 0.0 < loop < 1.0, "loop gain outside the stable range"
    gain = (t_c - t_c0) / 10.0
    assert 1.0 < gain < 1.0 / (1.0 - loop) + 0.01, (
        f"soil-step gain {gain:.5f} K/K is not consistent with a loop gain of "
        f"{loop:.5f}")


# =============================================================================
# 4d. THE n-STAGE LADDER FORM
# =============================================================================

@pytest.mark.bps
@pytest.mark.model
@pytest.mark.slow
def test_ladder_ini_reproduces_sum_r(char, ladder_model, built_ladder):
    """
    The n-stage model must solve its own steady state exactly: every node sits
    above the soil by the thermal resistance still downstream of it, driven by
    the TOTAL loss. ``sum(R)`` is therefore the DC conductor-to-soil resistance.
    """
    model, block = ladder_model, built_ladder['block']
    R = block['R']
    n = len(R)

    states = [s for s in model.x_names if s.endswith('_1_2')]
    assert len(states) == n, f"expected {n} thermal states, got {states}"
    assert 't_conductor_1_2' in states, \
        "node 1 must be t_conductor -- the line admittance is written on it"
    assert 't_sheath_1_2' not in states, \
        "the ladder form has no distinct sheath node"

    g = lambda k: float(model.get_value(f'{k}_1_2'))
    nodes = [g('t_conductor')] + [g(f't_ladder_{i}') for i in range(2, n + 1)]
    t_soil, w_total = g('t_soil'), g('W_total')

    for i, theta in enumerate(nodes):
        assert theta == pytest.approx(t_soil + w_total * sum(R[i:]), abs=1e-6)
    assert all(a > b for a, b in zip(nodes, nodes[1:])), "not monotonic outward"
    assert nodes[-1] > t_soil

    # Each R/C pair must have arrived, in order.
    for i, (r, c) in enumerate(zip(R, block['C']), 1):
        assert g(f'R{i}') == pytest.approx(r, rel=1e-12)
        assert g(f'C{i}') == pytest.approx(c, rel=1e-12)

    # Same DC resistance as the two-node block, hence the same steady state.
    z_ladder = (1.0 + g('lambda1')) * sum(R)
    assert z_ladder == pytest.approx(char.fit.z_equivalent, rel=2e-3)
    assert nodes[0] == pytest.approx(char.fit.t_conductor_self_consistent,
                                     abs=0.05)


@pytest.mark.bps
@pytest.mark.model
@pytest.mark.slow
def test_ladder_recovers_the_daily_swing(char, built, built_ladder):
    """
    The reason the n-stage form exists: on a 24 h cycle the two-node
    truncation flattens the conductor swing by about 2x, and therefore the
    peak, which is the unsafe direction for a rating study.

    Both models start from the same steady state and see the same current, so
    the difference measured here is the truncation alone.
    """
    y2 = _cycle(built['model'], built['load'])
    yl = _cycle(built_ladder['model'], built_ladder['load'])

    swing_2n, swing_lad = float(np.ptp(y2)), float(np.ptp(yl))
    peak_2n, peak_lad = float(y2.max()), float(yl.max())

    # They must agree at the start -- same DC resistance, same initial steady
    # state -- so any divergence is transient.
    assert y2[0] == pytest.approx(yl[0], abs=0.05)

    # The ladder swings substantially more, and peaks higher.
    assert swing_lad > 1.5 * swing_2n, (
        f"ladder swing {swing_lad:.2f} K vs two-node {swing_2n:.2f} K; the "
        "whole point of the n-stage form is that it does not flatten the cycle")
    assert peak_lad > peak_2n + 2.0, (
        f"ladder peak {peak_lad:.2f} degC vs two-node {peak_2n:.2f} degC")

    # Both stay physical: above the soil, below the XLPE limit at this loading.
    t_soil = float(built_ladder['model'].get_value('t_soil_1_2'))
    for y in (y2, yl):
        assert y.min() > t_soil
        assert y.max() < 90.0

    # The peak must lag the 18:00 current peak -- the outer stages have time
    # constants of hours to days, so the cable cannot track the load.
    t = np.array(built_ladder['model'].Time, dtype=float)
    peak_hour = float(t[int(np.argmax(yl))] / 3600.0)
    assert CYCLE_PEAK_HOUR < peak_hour < CYCLE_PEAK_HOUR + 4.0, \
        f"thermal peak at {peak_hour:.1f} h for an {CYCLE_PEAK_HOUR:.0f} h " \
        "current peak"


@pytest.mark.bps
@pytest.mark.model
@pytest.mark.slow
def test_ladder_matches_an_independent_integration(built_ladder, char):
    """
    Confirm pydae-bps is integrating the ladder ``cable_fem`` identified, and
    not some transposition of it: ``_dtr_cable_ladder`` rebuilds the network
    symbolically from the R/C lists, where a sign or index slip would give a
    plausible-looking but wrong trajectory.

    The Python reference holds the loss constant over each output interval
    while pydae keeps updating it with the conductor temperature, so the two
    cannot agree exactly at any one step size. Running at two step sizes
    separates that from a real error: a zero-order-hold residual falls linearly
    with ``dt``, while a wrong network would sit on a floor.
    """
    ladder = char.ladder
    model, load = built_ladder['model'], built_ladder['load']
    A, b, _c = ladder.matrices()

    errs = []
    for dt in (CYCLE_DT, CYCLE_DT / 5.0):
        _cycle(model, load, CYCLE_HOURS, dt)
        t = np.array(model.Time, dtype=float)
        w = np.array(model.get_values('W_total_1_2'), dtype=float)
        y = np.array(model.get_values('t_conductor_1_2'), dtype=float)
        t_soil = float(model.get_value('t_soil_1_2'))
        y0 = -np.linalg.solve(A, b * w[0])       # steady at the initial loss
        ref = t_soil + ladder.simulate(t, w, y0=y0)
        errs.append(float(np.max(np.abs(y - ref))))

    assert errs[0] < 1.0, f"coarse-step residual {errs[0]:.3f} K is too large " \
        "to be a zero-order hold"
    # First order: a 5x finer step must shrink the residual ~5x.
    assert errs[0] / errs[1] == pytest.approx(5.0, rel=0.25), (
        f"residual {errs[0]:.4f} -> {errs[1]:.4f} K for a 5x finer step is "
        "not first-order convergence, so it is not the hold")


@pytest.mark.bps
def test_ladder_block_rejects_ambiguous_data():
    """
    ``R``/``C`` and ``T1``/``T4``/``Q1``/``Q2``/``p`` are alternative models,
    so mixing them must raise rather than silently honouring one.

    This matters because a ``dtr_codes`` entry plus per-line overrides makes an
    accidental mix easy, and the two forms inject the loss differently — the
    ladder takes the total at node 1, the two-node form splits it — so a quiet
    fallback would be wrong rather than merely approximate.
    """
    from pydae.bps.lines.line_dtr_cable import _resolve_ladder

    R, C = [0.1, 0.2, 0.3], [1e2, 1e4, 1e6]
    assert _resolve_ladder({'R': R, 'C': C}) == (R, C)
    assert _resolve_ladder({}) is None
    assert _resolve_ladder({'T1': 0.37, 'T4': 1.3, 'Q1': 400.0}) is None

    for clash in ({'T1': 0.37}, {'T4': 1.3}, {'Q1': 400.0}, {'Q2': 1e5},
                  {'p': 0.4}):
        with pytest.raises(ValueError, match="alternative models"):
            _resolve_ladder({'R': R, 'C': C, **clash})

    for bad, why in (({'R': R}, "only R"),
                     ({'C': C}, "only C"),
                     ({'R': R, 'C': C[:-1]}, "length mismatch"),
                     ({'R': [], 'C': []}, "empty"),
                     ({'R': [0.1, -0.2], 'C': [1.0, 2.0]}, "negative R"),
                     ({'R': [0.1, 0.2], 'C': [1.0, 0.0]}, "zero C"),
                     ({'R': ['a'], 'C': [1.0]}, "non-numeric")):
        with pytest.raises(ValueError):
            _resolve_ladder(bad)


# =============================================================================
# 5. THE WHOLE SUITE IN ONE CALL
# =============================================================================

@pytest.mark.bps
@pytest.mark.model
@pytest.mark.slow
def test_run_suite_end_to_end(tmp_path):
    """
    ``run_suite`` is the documented entry point and ``verify=True`` the
    documented way to close the loop. Run both on the test-grade mesh and check
    the artefacts they promise.
    """
    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        res = cf.run_suite(cf.cigre_mv_eu_case(TEST_CURRENT),
                           n_stages=5,
                           in_cable_passes=IN_CABLE_PASSES,
                           steady_passes=None,
                           refine_factors=REFINE_FACTORS,
                           orders=(2, 5),
                           validate=True,
                           verify=True,
                           json_path=str(tmp_path / "dtr.json"),
                           verbose=False)
    finally:
        os.chdir(cwd)

    assert set(res) >= {'case', 'fem', 'steady', 'ladder', 'fit', 'orders',
                        'validation', 'verification'}

    # A higher-order ladder must beat a lower-order one.
    assert res['orders'][5]['err'] < res['orders'][2]['err']

    # The ladder validation against the direct transient FEM solve ran. The
    # bound tracks the reported order: the n = 5 ladder sits within ~0.02
    # K.m/W of a direct backward-Euler solve, n = 4 within ~0.12.
    val = res['validation']
    assert val['err_ladder_vs_fem'] < 0.03
    assert val['err_ladder_vs_foster'] < 1e-6
    assert val['dc_gain'] == pytest.approx(res['ladder'].R.sum(), abs=1e-8)

    # verify=True built the real pydae-bps model and checked it.
    v = res['verification']
    assert v is not None, "verification was skipped; pydae-bps not importable?"
    assert v['ok'], v
    assert v['monotonic']
    assert v['err_steady_relation'] < 1e-3
    assert v['responds_to_soil_step']

    # The promised JSON artefact, with both export modes and the full ladder.
    payload = json.loads((tmp_path / "dtr.json").read_text())
    assert payload['two_node']['model'] == 'cable'
    assert not {'T1', 'T4'} & set(payload['two_node_geometry_only'])
    assert len(payload['ladder']['R']) == len(payload['ladder']['C']) == 5
    assert payload['steady']['z_hot'] > 0.0
