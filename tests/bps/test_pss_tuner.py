# -*- coding: utf-8 -*-
"""Tests for the residue PSS tuner.

The two integration assertions are the ones that matter.  ``arg(R H) = 180 deg``
catches any reintroduced sign or 180-degree error in the phase derivation, and
first-order agreement at low gain catches a residue taken on the wrong loop --
both failure modes produce entirely plausible-looking numbers otherwise.

    uv run pytest tests/bps/test_pss_tuner.py
"""
import os

import numpy as np
import pytest

from pydae.bps.utils.pss_tuner import (
    PSS_MODELS, _spec, _param, _wrap, _zeta,
    design_compensator, required_phase, pick_gain, write_pss_params,
    machines_with_pss, Design,
)

HERE = os.path.dirname(os.path.abspath(__file__))

# The Kundur two-area case lives in the separate benchmarks_public repo, checked
# out next to this one. Override with PYDAE_K2A_DIR if it sits elsewhere. When
# it is not present these tests skip rather than fail -- CI does not check it
# out, and nothing else in the suite depends on an external repository.
K2A = os.environ.get('PYDAE_K2A_DIR') or os.path.normpath(
    os.path.join(HERE, '..', '..', '..',
                 'benchmarks_public', 'kundur2area', 'cases', 'pestr18'))

pytestmark = pytest.mark.skipif(
    not os.path.isdir(K2A),
    reason=f'Kundur two-area benchmark not found at {K2A}; '
           'clone benchmarks_public beside pydae or set PYDAE_K2A_DIR',
)


# ─────────────────────────────────────────────────────────────────────────────
# Unit level
# ─────────────────────────────────────────────────────────────────────────────

def test_unsupported_pss_type_names_the_supported_ones():
    with pytest.raises(ValueError, match='not supported'):
        _spec('pss2a')


def test_param_naming():
    assert _param('K_stab', 'G01') == 'K_stab_pss_G01'
    assert _param('T_5', 'biomass_ES120') == 'T_5_pss_biomass_ES120'


def test_every_registered_model_is_self_consistent():
    for name, spec in PSS_MODELS.items():
        assert spec['stages'], f'{name} has no lead-lag stages'
        for num, den in spec['stages']:
            assert num != den


# arg(R) near 0 is deliberately absent: it demands ~180 deg of lead, i.e. 90
# deg/stage, which no lead-lag can produce at any alpha.  That case belongs to
# test_over_phase_is_reported_not_silently_inverted.
@pytest.mark.parametrize('arg_R', [-170, -90, -30, 45, 120, 179])
def test_compensator_lands_on_180_degrees(arg_R):
    """The design must put arg(R H) at 180 deg, whatever the residue angle."""
    lam = -0.2 + 6.3j
    R = 0.15 * np.exp(1j * np.deg2rad(arg_R))
    out, _reason = design_compensator(lam, R, T_w=10.0, n_stages=2,
                                      phi_max_per_stage=89.0)
    assert out is not None
    T_1, T_2, phi, alpha, H_unit = out
    assert abs(_wrap(np.angle(R * H_unit, deg=True) - 180.0)) < 1.0
    assert alpha == pytest.approx(T_1 / T_2)


def test_over_phase_is_reported_not_silently_inverted():
    """A residue near 0 deg needs ~180 deg of lead; that must be refused."""
    lam = -0.9 + 10.9j
    R = 0.15 * np.exp(1j * np.deg2rad(2.5))
    out, reason = design_compensator(lam, R, T_w=10.0, n_stages=2,
                                     phi_max_per_stage=70.0)
    assert out is None
    assert 'deg/stage' in reason


def test_washout_is_accounted_for():
    lam = -0.2 + 6.3j
    R = 0.1 * np.exp(1j * np.deg2rad(60))
    phi_long, _ = required_phase(lam, R, T_w=10.0)     # negligible washout
    phi_short, _ = required_phase(lam, R, T_w=0.5)     # substantial washout
    assert abs(phi_short - phi_long) > 5.0


# Rows are (K, lam, zeta, floor, corr, held).  `zeta` is the target mode; `floor`
# is the worst electromechanical mode in the system, and is what selection is on.

def test_pick_gain_discards_rows_where_tracking_was_lost():
    # the lost row carries by far the best floor
    trace = [(0.0, -0.1+6.3j, 0.016, 0.016, 1.00, True),
             (0.5, -0.4+6.3j, 0.063, 0.063, 0.99, True),
             (1.0, -0.7+6.3j, 0.110, 0.110, 0.97, True),
             (3.0, -2.7+3.3j, 0.630, 0.630, 0.79, False)]
    K, z, lam, fl = pick_gain(trace, zeta_target=0.10)
    assert K == 1.0 and z == pytest.approx(0.110) and fl == pytest.approx(0.110)


def test_pick_gain_falls_back_to_the_knee_when_target_unreachable():
    trace = [(0.0, -0.1+6.3j, 0.016, 0.016, 1.00, True),
             (0.5, -0.4+6.3j, 0.045, 0.045, 0.99, True),
             (1.0, -0.42+6.3j, 0.047, 0.047, 0.97, True),
             (2.0, -0.43+6.3j, 0.048, 0.048, 0.95, True)]
    K, _z, _lam, _fl = pick_gain(trace, zeta_target=0.10)
    assert K == 0.5                       # smallest within half a point of best


def test_pick_gain_returns_none_when_nothing_held():
    trace = [(1.0, -2.7+3.3j, 0.63, 0.63, 0.5, False)]
    assert pick_gain(trace, 0.10) == (None, None, None, None)


def test_pick_gain_refuses_a_local_gain_that_costs_the_inter_area_mode():
    """The point of selecting on the floor rather than the target mode.

    K = 2.0 damps the target mode best (12%) but drags the worst mode in the
    system down to 1%.  K = 0.5 is the better system-wide answer even though its
    target-mode damping is half as good.
    """
    trace = [(0.0, -0.10+6.3j, 0.016, 0.016, 1.00, True),
             (0.5, -0.40+6.3j, 0.063, 0.055, 0.99, True),
             (1.0, -0.70+6.3j, 0.098, 0.030, 0.98, True),
             (2.0, -0.85+6.3j, 0.120, 0.010, 0.96, True)]
    K, z, _lam, fl = pick_gain(trace, zeta_target=0.10)
    assert K == 0.5
    assert fl == pytest.approx(0.055)
    assert z == pytest.approx(0.063)      # not the best target-mode damping


# ─────────────────────────────────────────────────────────────────────────────
# hjson write-back
# ─────────────────────────────────────────────────────────────────────────────

def test_write_back_preserves_layout_and_touches_only_its_fields(tmp_path):
    src = tmp_path / 'sys.hjson'
    src.write_text(
        'syns: [\n'
        '  // a comment that must survive\n'
        '  {bus: "01", name: "G01", type: "genroe", H: 6.5,\n'
        '    pss: {type: "pss_kundur_2", K_stab: 20.0, T_w: 10.0, T_1: 0.05, '
        'T_2: 0.02, T_3: 3.0, T_4: 5.4, V_Smax: 0.05, V_Smin: -0.05}\n'
        '  },\n'
        '  {bus: "02", name: "G02", type: "genroe", H: 6.5,\n'
        '    pss: {type: "pss_kundur_2", K_stab: 20.0, T_w: 10.0, T_1: 0.05, '
        'T_2: 0.02, T_3: 3.0, T_4: 5.4, V_Smax: 0.05, V_Smin: -0.05}\n'
        '  }\n'
        ']\n', encoding='utf-8')

    d = Design(machine='G02', pss_type='pss_kundur_2', f_hz=1.0,
               zeta_before=0.01, zeta_after=0.11, lam_before=0j, lam_after=0j,
               R_abs=0.1, R_deg=60.0, participation=0.5, phi_deg=110.0,
               alpha=10.0, T_1=0.5097, T_2=0.0489, gain=0.4,
               gain_param='K_stab_pss_G02')
    write_pss_params(str(src), [d])
    out = src.read_text(encoding='utf-8')

    assert 'a comment that must survive' in out
    g01, g02 = out.split('name: "G02"')
    assert 'K_stab: 20.0' in g01 and 'T_1: 0.05' in g01     # G01 untouched
    assert 'K_stab: 0.4' in g02
    assert 'T_1: 0.5097' in g02 and 'T_3: 0.5097' in g02
    assert 'T_2: 0.0489' in g02 and 'T_4: 0.0489' in g02
    assert 'V_Smax: 0.05' in g02                            # limits left alone
    assert 'T_w: 10.0' in g02                               # washout left alone


def test_disable_comments_out_the_whole_pss_entry(tmp_path):
    """An unused stabiliser is commented out, not written with a zero gain.

    hjson must still parse, the machine must no longer carry a pss block at all
    (so add_pss never runs and the model gains no dead states), and the values
    must survive in the comment so the block can be put back.
    """
    hjson = pytest.importorskip('hjson')
    src = tmp_path / 'sys.hjson'
    src.write_text(
        'syns: [\n'
        '  {bus: "01", name: "G01", type: "genroe", H: 6.5,\n'
        '   "pss": {\n'
        '    "type": "pss_kundur_2",\n'
        '    "K_stab": 20.0,\n'
        '    "T_1": 0.05,\n'
        '    "T_2": 0.02,\n'
        '    "T_3": 3.0,\n'
        '    "T_4": 5.4\n'
        '   },\n'
        '   "lc": {"K_i": 0.01}\n'
        '  },\n'
        '  {bus: "02", name: "G02", type: "genroe", H: 6.5,\n'
        '   "pss": {"type": "pss_kundur_2", "K_stab": 20.0, "T_1": 0.05,\n'
        '           "T_2": 0.02, "T_3": 3.0, "T_4": 5.4}\n'
        '  }\n'
        ']\n', encoding='utf-8')

    write_pss_params(str(src), [], disable=['G01'])
    out = src.read_text(encoding='utf-8')

    parsed = hjson.loads(out)
    by_name = {s['name']: s for s in parsed['syns']}
    assert 'pss' not in by_name['G01'], 'G01 should carry no pss block'
    assert 'lc' in by_name['G01'], 'the rest of the machine must survive'
    assert by_name['G02']['pss']['K_stab'] == 20.0, 'G02 must be untouched'
    assert '// ' in out and 'K_stab": 20.0' in out, 'values kept in the comment'


def test_write_back_rejects_a_machine_without_a_pss_block(tmp_path):
    src = tmp_path / 'sys.hjson'
    src.write_text('syns: [\n  {name: "G01", type: "genroe"},\n'
                   '  {name: "G02", pss: {type: "pss_kundur_2", K_stab: 1.0, '
                   'T_1: 1.0, T_2: 1.0, T_3: 1.0, T_4: 1.0}}\n]\n',
                   encoding='utf-8')
    d = Design(machine='G01', pss_type='pss_kundur_2', f_hz=1.0,
               zeta_before=0.0, zeta_after=0.1, lam_before=0j, lam_after=0j,
               R_abs=0.1, R_deg=0.0, participation=0.0, phi_deg=0.0, alpha=1.0,
               T_1=1.0, T_2=1.0, gain=1.0, gain_param='K_stab_pss_G01')
    with pytest.raises(ValueError, match='no pss block'):
        write_pss_params(str(src), [d])


# ─────────────────────────────────────────────────────────────────────────────
# Integration: Kundur two-area, PES-TR18 variant
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope='module')
def k2a():
    """Built Kundur two-area model, all stabiliser gains at zero."""
    pytest.importorskip('casadi')
    from pydae.bps import BpsBuilder
    from pydae.core.builder.casadi_builder import CasadiBuilder
    from pydae.core.model.casadi_model import CasadiModel

    cwd = os.getcwd()
    os.chdir(K2A)
    try:
        grid = BpsBuilder('kundur2area.hjson', use_casadi=True)
        grid.uz_jacs = False
        grid.construct('k2a_tuner_test')
        model = CasadiModel(CasadiBuilder(grid.sys_dict).build())
        yield model, 'kundur2area.hjson'
    finally:
        os.chdir(cwd)


def test_machines_with_pss_finds_all_four(k2a):
    _model, data = k2a
    os.chdir(K2A)
    assert sorted(m for m, _t in machines_with_pss(data)) == \
        ['G01', 'G02', 'G03', 'G04']


def test_tuner_stabilises_the_two_area_system(k2a):
    """From an unstable inter-area mode to every mode above target."""
    from pydae.bps.utils.pss_tuner import tune_psss
    from pydae.ssa import eig

    model, data = k2a
    os.chdir(K2A)
    target = 0.10
    report = tune_psss(model, data, zeta_target=target, verbose=False)

    assert report.designs, 'the tuner produced no design at all'

    # It starts from an unstable inter-area mode.
    assert min(d.zeta_before for d in report.designs) < 0.0

    eig(model)
    ev = model.eigenvalues
    emech = [e for e in ev
             if e.imag > 0.05 and 0.1 <= e.imag/(2*np.pi) <= 2.5]
    worst = min(_zeta(e) for e in emech)
    assert worst >= target - 1e-3, f'worst electromechanical zeta is {worst:.4f}'


def test_designs_satisfy_the_180_degree_rule(k2a):
    """arg(R H) = 180 deg, exactly for a point design, approximately across a band.

    A compensator fitted to several modes deliberately gives up exactness at any
    one of them, so the invariant is only strict when the band held a single mode.
    """
    from pydae.bps.utils.pss_tuner import tune_psss

    model, data = k2a
    os.chdir(K2A)
    report = tune_psss(model, data, zeta_target=0.10, verbose=False)

    for d in report.designs:
        lam, T_w = d.lam_before, model.get_value(_param('T_w', d.machine))
        n = len(_spec(d.pss_type)['stages'])
        R = d.R_abs * np.exp(1j * np.deg2rad(d.R_deg))
        G_w = (lam * T_w) / (1.0 + lam * T_w)
        G_c = ((1.0 + d.T_1 * lam) / (1.0 + d.T_2 * lam)) ** n
        err = abs(_wrap(np.angle(R * G_w * G_c, deg=True) - 180.0))

        if d.n_band == 1:
            assert err < 1.0, f'{d.machine}: arg(R H) off 180 deg by {err:.2f}'
        else:
            assert d.phase_err_deg < 30.0, (
                f'{d.machine}: weighted phase error {d.phase_err_deg:.1f} deg '
                f'over {d.n_band} modes')


def test_band_fit_reduces_to_the_point_design_for_one_mode():
    """With a single mode in the band the fit must reproduce design_compensator."""
    from pydae.bps.utils.pss_tuner import design_compensator_band

    lam = -0.2 + 6.3j
    R = 0.15 * np.exp(1j * np.deg2rad(70))
    point, _ = design_compensator(lam, R, T_w=10.0, n_stages=2)
    band, _ = design_compensator_band([lam], [R], T_w=10.0, n_stages=2)

    assert band is not None
    assert band[0] == pytest.approx(point[0], rel=1e-3)     # T_1
    assert band[1] == pytest.approx(point[1], rel=1e-3)     # T_2
    assert band[5] < 1.0                                    # residual phase error


def test_band_fit_beats_the_point_design_across_two_modes():
    """The reason for fitting a band: a design centred on one mode is wrong at the other.

    An inter-area mode at 0.34 Hz and a local mode at 1.05 Hz, both with authority.
    The point design nails the local mode and is badly off at the inter-area one;
    the band fit must have lower weighted error over the pair.
    """
    from pydae.bps.utils.pss_tuner import design_compensator_band, required_phase

    T_w, n = 10.0, 2
    lams = [-0.05 + 2.15j, -0.30 + 6.60j]
    Rs = [0.05 * np.exp(1j * np.deg2rad(150)),
          0.13 * np.exp(1j * np.deg2rad(60))]

    point, _ = design_compensator(lams[1], Rs[1], T_w, n)   # centred on the local mode
    band, _ = design_compensator_band(lams, Rs, T_w, n)
    assert point is not None and band is not None

    def werr(T_1, T_2):
        num = tot = 0.0
        for lam, R in zip(lams, Rs):
            phi, _gw = required_phase(lam, R, T_w)
            got = np.angle(((1 + T_1*lam) / (1 + T_2*lam)) ** n, deg=True)
            num += abs(R) * _wrap(got - phi) ** 2
            tot += abs(R)
        return np.sqrt(num / tot)

    assert werr(band[0], band[1]) < werr(point[0], point[1])


def test_tuning_never_gives_back_the_inter_area_mode(k2a):
    """Damping a local mode must not be paid for out of the inter-area mode.

    The failure this pins down is real: during manual tuning of the Spanish case
    the 0.34 Hz inter-area mode went -3.08% -> -2.54% -> -2.90% as local-mode
    stabilisers were added one at a time.  Every slow mode must end at least as
    well damped as it started.
    """
    from pydae.bps.utils.pss_tuner import tune_psss
    from pydae.ssa import eig

    model, data = k2a
    os.chdir(K2A)

    # One run only: the open loop comes from the report, because a second ini()
    # on the same model lands on a different operating point.
    report = tune_psss(model, data, zeta_target=0.10, verbose=False)
    before = {e.imag: _zeta(e) for e in report.modes_before
              if e.imag/(2*np.pi) <= 0.7}
    assert before, 'no slow modes to check'

    eig(model)
    after = [e for e in model.eigenvalues
             if e.imag > 0.05 and e.imag/(2*np.pi) <= 0.9]

    for imag, z0 in before.items():
        # follow the slow mode by frequency; it barely moves in frequency
        near = [e for e in after if abs(e.imag - imag) <= 0.35 * imag]
        assert near, f'slow mode at {imag:.3f} rad/s vanished'
        z1 = max(_zeta(e) for e in near)
        assert z1 >= z0 - 1e-3, (
            f'inter-area mode at {imag/(2*np.pi):.2f} Hz went from '
            f'{100*z0:.2f}% to {100*z1:.2f}%')


def test_first_order_prediction_matches_the_closed_loop(k2a):
    """The residue is only right if a small gain moves the mode where it says."""
    from pydae.bps.utils.pss_tuner import (loop_residue, sweep_gain,
                                           _candidate_modes, _spec, _param)
    from pydae.ssa import eig

    model, data = k2a
    os.chdir(K2A)
    for m, t in machines_with_pss(data):
        model.set_value(_param(_spec(t)['gain'], m), 0.0)
    model.ini({})
    eig(model)

    k = min(_candidate_modes(model, (0.1, 2.5), 0.10, []),
            key=lambda k: _zeta(model.eigenvalues[k]))
    host = max((m for m, _t in machines_with_pss(data)),
               key=lambda m: abs(loop_residue(model, m, k)))
    R = loop_residue(model, host, k)
    lam0 = model.eigenvalues[k]

    T_w = model.get_value(_param('T_w', host))
    out, _r = design_compensator(lam0, R, T_w, 2, phi_max_per_stage=89.0)
    T_1, T_2, _phi, _alpha, H_unit = out
    for num, den in _spec('pss_kundur_2')['stages']:
        model.set_value(_param(num, host), T_1)
        model.set_value(_param(den, host), T_2)

    # A gain small enough that the linearisation must hold.
    K = 0.02 * abs(lam0.real) / max(abs(R * H_unit), 1e-9)
    trace = sweep_gain(model, _param('K_stab', host), k, R, H_unit, 0.10,
                       gains=[0.0, K])
    lam_cl = trace[1][1]
    pred = lam0 + K * R * H_unit
    assert abs(lam_cl - pred) < 0.02 * abs(lam0), (
        f'first-order prediction {pred} vs closed loop {lam_cl}')
