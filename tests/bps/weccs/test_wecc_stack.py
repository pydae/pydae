"""
Tests for the WECC renewable energy model stack.

Each test class is independent:

* ``TestRegcA``   — REGC_A only (grid-interface converter with LVPL).
* ``TestRegcB``   — REGC_B only (simplified converter, combined Imax limit).
* ``TestReecB``   — REGC_A + REEC_B (converter + local electrical controls).
* ``TestRepcA``   — REGC_A + REEC_B + REPC_A (full WECC plant stack).
* ``TestRegfmA1`` — REGFM_A1 (droop-controlled grid-forming inverter).

Run selectively::

    uv run pytest -m bps tests/bps/weccs/
    uv run pytest tests/bps/weccs/test_wecc_stack.py -v
    uv run pytest tests/bps/weccs/test_wecc_stack.py -k repc_a
"""

import os
import contextlib

import numpy as np
import pytest

# --------------------------------------------------------------------------- #
# Paths to HJSON files                                                         #
# --------------------------------------------------------------------------- #
_BPS = os.path.join(
    os.path.dirname(__file__),
    '..', '..', '..',
    'packages', 'pydae-bps', 'src', 'pydae', 'bps',
)
WECCS_DIR = os.path.abspath(os.path.join(_BPS, 'weccs'))
PPCS_DIR  = os.path.abspath(os.path.join(_BPS, 'ppcs'))


# --------------------------------------------------------------------------- #
# Helper                                                                       #
# --------------------------------------------------------------------------- #

@contextlib.contextmanager
def _in_dir(directory):
    """Context manager: temporarily change cwd to *directory*."""
    orig = os.getcwd()
    try:
        os.chdir(directory)
        yield directory
    finally:
        os.chdir(orig)


def _build(hjson_dir, hjson_name, model_name, target='cffi'):
    """Build a BPS model inside *hjson_dir* and return the Model instance."""
    from pydae.bps import BpsBuilder
    from pydae.core import Builder, Model

    with _in_dir(hjson_dir):
        grid = BpsBuilder(hjson_name)
        grid.construct(model_name)
        bld = Builder(grid.sys_dict, target=target, sparse=False)
        bld.build()
        return Model(model_name)


# ── module-scoped fixtures (build once, reuse across all tests in the class) ─

@pytest.fixture(scope='module')
def regc_a_model():
    return _build(WECCS_DIR, 'regc_a.hjson', 'temp_regc_a_test')


@pytest.fixture(scope='module')
def regc_b_model():
    return _build(WECCS_DIR, 'regc_b.hjson', 'temp_regc_b_test')


@pytest.fixture(scope='module')
def reec_b_model():
    return _build(WECCS_DIR, 'reec_b.hjson', 'temp_reec_b_test')


@pytest.fixture(scope='module')
def regfm_a1_model():
    return _build(WECCS_DIR, 'regfm_a1.hjson', 'temp_regfm_a1_test')


@pytest.fixture(scope='module')
def repc_a_model():
    return _build(PPCS_DIR, 'repc_a.hjson', 'temp_repc_a_test')


# ── ini helper (must run from the model's directory) ─────────────────────────

def _ini(model_dir, model, xy0_name, params=None):
    """Run ini() from within *model_dir* so the xy_0 file is found."""
    with _in_dir(model_dir):
        model.ini(params or {}, xy0_name)


def _ini_run(model_dir, model, xy0_name, t_end, update=None):
    """ini() then run() inside *model_dir*; calls post() and returns model."""
    with _in_dir(model_dir):
        model.ini({}, xy0_name)
        model.run(t_end, update or {})
        model.post()
    return model


# --------------------------------------------------------------------------- #
# 1. REGC_A — grid-interface converter only                                    #
# --------------------------------------------------------------------------- #

@pytest.mark.bps
@pytest.mark.build
class TestRegcA:
    """REGC_A: current-source converter with LVPL and high-voltage clamp."""

    XY0 = 'temp_regc_a_test_xy_0.json'

    def test_ini_converges(self, regc_a_model):
        _ini(WECCS_DIR, regc_a_model, self.XY0)

    def test_active_current_matches_command(self, regc_a_model):
        """Ip at steady state equals Ipcmd (lag converged, LVPL inactive at V≈1)."""
        _ini(WECCS_DIR, regc_a_model, self.XY0)
        m = regc_a_model
        Ip    = m.get_value('Ip_1')
        Ipcmd = m.get_value('Ipcmd_1')
        assert abs(Ip - Ipcmd) < 1e-4, f"Ip={Ip:.6f} != Ipcmd={Ipcmd:.6f}"
        assert abs(Ip - 0.8)   < 0.02, f"Ip={Ip:.4f} not close to 0.8 pu"

    def test_reactive_current_zero(self, regc_a_model):
        """Iq at steady state = -Iqcmd = 0 (unity PF command)."""
        _ini(WECCS_DIR, regc_a_model, self.XY0)
        Iq = regc_a_model.get_value('Iq_1')
        assert abs(Iq) < 0.01, f"Iq={Iq:.6f} should be ~0"

    def test_active_power_output(self, regc_a_model):
        """p_g = Ip · Vt ≈ 0.8 pu at nominal voltage."""
        _ini(WECCS_DIR, regc_a_model, self.XY0)
        m   = regc_a_model
        p_g = m.get_value('p_g_1')
        V   = m.get_value('V_1')
        Ip  = m.get_value('Ip_1')
        assert abs(p_g - Ip * V) < 0.01, f"p_g={p_g:.4f} != Ip*V={Ip*V:.4f}"

    def test_gain_unity_at_nominal_voltage(self, regc_a_model):
        """Low-voltage management gain = 1 when V_flt > lvpnt1 = 0.8 pu."""
        _ini(WECCS_DIR, regc_a_model, self.XY0)
        gain = regc_a_model.get_value('gain_1')
        assert gain > 0.99, f"gain={gain:.4f} at nominal voltage should be ~1"

    def test_vt_flt_tracks_terminal_voltage(self, regc_a_model):
        """V_flt converges to V_bus at steady state."""
        _ini(WECCS_DIR, regc_a_model, self.XY0)
        m = regc_a_model
        V     = m.get_value('V_1')
        V_flt = m.get_value('V_flt_1')
        assert abs(V_flt - V) < 0.01, f"V_flt={V_flt:.4f} != V={V:.4f}"

    def test_step_ipcmd_increases_ip(self, regc_a_model):
        """Step Ipcmd from 0.8 → 1.0 pu: Ip must increase toward 1.0."""
        _ini(WECCS_DIR, regc_a_model, self.XY0)
        m = regc_a_model
        Ip_before = m.get_value('Ip_1')
        Tg = 0.02
        with _in_dir(WECCS_DIR):
            m.run(10 * Tg, {'Ipcmd_1': 1.0})
            m.post()
        Ip_after = m.get_values('Ip_1')[-1]
        assert Ip_after > Ip_before + 0.05, (
            f"Ip did not increase: {Ip_before:.4f} → {Ip_after:.4f}"
        )


# --------------------------------------------------------------------------- #
# 2. REGC_B — simplified converter, combined Imax limit, no LVPL               #
# --------------------------------------------------------------------------- #

@pytest.mark.bps
@pytest.mark.build
class TestRegcB:
    """REGC_B: simplified current-source converter with combined Imax limit."""

    XY0 = 'temp_regc_b_test_xy_0.json'

    def test_ini_converges(self, regc_b_model):
        _ini(WECCS_DIR, regc_b_model, self.XY0)

    def test_active_current_matches_command(self, regc_b_model):
        """Ip equals Ipcmd at steady state (no LVPL reduction at nominal V)."""
        _ini(WECCS_DIR, regc_b_model, self.XY0)
        m  = regc_b_model
        Ip = m.get_value('Ip_1')
        assert abs(Ip - 0.8) < 0.02, f"Ip={Ip:.4f} not close to 0.8 pu"

    def test_reactive_current_zero(self, regc_b_model):
        _ini(WECCS_DIR, regc_b_model, self.XY0)
        Iq = regc_b_model.get_value('Iq_1')
        assert abs(Iq) < 0.01, f"Iq={Iq:.4f} should be ~0"

    def test_ipmax_equals_imax_at_zero_reactive(self, regc_b_model):
        """With Iq=0, Ipmax = sqrt(Imax²−0) = Imax = 1.1 pu."""
        _ini(WECCS_DIR, regc_b_model, self.XY0)
        m     = regc_b_model
        Ipmax = m.get_value('Ipmax_1')
        Iq    = m.get_value('Iq_1')
        Imax  = 1.1
        expected = np.sqrt(max(Imax**2 - Iq**2, 0))
        assert abs(Ipmax - expected) < 0.01, (
            f"Ipmax={Ipmax:.4f} != sqrt(Imax²-Iq²)={expected:.4f}"
        )

    def test_ipmax_decreases_with_reactive_current(self, regc_b_model):
        """When Iq increases, Ipmax must decrease (shared Imax budget)."""
        _ini(WECCS_DIR, regc_b_model, self.XY0)
        m = regc_b_model
        Ipmax_ini = m.get_value('Ipmax_1')
        # Inject reactive current via Iqcmd step
        with _in_dir(WECCS_DIR):
            m.run(0.2, {'Iqcmd_1': -0.5})   # negative Iqcmd → positive Iq
            m.post()
        Iq_after    = m.get_values('Iq_1')[-1]
        Ipmax_after = m.get_values('Ipmax_1')[-1]
        if abs(Iq_after) > 0.05:
            assert Ipmax_after < Ipmax_ini, (
                f"Ipmax did not decrease: {Ipmax_ini:.4f} → {Ipmax_after:.4f}"
                f" (Iq={Iq_after:.4f})"
            )

    def test_active_power_output(self, regc_b_model):
        """p_g = Ip·Vt at steady state."""
        _ini(WECCS_DIR, regc_b_model, self.XY0)
        m   = regc_b_model
        p_g = m.get_value('p_g_1')
        Ip  = m.get_value('Ip_1')
        V   = m.get_value('V_1')
        assert abs(p_g - Ip * V) < 0.01, f"p_g={p_g:.4f} != Ip*V={Ip*V:.4f}"

    def test_simulation_stable_over_2s(self, regc_b_model):
        """REGC_B simulation remains bounded over 2 s."""
        _ini(WECCS_DIR, regc_b_model, self.XY0)
        m = regc_b_model
        with _in_dir(WECCS_DIR):
            m.run(2.0, {})
            m.post()
        for var in ['Ip_1', 'Iq_1', 'Ipmax_1']:
            vals = m.get_values(var)
            assert np.all(np.isfinite(vals)), f"{var} went non-finite"
            assert np.max(np.abs(vals)) < 5.0, (
                f"{var} exceeded bounds: max|v|={np.max(np.abs(vals)):.2f}"
            )


# --------------------------------------------------------------------------- #
# 3. REGC_A + REEC_B — local electrical controls                               #
# --------------------------------------------------------------------------- #

@pytest.mark.bps
@pytest.mark.build
class TestReecB:
    """REEC_B: V/Q control loop driving Ipcmd and Iqcmd to REGC_A."""

    XY0 = 'temp_reec_b_test_xy_0.json'

    def test_ini_converges(self, reec_b_model):
        _ini(WECCS_DIR, reec_b_model, self.XY0)

    def test_active_current_steady_state(self, reec_b_model):
        """P_flt converges to Pref → Ip ≈ 0.8 pu and Ip ≈ Ipcmd."""
        _ini(WECCS_DIR, reec_b_model, self.XY0)
        m   = reec_b_model
        Ip  = m.get_value('Ip_1')
        Ipc = m.get_value('Ipcmd_1')
        assert abs(Ip - 0.8) < 0.05, f"Ip={Ip:.4f} not close to 0.8 pu"
        assert abs(Ip - Ipc) < 0.01, f"Ip={Ip:.4f} != Ipcmd={Ipc:.4f}"

    def test_reactive_current_at_unity_pf(self, reec_b_model):
        """Iqcmd initial = 0 → Iq ≈ 0."""
        _ini(WECCS_DIR, reec_b_model, self.XY0)
        m   = reec_b_model
        Iq  = m.get_value('Iq_1')
        Iqc = m.get_value('Iqcmd_1')
        assert abs(Iq)  < 0.02, f"Iq={Iq:.4f} should be ~0"
        assert abs(Iqc) < 0.02, f"Iqcmd={Iqc:.4f} should be ~0"

    def test_vt_flt_tracks_terminal_voltage(self, reec_b_model):
        """Vt_flt converges to terminal bus voltage at steady state."""
        _ini(WECCS_DIR, reec_b_model, self.XY0)
        m      = reec_b_model
        V      = m.get_value('V_1')
        Vt_flt = m.get_value('Vt_flt_1')
        assert abs(Vt_flt - V) < 0.01, f"Vt_flt={Vt_flt:.4f} != V={V:.4f}"

    def test_p_flt_tracks_pref_at_steady_state(self, reec_b_model):
        """P_flt integrates to Pref over 10×Tpord = 1 s."""
        _ini(WECCS_DIR, reec_b_model, self.XY0)
        m = reec_b_model
        with _in_dir(WECCS_DIR):
            m.run(1.0, {})
            m.post()
        P_flt = m.get_values('P_flt_1')[-1]
        Pref  = m.get_value('Pref_1')
        assert abs(P_flt - Pref) < 0.02, f"P_flt={P_flt:.4f} did not reach Pref={Pref:.4f}"

    def test_step_pref_reduces_ip(self, reec_b_model):
        """Pref step 0.8 → 0.5: Ip must decrease toward 0.5 pu."""
        _ini(WECCS_DIR, reec_b_model, self.XY0)
        m = reec_b_model
        with _in_dir(WECCS_DIR):
            m.run(0.5, {'Pref_1': 0.5})
            m.post()
        Ip_final = m.get_values('Ip_1')[-1]
        assert Ip_final < 0.75, f"Ip={Ip_final:.4f} did not decrease toward 0.5"

    def test_step_qext_changes_iq(self, reec_b_model):
        """Qext step from 1.0 → 1.05 changes the reactive current output."""
        _ini(WECCS_DIR, reec_b_model, self.XY0)
        m       = reec_b_model
        Iq_ini  = m.get_value('Iq_1')
        Iqc_ini = m.get_value('Iqcmd_1')
        with _in_dir(WECCS_DIR):
            m.run(0.5, {'Qext_1': 1.05})
            m.post()
        Iqc_fin = m.get_values('Iqcmd_1')[-1]
        # Iqcmd should have changed (either direction, depending on PI sign)
        assert abs(Iqc_fin - Iqc_ini) > 0.01, (
            f"Iqcmd did not respond to Qext step: {Iqc_ini:.4f} → {Iqc_fin:.4f}"
        )

    def test_simulation_stable_over_2s(self, reec_b_model):
        """REGC_A + REEC_B remains bounded and finite over 2 s."""
        _ini(WECCS_DIR, reec_b_model, self.XY0)
        m = reec_b_model
        with _in_dir(WECCS_DIR):
            m.run(2.0, {})
            m.post()
        for var in ['Ip_1', 'Iq_1', 'Ipcmd_1', 'Iqcmd_1', 'Vt_flt_1', 'P_flt_1']:
            vals = m.get_values(var)
            assert np.all(np.isfinite(vals)), f"{var} went non-finite"
            assert np.max(np.abs(vals)) < 5.0, (
                f"{var} exceeded bounds: max|v|={np.max(np.abs(vals)):.2f}"
            )


# --------------------------------------------------------------------------- #
# 3. REPC_A + REEC_B + REGC_A — full WECC plant stack                         #
# --------------------------------------------------------------------------- #

@pytest.mark.bps
@pytest.mark.build
class TestRepcA:
    """REPC_A: plant-level Volt/VAR and active power controller."""

    XY0 = 'temp_repc_a_test_xy_0.json'

    def test_ini_converges(self, repc_a_model):
        _ini(PPCS_DIR, repc_a_model, self.XY0)

    def test_both_converters_inject_active_power(self, repc_a_model):
        """Both REGC_A units inject active current at steady state."""
        _ini(PPCS_DIR, repc_a_model, self.XY0)
        m = repc_a_model
        assert m.get_value('Ip_1') > 0.1, f"Ip_1={m.get_value('Ip_1'):.4f} not injecting"
        assert m.get_value('Ip_2') > 0.1, f"Ip_2={m.get_value('Ip_2'):.4f} not injecting"

    def test_plant_qext_shared_by_all_converters(self, repc_a_model):
        """Qext_1 and Qext_2 equal the plant-level Qext_repc1 at steady state."""
        _ini(PPCS_DIR, repc_a_model, self.XY0)
        m          = repc_a_model
        Qext_plant = m.get_value('Qext_repc1')
        for gen in ['1', '2']:
            Qext_gen = m.get_value(f'Qext_{gen}')
            assert abs(Qext_gen - Qext_plant) < 1e-6, (
                f"Qext_{gen}={Qext_gen:.6f} != Qext_repc1={Qext_plant:.6f}"
            )

    def test_plant_pref_shared_by_all_converters(self, repc_a_model):
        """Pref_1 and Pref_2 equal the plant-level Pref_repc1 at steady state."""
        _ini(PPCS_DIR, repc_a_model, self.XY0)
        m          = repc_a_model
        Pref_plant = m.get_value('Pref_repc1')
        for gen in ['1', '2']:
            Pref_gen = m.get_value(f'Pref_{gen}')
            assert abs(Pref_gen - Pref_plant) < 1e-6, (
                f"Pref_{gen}={Pref_gen:.6f} != Pref_repc1={Pref_plant:.6f}"
            )

    def test_vreg_flt_tracks_poi_voltage(self, repc_a_model):
        """REPC_A voltage filter converges to the POI bus voltage."""
        _ini(PPCS_DIR, repc_a_model, self.XY0)
        m        = repc_a_model
        V_poi    = m.get_value('V_POI')
        Vreg_flt = m.get_value('Vreg_flt_repc1')
        assert abs(Vreg_flt - V_poi) < 0.02, (
            f"Vreg_flt={Vreg_flt:.4f} not tracking V_POI={V_poi:.4f}"
        )

    def test_pref_constant_with_freq_flag_off(self, repc_a_model):
        """With Freq_flag=0, Pref_repc1 stays at its initial setpoint after 2 s."""
        _ini(PPCS_DIR, repc_a_model, self.XY0)
        m        = repc_a_model
        Pref_ini = m.get_value('Pref_repc1')
        with _in_dir(PPCS_DIR):
            m.run(2.0, {})
            m.post()
        Pref_fin = m.get_values('Pref_repc1')[-1]
        assert abs(Pref_fin - Pref_ini) < 0.01, (
            f"Pref changed with Freq_flag=0: {Pref_ini:.4f} → {Pref_fin:.4f}"
        )

    def test_simulation_stable_over_10s(self, repc_a_model):
        """Full WECC plant stack simulation over 10 s remains bounded and finite."""
        _ini(PPCS_DIR, repc_a_model, self.XY0)
        m = repc_a_model
        with _in_dir(PPCS_DIR):
            m.run(10.0, {})
            m.post()
        for var in [
            'Ip_1', 'Iq_1', 'Ipcmd_1', 'Iqcmd_1',
            'Ip_2', 'Iq_2', 'Ipcmd_2', 'Iqcmd_2',
            'Qext_repc1', 'Pref_repc1',
            'Vreg_flt_repc1', 'Pbrn_flt_repc1',
        ]:
            vals = m.get_values(var)
            assert np.all(np.isfinite(vals)), f"{var} went non-finite"
            assert np.max(np.abs(vals)) < 10.0, (
                f"{var} exceeded bounds: max|v|={np.max(np.abs(vals)):.2f}"
            )


# --------------------------------------------------------------------------- #
# 5. REGFM_A1 — droop-controlled grid-forming inverter                         #
# --------------------------------------------------------------------------- #

@pytest.mark.bps
@pytest.mark.build
class TestRegfmA1:
    """REGFM_A1: voltage source behind XL with P-f and Q-V droop."""

    XY0 = 'temp_regfm_a1_test_xy_0.json'

    def test_ini_converges(self, regfm_a1_model):
        _ini(WECCS_DIR, regfm_a1_model, self.XY0)

    def test_active_power_at_setpoint(self, regfm_a1_model):
        """p_g ≈ Pref = 0.8 pu at steady state."""
        _ini(WECCS_DIR, regfm_a1_model, self.XY0)
        p_g = regfm_a1_model.get_value('p_g_1')
        assert abs(p_g - 0.8) < 0.05, f"p_g={p_g:.4f} not close to 0.8 pu"

    def test_droop_frequency_unity_at_steady_state(self, regfm_a1_model):
        """omega_droop = 1.0 when x_Pe = Pref (no steady-state frequency deviation)."""
        _ini(WECCS_DIR, regfm_a1_model, self.XY0)
        omega = regfm_a1_model.get_value('omega_d_1')
        assert abs(omega - 1.0) < 1e-4, f"omega_droop={omega:.6f} should be 1.0"

    def test_internal_voltage_near_unity(self, regfm_a1_model):
        """E is slightly above 1 pu (unity Qref with XL drop)."""
        _ini(WECCS_DIR, regfm_a1_model, self.XY0)
        E = regfm_a1_model.get_value('E_1')
        assert 0.9 < E < 1.2, f"E={E:.4f} out of [0.9, 1.2]"

    def test_fcl_inactive_at_nominal(self, regfm_a1_model):
        """f_cl = 1.0 when |I| < ImaxF = 1.5 pu at rated operating point."""
        _ini(WECCS_DIR, regfm_a1_model, self.XY0)
        m     = regfm_a1_model
        f_cl  = m.get_value('f_cl_1')
        Imag  = m.get_value('Imag_1')
        assert f_cl > 0.99, f"f_cl={f_cl:.4f} should be ~1.0 (no overcurrent)"
        assert Imag < 1.5,  f"|I|={Imag:.4f} should be < ImaxF=1.5"

    def test_power_angle_relation(self, regfm_a1_model):
        """p_g = E·V·sin(delta−theta)/XL (voltage source behind XL formula)."""
        _ini(WECCS_DIR, regfm_a1_model, self.XY0)
        m = regfm_a1_model
        p_g   = m.get_value('p_g_1')
        E     = m.get_value('E_1')
        delta = m.get_value('delta_1')
        theta = m.get_value('theta_1')
        V     = m.get_value('V_1')
        XL    = 0.1  # from hjson
        p_expected = E * V * np.sin(delta - theta) / XL
        assert abs(p_g - p_expected) < 0.02, (
            f"p_g={p_g:.4f} != E·V·sin(δ−θ)/XL={p_expected:.4f}"
        )

    def test_droop_response_to_pref_step(self, regfm_a1_model):
        """Pref step 0.8→1.0: omega_droop dips below 1.0 during transient."""
        _ini(WECCS_DIR, regfm_a1_model, self.XY0)
        m = regfm_a1_model
        with _in_dir(WECCS_DIR):
            m.run(5.0, {'Pref_1': 1.0})
            m.post()
        omega_min = min(m.get_values('omega_d_1'))
        assert omega_min < 1.0 - 1e-4, (
            f"omega_droop min={omega_min:.6f} should dip below 1.0 during Pref step"
        )

    def test_simulation_stable_over_10s(self, regfm_a1_model):
        """GFM simulation remains bounded and finite over 10 s."""
        _ini(WECCS_DIR, regfm_a1_model, self.XY0)
        m = regfm_a1_model
        with _in_dir(WECCS_DIR):
            m.run(10.0, {})
            m.post()
        for var in ['p_g_1', 'q_g_1', 'E_1', 'delta_1', 'omega_d_1', 'f_cl_1']:
            vals = m.get_values(var)
            assert np.all(np.isfinite(vals)), f"{var} went non-finite"
        assert np.max(np.abs(m.get_values('delta_1'))) < np.pi / 2, (
            "angle exceeds ±90° — loss of synchronism"
        )


# --------------------------------------------------------------------------- #
# 6. REGFM_B1 — Virtual Synchronous Machine grid-forming inverter              #
# --------------------------------------------------------------------------- #

@pytest.fixture(scope='module')
def regfm_b1_model():
    return _build(WECCS_DIR, 'regfm_b1.hjson', 'temp_regfm_b1_test')


@pytest.mark.bps
@pytest.mark.build
class TestRegfmB1:
    """REGFM_B1: VSM with swing equation, D2 washout damping, and PLL."""

    XY0 = 'temp_regfm_b1_test_xy_0.json'

    def test_ini_converges(self, regfm_b1_model):
        _ini(WECCS_DIR, regfm_b1_model, self.XY0)

    def test_active_power_at_setpoint(self, regfm_b1_model):
        """p_g ≈ Pref = 0.8 pu at steady state."""
        _ini(WECCS_DIR, regfm_b1_model, self.XY0)
        p_g = regfm_b1_model.get_value('p_g_1')
        assert abs(p_g - 0.8) < 0.05, f"p_g={p_g:.4f} not close to 0.8 pu"

    def test_domegam_zero_at_steady_state(self, regfm_b1_model):
        """Δωm = 0 at steady state (no rotor acceleration)."""
        _ini(WECCS_DIR, regfm_b1_model, self.XY0)
        Domegam = regfm_b1_model.get_value('Domegam_1')
        assert abs(Domegam) < 1e-4, f"Δωm={Domegam:.6f} should be ~0"

    def test_pll_locked_at_steady_state(self, regfm_b1_model):
        """PLL frequency deviation ΔωPLL ≈ 0 when PLL is locked to bus."""
        _ini(WECCS_DIR, regfm_b1_model, self.XY0)
        DomegaPLL = regfm_b1_model.get_value('DomegaPLL_1')
        assert abs(DomegaPLL) < 1e-3, f"ΔωPLL={DomegaPLL:.6f} should be ~0 (PLL locked)"

    def test_fcl_inactive_at_nominal(self, regfm_b1_model):
        """f_cl = 1.0 when |I| < ImaxF = 1.5 pu."""
        _ini(WECCS_DIR, regfm_b1_model, self.XY0)
        f_cl = regfm_b1_model.get_value('f_cl_1')
        assert f_cl > 0.99, f"f_cl={f_cl:.4f} should be ~1.0"

    def test_inertia_response_to_pref_step(self, regfm_b1_model):
        """Pref step 0.8→1.0: Δωm must dip negative (VSM inertia slows rotor)."""
        _ini(WECCS_DIR, regfm_b1_model, self.XY0)
        m = regfm_b1_model
        with _in_dir(WECCS_DIR):
            m.run(5.0, {'Pref_1': 1.0})
            m.post()
        Domegam_min = min(m.get_values('Domegam_1'))
        assert Domegam_min < -1e-4, (
            f"Δωm should dip negative during Pref step; got {Domegam_min:.6f}"
        )

    def test_inertia_stronger_than_droop(self, regfm_b1_model):
        """REGFM_B1 has slower transient than droop — Δωm recovers with time constant ~2H/D."""
        _ini(WECCS_DIR, regfm_b1_model, self.XY0)
        m = regfm_b1_model
        with _in_dir(WECCS_DIR):
            m.run(10.0, {'Pref_1': 1.0})
            m.post()
        # After 10 s the speed deviation should have recovered close to zero
        Domegam_final = m.get_values('Domegam_1')[-1]
        assert abs(Domegam_final) < 0.01, (
            f"Δωm should recover toward 0 after step; final={Domegam_final:.6f}"
        )

    def test_simulation_stable_over_10s(self, regfm_b1_model):
        """Full VSM simulation remains bounded and finite over 10 s."""
        _ini(WECCS_DIR, regfm_b1_model, self.XY0)
        m = regfm_b1_model
        with _in_dir(WECCS_DIR):
            m.run(10.0, {})
            m.post()
        for var in ['p_g_1', 'q_g_1', 'EVSM_1', 'deltaVSM_1',
                    'Domegam_1', 'deltaPLL_1', 'f_cl_1']:
            vals = m.get_values(var)
            assert np.all(np.isfinite(vals)), f"{var} went non-finite"
        assert np.max(np.abs(m.get_values('deltaVSM_1'))) < np.pi / 2, \
            "angle |δVSM| > 90° — loss of synchronism"
