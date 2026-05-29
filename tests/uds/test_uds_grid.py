"""Backend parity tests for the unbalanced distribution builder.

A minimal 4-wire network (ideal AC source + ZIP load joined by one line) is
assembled through ``UdsBuilder`` with both the SymPy and CasADi backends, then
built/initialised/simulated. The two backends must agree on the steady-state
node voltages.

A second network adds a Dyn11 transformer with ``monitor: true`` to exercise
``add_trafo_monitors``, which must emit real/imag/magnitude branch-current
outputs through the backend-agnostic real-form arrays (not the SymPy-only
complex ``self.I_lines``).

Run selectively:
    uv run pytest tests/uds/test_uds_grid.py -v
"""
import numpy as np
import pytest


def _network():
    return {
        "system": {"S_base": 1e6, "K_p_agc": 0.01, "K_i_agc": 0.01, "K_xif": 0.01},
        "buses": [
            {"name": "A1", "U_kV": 0.4, "N_nodes": 4, "monitor": True},
            {"name": "A2", "U_kV": 0.4, "N_nodes": 4, "monitor": True},
        ],
        "lines": [
            {"bus_j": "A1", "bus_k": "A2", "R": 0.1, "X": 0.1, "N_branches": 4},
        ],
        "sources": [
            {"type": "ac3ph4w_ideal", "bus": "A1"},
        ],
        "loads": [
            {"bus": "A2", "type": "3P+N", "model": "ZIP", "kVA": 10.0, "pf": 0.95},
        ],
    }


# node voltage variable names shared by both backends
_V_NAMES = [f"V_{bus}_{node}_{ri}"
            for bus in ("A1", "A2")
            for node in range(4)
            for ri in ("r", "i")]


@pytest.mark.uds
class TestUdsConstructBothBackends:
    """construct() must produce an identically-shaped, square DAE on both backends."""

    def test_sympy_and_casadi_shapes_match(self):
        from pydae.uds import UdsBuilder

        g_sym = UdsBuilder(_network())
        g_sym.construct("uds_grid_shape_sym")
        sd = g_sym.sys_dict

        g_ca = UdsBuilder(_network(), use_casadi=True)
        g_ca.construct("uds_grid_shape_ca")
        sdc = g_ca.sys_dict

        # DAE is square in each phase
        assert len(sd["g_list"]) == len(sd["y_ini_list"])
        assert len(sd["f_list"]) == len(sd["x_list"])

        # both backends agree on dimensions
        for key in ("g_list", "y_ini_list", "y_run_list", "x_list", "f_list"):
            assert len(sd[key]) == len(sdc[key]), f"{key} length differs between backends"

        # both expose the same variable / parameter / input names
        assert {str(s) for s in sd["y_ini_list"]} == {str(s) for s in sdc["y_ini_list"]}
        assert set(sd["params_dict"]) == set(sdc["params_dict"])
        assert set(sd["u_run_dict"]) == set(sdc["u_run_dict"])


@pytest.mark.uds
@pytest.mark.model
class TestUdsBackendParity:
    """Full build/ini/run on both backends must converge to the same operating point."""

    def test_sympy_casadi_voltage_parity(self):
        from pydae.uds import UdsBuilder
        from pydae.core import Builder, Model
        from pydae.core.builder import CasadiBuilder, CasadiModel

        # --- SymPy backend: symbolic -> C -> trapezoidal runtime ---
        g_sym = UdsBuilder(_network())
        g_sym.construct("uds_grid_parity_sym")
        Builder(g_sym.sys_dict, target="ctypes", sparse=False).build()
        m = Model("uds_grid_parity_sym")
        m.ini({}, g_sym.dae["xy_0_dict"])
        m.run(1.0, {})
        m.post()

        # --- CasADi backend: SX graph -> IDAS runtime ---
        g_ca = UdsBuilder(_network(), use_casadi=True)
        g_ca.construct("uds_grid_parity_ca")
        mc = CasadiModel(CasadiBuilder(g_ca.sys_dict).build())
        mc.ini({}, xy_0=g_ca.dae["xy_0_dict"])
        mc.run(1.0, {})
        mc.post()

        # all node voltages agree between backends
        for name in _V_NAMES:
            assert m.get_value(name) == pytest.approx(mc.get_value(name), abs=1e-6), name

        # sanity: source holds bus A1 near the nominal phase-neutral voltage,
        # and the loaded bus A2 sits slightly lower
        v_nom = 400.0 / np.sqrt(3)
        assert m.get_value("V_A1_an") == pytest.approx(v_nom, abs=2.0)
        assert m.get_value("V_A2_an") < m.get_value("V_A1_an")

        # trajectories are finite
        assert np.isfinite(m.X).all() and np.isfinite(m.Y).all()


def _trafo_network():
    """20 kV (delta) -> 0.4 kV (wye) Dyn11 transformer feeding a ZIP load."""
    return {
        "system": {"S_base": 1e6, "K_p_agc": 0.01, "K_i_agc": 0.01, "K_xif": 0.01},
        "buses": [
            {"name": "MV", "U_kV": 20.0, "N_nodes": 4, "monitor": True},
            {"name": "LV", "U_kV": 0.4, "N_nodes": 4, "monitor": True},
        ],
        "transformers": [
            {"bus_j": "MV", "bus_k": "LV", "S_n_kVA": 100, "U_j_kV": 20, "U_k_kV": 0.4,
             "R_cc_pu": 0.01, "X_cc_pu": 0.04, "connection": "Dyn11", "monitor": True},
        ],
        "sources": [{"type": "ac3ph4w_ideal", "bus": "MV"}],
        "loads": [{"bus": "LV", "type": "3P+N", "model": "ZIP", "kVA": 10.0, "pf": 0.95}],
    }


# the ideal source defaults to 400 V L-L; drive it at 20 kV so the transformer
# primary is energised at its rated voltage.
_MV_EMF = {f"e_{ph}o_m_MV": 20000.0 / np.sqrt(3) for ph in ("a", "b", "c")}


@pytest.mark.uds
class TestUdsTransformerConstruct:
    """A monitored Dyn11 transformer must construct identically on both backends."""

    def test_trafo_monitor_names_match(self):
        from pydae.uds import UdsBuilder

        g_sym = UdsBuilder(_trafo_network())
        g_sym.construct("uds_trafo_shape_sym")
        sd = g_sym.sys_dict

        g_ca = UdsBuilder(_trafo_network(), use_casadi=True)
        g_ca.construct("uds_trafo_shape_ca")
        sdc = g_ca.sys_dict

        # backends agree on DAE size and the full h_dict output set
        for key in ("g_list", "y_ini_list", "x_list", "f_list"):
            assert len(sd[key]) == len(sdc[key]), f"{key} length differs between backends"
        assert set(sd["h_dict"]) == set(sdc["h_dict"])

        # transformer current monitors exist with real/imag/magnitude on both
        # backends (regression: add_trafo_monitors used to AttributeError on CasADi)
        i_t = {k for k in sd["h_dict"] if k.startswith("i_t_")}
        assert i_t == {k for k in sdc["h_dict"] if k.startswith("i_t_")}
        # 7 branches (3 delta primary + 4 wye secondary) x {r, i, m}
        assert len(i_t) == 7 * 3
        assert sum(k.endswith("_m") for k in i_t) == 7


@pytest.mark.uds
@pytest.mark.model
class TestUdsTransformerParity:
    """Full build/ini/run parity for the transformer network, incl. branch currents."""

    def test_trafo_voltage_and_current_parity(self):
        from pydae.uds import UdsBuilder
        from pydae.core import Builder, Model
        from pydae.core.builder import CasadiBuilder, CasadiModel

        g_sym = UdsBuilder(_trafo_network())
        g_sym.construct("uds_trafo_parity_sym")
        Builder(g_sym.sys_dict, target="ctypes", sparse=False).build()
        m = Model("uds_trafo_parity_sym")
        m.ini(_MV_EMF, g_sym.dae["xy_0_dict"])
        m.run(1.0, _MV_EMF)
        m.post()

        g_ca = UdsBuilder(_trafo_network(), use_casadi=True)
        g_ca.construct("uds_trafo_parity_ca")
        mc = CasadiModel(CasadiBuilder(g_ca.sys_dict).build())
        mc.ini(_MV_EMF, xy_0=g_ca.dae["xy_0_dict"])
        mc.run(1.0, _MV_EMF)
        mc.post()

        # node voltages agree between backends
        v_names = [f"V_{b}_{n}_{ri}" for b in ("MV", "LV") for n in range(4) for ri in ("r", "i")]
        for name in v_names:
            assert m.get_value(name) == pytest.approx(mc.get_value(name), abs=1e-6), name

        # transformer current monitors (r/i/m) agree between backends
        i_t = [k for k in g_sym.sys_dict["h_dict"] if k.startswith("i_t_")]
        assert i_t, "no transformer current monitors emitted"
        for name in i_t:
            assert m.get_value(name) == pytest.approx(mc.get_value(name), abs=1e-6), name

        # physical sanity: 20 kV primary stepped down to ~0.4 kV balanced secondary
        assert m.get_value("V_MV_an") == pytest.approx(20000.0 / np.sqrt(3), abs=5.0)
        for ph in ("a", "b", "c"):
            assert m.get_value(f"V_LV_{ph}n") == pytest.approx(400.0 / np.sqrt(3), abs=5.0)

        assert np.isfinite(m.X).all() and np.isfinite(m.Y).all()
