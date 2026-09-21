# tests/core/test_control_builder.py
"""
Tests for the control system builder (``pydae.control.control_builder``),
which assembles a DAE ``sys_dict`` from a list of continuous control
blocks (``pydae.control.continuos``) described in a ``"control"``
section of a data dict, mirroring ``pydae.bps.bps_builder.BpsBuilder``
for standalone control systems.

Run selectively:
    uv run pytest tests/core/test_control_builder.py -v
"""
import pkgutil

import pytest

import pydae.control.continuos as continuos
from pydae.control.control_builder import ControlBuilder, BUILDERS


@pytest.mark.parse
def test_builders_cover_continuos_modules():
    """All continuous control blocks are registered under their module name.

    Derived from the package contents rather than hard-coded, so adding a
    block module without registering it in BUILDERS (or vice versa) fails
    here instead of silently leaving the block unreachable from HJSON.
    """
    modules = {
        name for _, name, _ in pkgutil.iter_modules(continuos.__path__)
        if not name.startswith('_')
    }
    assert set(BUILDERS) == modules


@pytest.mark.parse
def test_unknown_type_raises():
    cb = ControlBuilder({'control': [{'type': 'not_a_block', 'name': '1'}]}, use_casadi=True)
    with pytest.raises(ValueError):
        cb.construct('bad')


@pytest.mark.parse
def test_missing_type_or_name_raises():
    cb = ControlBuilder({'control': [{'name': '1'}]}, use_casadi=True)
    with pytest.raises(ValueError):
        cb.construct('bad')

    cb = ControlBuilder({'control': [{'type': 'lpf1'}]}, use_casadi=True)
    with pytest.raises(ValueError):
        cb.construct('bad')


@pytest.fixture
def control_model():
    """Build a standalone CasadiModel from a small multi-block 'control'
    description (one of each filter/controller family)."""
    from pydae.core.builder.casadi_builder import CasadiBuilder
    from pydae.core.model.casadi_model import CasadiModel

    data = {
        'control': [
            {'type': 'lpf1', 'name': 'meas_1', 'T': 0.1, 'u': 0.3},
            {'type': 'leadlag', 'name': 'comp_1', 'T_num': 1.0, 'T_den': 0.5, 'u': 0.4},
            {'type': 'washout', 'name': 'pss_1', 'T_wo': 5.0, 'u': 0.7},
        ]
    }

    cb = ControlBuilder(data, use_casadi=True)
    cb.construct('test_control_builder')

    model = CasadiModel(CasadiBuilder(cb.sys_dict).build())
    model.ini({}, cb.dae['xy_0_dict'])

    return model


@pytest.mark.model
class TestControlBuilder:

    def test_lpf1_steady_state_tracks_input(self, control_model):
        model = control_model
        assert abs(model.get_value('z_lpf1_meas_1') - 0.3) < 1e-6

    def test_leadlag_steady_state_is_transparent(self, control_model):
        model = control_model
        assert abs(model.get_value('z_leadlag_comp_1') - 0.4) < 1e-6

    def test_washout_steady_state_output_is_zero(self, control_model):
        model = control_model
        assert abs(model.get_value('z_washout_pss_1') - 0.0) < 1e-6

    def test_step_response_independent_blocks(self, control_model):
        """Each block reacts independently to its own input step."""
        model = control_model

        model.run(2.0, {'u_lpf1_meas_1': 1.0, 'u_washout_pss_1': 1.0})
        model.post()

        z_lpf1 = model.get_values('z_lpf1_meas_1')
        z_washout = model.get_values('z_washout_pss_1')

        assert z_lpf1[-1] > z_lpf1[0]                       # lpf1 rises toward the new input
        assert abs(max(z_washout) - (1.0 - 0.7)) < 0.02     # washout jumps by the step size
