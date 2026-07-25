"""Tests for ssa.damp CSV / portable-HTML export options.

Uses a plain 4x4 state matrix with two lightly-damped oscillatory mode pairs
(no Model needed) so the export paths are exercised without a compile step.

Run:  uv run pytest tests/core/test_ssa_damp_export.py -v
"""
import re

import numpy as np
import pytest

from pydae.ssa import ssa

pytestmark = pytest.mark.core


def _A():
    # block-diagonal: two damped oscillators, eigenvalues -0.2±5j and -1±12j
    def blk(sigma, omega):
        return np.array([[sigma, omega], [-omega, sigma]])
    A = np.zeros((4, 4))
    A[:2, :2] = blk(-0.2, 5.0)
    A[2:, 2:] = blk(-1.0, 12.0)
    return A


def test_csv_export(tmp_path):
    p = tmp_path / "damp.csv"
    ss = ssa.damp(_A(), sort='damp', csv=str(p))
    assert p.exists()
    header = p.read_text().splitlines()[0]
    assert header.split(',')[:5] == ['Mode', 'Real', 'Imag', 'Freq', 'Damp']
    # 4 eigenvalues -> 4 data rows
    assert len(p.read_text().splitlines()) == 5
    assert 'eigvalues' in ss


def test_html_export_is_self_contained(tmp_path):
    pytest.importorskip("plotly")
    p = tmp_path / "damp.html"
    ssa.damp(_A(), sort='damp', html=str(p))
    assert p.exists()
    h = p.read_text()

    # no external resources -> portable/offline
    ext = re.findall(r'<script[^>]*src=["\']([^"\']+)["\']', h)
    assert ext == [], f"unexpected external scripts: {ext}"
    # plotly.js is embedded (large inline bundle) and a plot is drawn
    assert p.stat().st_size > 1_000_000
    assert 'Plotly.newPlot' in h
    # dependency-free sortable table with the damp_report columns
    assert 'function sortTable' in h
    for col in ('Mode', 'Real', 'Imag', 'Freq', 'Damp'):
        assert f'>{col} ' in h
    assert h.count('<tr>') == 5      # 1 header row + 4 data rows


def test_html_true_uses_default_name(tmp_path, monkeypatch):
    pytest.importorskip("plotly")
    monkeypatch.chdir(tmp_path)
    ssa.damp(_A(), html=True)
    assert (tmp_path / "damp.html").exists()
