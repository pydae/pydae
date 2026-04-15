# tests/conftest.py
"""
Pytest configuration and markers for selective testing.

Usage:
    uv run pytest                              # run everything
    uv run pytest -m parse                     # only parser tests
    uv run pytest -m "not slow"                # skip slow tests
    uv run pytest tests/core/                  # only core package
    uv run pytest tests/core/test_pendulum.py  # one file
    uv run pytest -k "test_build"              # match by name
"""
import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "parse: parser and validation tests")
    config.addinivalue_line("markers", "symbolic: Jacobian computation tests")
    config.addinivalue_line("markers", "codegen: C code generation tests")
    config.addinivalue_line("markers", "build: full compile pipeline (needs gcc)")
    config.addinivalue_line("markers", "model: end-to-end simulation tests")
    config.addinivalue_line("markers", "slow: tests that take > 5 seconds")
    config.addinivalue_line("markers", "bps: balanced power systems tests")
    config.addinivalue_line("markers", "uds: unbalanced distribution tests")
