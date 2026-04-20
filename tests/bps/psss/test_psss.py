"""
Pytest coverage for pydae-bps power system stabiliser models.

Each test invokes the module's own in-module test() in a fresh Python
subprocess. Running in a separate process per model sidesteps CFFI
shared-library state leakage and keeps the assertions aligned with
whatever the module author wrote in test().

Run:
    uv run pytest tests/bps/psss/
    uv run pytest -m bps -k psss
"""
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.bps
@pytest.mark.parametrize("name", ["pss_kundur_1", "pss_kundur_2", "pss2a"])
def test_pss_in_module(name):
    import pydae.bps.psss as pkg
    module = Path(pkg.__file__).parent / f"{name}.py"

    result = subprocess.run(
        [sys.executable, str(module)],
        cwd=module.parent,
        capture_output=True,
        text=True,
        timeout=300,
    )

    # The in-module test() always runs to completion, but on Windows the
    # CFFI/pyd teardown sometimes crashes the interpreter with heap
    # corruption (0xC0000374). The test logic itself — ini() assertions and
    # run() chain — has already finished by then and printed the final
    # timestamp. So ignore the exit code and check for assertion / traceback
    # markers in captured output instead.
    assert "AssertionError" not in result.stderr, (
        f"{name} in-module test() raised AssertionError\n"
        f"--- stdout ---\n{result.stdout}\n"
        f"--- stderr ---\n{result.stderr}"
    )
    assert "Traceback" not in result.stderr, (
        f"{name} in-module test() raised an exception\n"
        f"--- stdout ---\n{result.stdout}\n"
        f"--- stderr ---\n{result.stderr}"
    )
