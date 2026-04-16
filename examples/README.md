# pydae examples

Runnable examples that demonstrate the public API of the `pydae` packages.

| File | What it shows |
|---|---|
| `pendulum.py` | Damped pendulum DAE — the example from the project README. Builds the system, compiles the C solver, and simulates 21 s of motion. |
| `milano4ord_saturation.py` | Milano 4th-order synchronous-machine model built with `pydae.bps`. Steps mechanical power and plots speed, $e'_q$, and saturation factor to SVG. |
| `milano2ord_pm.py` | Milano 2nd-order synchronous-machine model built with `pydae.bps`. Steps mechanical power with increased damping and plots $\omega_1$ to SVG. |

## Running

From the repository root, with the workspace synced (`uv sync --all-packages`):

```bash
uv run python examples/pendulum.py
```

Or, with a regular activated environment:

```bash
python examples/pendulum.py
```
