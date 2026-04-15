# pydae examples

Runnable examples that demonstrate the public API of the `pydae` packages.

| File | What it shows |
|---|---|
| `pendulum.py` | Damped pendulum DAE — the example from the project README. Builds the system, compiles the C solver, and simulates 21 s of motion. |

## Running

From the repository root, with the workspace synced (`uv sync --all-packages`):

```bash
uv run python examples/pendulum.py
```

Or, with a regular activated environment:

```bash
python examples/pendulum.py
```
