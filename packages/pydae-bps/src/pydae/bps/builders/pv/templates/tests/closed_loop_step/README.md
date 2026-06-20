# closed_loop_step — PI gain-design step response

Runs the P/Q step with the **configured PI gains** (read from the plant JSON)
so you can design and compare loop tunings.

```bash
python run.py                              # configured gains, feedforward on
python run.py --name kip05 --K_ip 0.5      # override a gain, distinct outputs
python run.py --no-feedforward --name noff # feedforward A/B
```

Sequence: pre-settle at P=2 MW → step P 2→3 MW → step Q 0→1 Mvar, recording PPC
readback + pydae ground truth.

`--name` sets the output basename so sweeps don't clobber each other. Gain
overrides: `--K_pp --K_ip --K_pq --K_iq --ctrl_dt_s` (min 0.1 s, API-enforced).

With feedforward on, the P rise is much faster than with the integral alone;
use this test to trade off rise time vs overshoot per loop.

Outputs: `csv/<name>.csv`, `plots/<name>.svg`, `…/.html`.
