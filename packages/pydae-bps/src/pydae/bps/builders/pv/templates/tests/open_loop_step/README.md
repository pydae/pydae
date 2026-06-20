# open_loop_step — plant + communications time response

First settles **closed-loop** at the pre-step operating point, then sets all PI
gains to **zero** with **feedforward** on, so each inverter setpoint is driven
open-loop by ΔT/N with no feedback correction. The POI response is then the raw
plant + transport behaviour, and the step amplitude reveals the open-loop
**DC gain** of each channel. (Settling closed-loop first fixes the initial
condition, since the velocity-form feedforward is incremental.)

```bash
python run.py                 # writes csv/open_loop_step.csv + plots/
```

Sequence: closed-loop settle at P=2 MW, Q=0 → drop gains to 0 → step P 2→3 MW →
step Q 0→1 Mvar. Configured gains are restored on exit.

**What to look for**
- **P** ≈ unity gain: settles near the commanded value quickly (small offset,
  no integral).
- **Q** DC gain ≠ 1: the network amplifies reactive power, so a Q command
  settles above its target at the POI. The step amplitude *is* the reactive DC
  gain — it sets the per-loop feedforward scale `ff_gain_q ≈ 1/gain`, and shows
  why the closed-loop integral is needed to null the offset.

**Arrivals (optional):** to overlay per-inverter wire arrival times, launch the
cosim with `--arrivals-dir tests/open_loop_step/arrivals`.

Outputs: `csv/open_loop_step.csv`, `plots/open_loop_step.svg`, `…/.html`.
