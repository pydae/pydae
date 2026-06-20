# mitm_attack — Modbus man-in-the-middle

A MITM on the POI link falsifies the active-power holding register, so the PPC
reads a doctored POI value and drives the **real** plant to the wrong operating
point. The attack is armed when the cosim is launched (it is not controlled
from this driver); this script just records a window and plots the divergence
between what the controller believes and what the plant actually does.

Two falsification policies:

- **offset (default)** — add a small bias to the *real* POI reading
  (`cur·(1+frac)+abs`). The PPC sees a believable, slowly-diverging value and
  **settles at a small steady-state error** (e.g. ~5 %) without saturating —
  the plant ends up ~`frac` off target. This is the realistic, stealthy attack.
- **fixed (`--mitm-value-mw`)** — pin the readback at a constant. The PI error
  never closes, so the controller **rails to saturation** (the original
  over-power demo).

**1. Relaunch the cosim with the attack armed (Terminal A):**

```bash
cd ..    # modbus/
# offset mode — bias the real reading by +5%
sudo python3 run.py --no-cli --mitm --mitm-offset-frac 0.05 \
     --mitm-start-after 30 --mitm-duration 60

# fixed-spoof mode — pin readback at 5.0 MW (drives to saturation)
sudo python3 run.py --no-cli --mitm --mitm-value-mw 5.0 \
     --mitm-start-after 30 --mitm-duration 60
```

(`colinker.attacks.modbus_mitm`; uses `--explicit-redirect`, not ARP spoof.)

**2. Record across the attack window (Terminal B):**

```bash
python run.py --duration 90
```

**What to look for:**

- *offset mode:* `poi_mw` (PPC readback) tracks the target as usual, but
  `sim_p_line_POI_GRID` (plant ground truth) sits ~`frac` below it — the
  controller has been quietly biased to a wrong setpoint without any saturation
  or visible alarm. The steady gap between the red (PPC) and green (plant)
  traces is the attack's physical impact.
- *fixed mode:* `poi_mw` is pinned at the spoofed value while the plant
  ground truth diverges as the PI winds up to saturation.

Outputs: `csv/mitm_attack.csv`, `plots/mitm_attack.svg`, `…/.html`.

> Not run in the scaffolding pass — it needs the `--mitm` relaunch above.
