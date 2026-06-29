# mitm_attack — man-in-the-middle on the POI link

A MITM on the POI link falsifies the active-power measurement, so the PPC
reads a doctored POI value and drives the **real** plant to the wrong operating
point. The attack is armed when the cosim is launched (it is not controlled
from this driver); this script just records a window and plots the divergence
between what the controller believes and what the plant actually does.

The falsified field and the protocol-aware MITM module depend on the stack:
a Modbus holding register (`colinker.attacks.modbus_mitm`), an IEC 104
`M_ME_NC_1` short float (`colinker.attacks.iec104_mitm`), or a DNP3 Analog
Input float (`colinker.attacks.dnp3_mitm`). The benchmark's own `run.py`
wires the right one — the steps below are identical across protocols.

Falsification policies (set on the cosim's `run.py`):

- **over-power offset** (`--mitm-offset-frac` negative, e.g. `-0.0909`) — bias
  the *real* reading down so the PPC drives the plant **above** target. The
  readback reads exactly on-target while the real plant runs ~10 % over — a
  quiet grid-code breach with no saturation or alarm. *The realistic, stealthy
  attack.*
- **under-power offset** (positive frac, e.g. `0.05`) — same idea, plant
  settles ~`frac` below target.
- **fixed (`--mitm-value-mw`)** — pin the readback at a constant. The PI error
  never closes, so the controller **rails to saturation**.

**1. Relaunch the cosim with the attack armed (Terminal A):**

```bash
cd ..    # the protocol folder (modbus/ | iec104/ | dnp3/)
# over-power offset — plant driven ~10% above target, readback stays on-target
sudo python3 run.py --no-cli --mitm --mitm-offset-frac -0.0909 \
     --mitm-start-after 30 --mitm-duration 120

# fixed-spoof mode — pin readback at 5.0 MW (drives to saturation)
sudo python3 run.py --no-cli --mitm --mitm-value-mw 5.0 \
     --mitm-start-after 30 --mitm-duration 60
```

(The protocol-specific MITM module is selected by `run.py`; all use
`--explicit-redirect`, not ARP spoof.)

**2. Record the attack (Terminal B) — self-timing, no manual alignment:**

```bash
python tests/mitm_attack/run.py                 # auto-detects the window
python tests/mitm_attack/run.py --target-mw 12  # park the plant at 12 MW first
python tests/mitm_attack/run.py --blind 90      # old fixed-window mode
```

The driver records continuously, watches the live readback-vs-ground-truth
divergence, detects when the attack **arms**, then records `--record` more
seconds to span the steady state and the recovery after it disarms — so it
captures a clean episode no matter exactly when you start it (as long as it is
running before the attack disarms). `--target-mw` parks the plant at a chosen
operating point first (e.g. to match a reference case).

**What to look for:**

- *offset mode:* `poi_mw` (PPC readback) tracks the target as usual, but
  `sim_p_line_POI_GRID` (plant ground truth) sits ~`frac` off it — the
  controller has been quietly biased to a wrong setpoint without any saturation
  or visible alarm. The steady gap between the readback and plant traces is the
  attack's physical impact.
- *fixed mode:* `poi_mw` is pinned at the spoofed value while the plant
  ground truth diverges as the PI winds up to saturation.

Outputs: `csv/mitm_attack.csv`, `plots/mitm_attack.svg`, `…/.html`.

> Not run in the scaffolding pass — it needs the `--mitm` relaunch above.
