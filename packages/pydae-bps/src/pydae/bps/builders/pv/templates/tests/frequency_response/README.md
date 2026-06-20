# frequency_response — sinusoidal sweeps → Bode

Drives sinusoidal references via the PPC `/sine` API across a band of
frequencies and fits the reference→POI gain and phase at each, producing a
closed-loop Bode of the plant + controller.

```bash
python run.py --mode p      # sine on P, Q constant
python run.py --mode q      # sine on Q, P constant
python run.py --mode pq90   # both, Q lagging P by 90°
```

**Modes**
- `p` — excites active power only (reactive held at base).
- `q` — excites reactive power only (active held at base).
- `pq90` — excites both. The `/sine` API resets each loop's phase to 0 on arm,
  so the Q sinusoid is armed a quarter-period (`1/4f`) after P, giving exactly
  90° relative phase with no controller change.

Defaults: `--freqs 0.1 0.2 0.3 0.4 0.5`, `--amp-mw 2.0` on `--base-mw 5.0`,
`--amp-mvar 1.0` on `--base-mvar 0.0`, 20 s/frequency, 4 s settle, 0.05 s poll,
`--skip 4` s of transient dropped before fitting.

**Outputs per mode** (`<mode>` ∈ p/q/pq90):
- `csv/<mode>_f*.csv` — `t, ref_p, poi_p, ref_q, poi_q`
- `plots/<mode>_timeseries_{p,q}.{svg,html}`
- `plots/<mode>_bode_{p,q}.{svg,html}` + `csv/<mode>_bode_{p,q}.csv`

With feedforward enabled the P magnitude stays ~flat (0 dB) to 0.5 Hz vs the
~−4.5 dB roll-off without it — the reference-tracking bandwidth is extended
well past the swept range.
