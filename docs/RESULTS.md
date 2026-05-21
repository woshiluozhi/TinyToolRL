# Verified Run Results

Command:

```powershell
python scripts\run_quickstart.py --device auto
python -m unittest discover -s tests
python scripts\demo_agent.py --checkpoint outputs\quickstart\sft\checkpoint.pt --n 3 --device auto
```

Environment observed during this run:

- Python 3.10.10
- PyTorch available
- CUDA available on this machine

Summary from `outputs/quickstart/summary.json`:

| Stage | Accuracy | Format rate | Mean reward |
|---|---:|---:|---:|
| SFT | 0.97 | 1.00 | 0.9949 |
| DPO | 0.97 | 1.00 | 0.9760 |
| RL | 0.96 | 1.00 | 0.9738 |

Per-task SFT accuracy:

| Task | Accuracy |
|---|---:|
| acceleration | 0.9412 |
| density | 1.0000 |
| kinetic_energy | 1.0000 |
| ohm_law | 1.0000 |
| pressure | 0.8333 |
| wave_speed | 1.0000 |

Qualitative demo:

```text
Task: wave_speed
Generated: CALL: CALC[4*10]
OBS: 40
FINAL: 40 m/s

Task: acceleration
Generated: CALL: CALC[81/9]
OBS: 9
FINAL: 9 m/s^2

Task: pressure
Generated: CALL: CALC[78/6]
OBS: 13
FINAL: 13 Pa
```

Interpretation:

- SFT is already strong in this controlled setting.
- DPO preserves accuracy but does not clearly improve aggregate performance in this run.
- RL preserves high format reliability but slightly lowers exact greedy accuracy; this is a useful teaching result, not a failure. It shows why RL post-training needs careful KL, reward shaping, and validation.

