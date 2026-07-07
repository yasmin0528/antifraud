# Smoke Training Checklist

Use this checklist before starting long experiments.

## 1. Static Entry Checks

- `train.py` can be parsed by Python.
- `utils/config.py` can be imported by Python.
- Smoke configs exist:
  - `configs/smoke/cryptopia.yaml`
  - `configs/smoke/aml.yaml`

Recommended commands:

```bash
python -m py_compile refactored/train.py refactored/utils/config.py
python -m compileall refactored
```

## 2. Cryptopia Smoke

Run:

```bash
python train.py --config configs/default.yaml --config configs/dataset/cryptopia.yaml --config configs/smoke/cryptopia.yaml
```

Expected behavior:

- loads graph data
- builds models
- completes 1 epoch
- runs evaluation
- exports result artifacts

Expected artifacts:

- `results/results.csv`
- `results/metadata.json`
- `results/node_scores.csv`
- `results/edge_trace.csv`
- `results/ca3_memory.pt`
- `results/group_meta.json`

## 3. AML Smoke

Run:

```bash
python train.py --config configs/default.yaml --config configs/dataset/aml.yaml --config configs/smoke/aml.yaml
```

Expected behavior:

- allows first-time preprocessing
- completes 1 epoch
- runs evaluation
- exports result artifacts

Expected artifacts:

- `results/results.csv`
- `results/metadata.json`
- `results/sample_scores.csv`
- `results/edge_trace.csv`
- `results/ca3_memory.pt`
- `results/group_meta.json`

## 4. Smoke Pass Criteria

Smoke is considered passed only if:

- no syntax/import error occurs before entering trainer code
- no shape mismatch occurs during forward or loss computation
- no group metadata error occurs
- no explainability export step crashes
- LLM API unavailability falls back cleanly instead of aborting training

## 5. Current Environment Note

If the current machine has no usable Python interpreter, static checks and smoke runs must be executed later in a Python-enabled environment.
