# Tri-Fusion MIL-Net

This repository provides the implementation and reproducibility materials for the Tri-Fusion MIL-Net model developed for case-level differentiation of breast adenosis and carcinoma on automated breast ultrasound (ABUS).

## Repository contents

- `model.py` — model architecture, including three EfficientNet-Lite0 2D encoders, a 3D residual Transformer encoder, bidirectional cross-attention, gated fusion, and Top-5 MIL aggregation.
- `train.py` — case-level model training and validation.
- `infer.py` — single-case or CSV-based inference.
- `heatmap.py` — Grad-CAM visualization and cross-attention matrices.
- `weights.pt` — trained checkpoint of the final Tri-Fusion MIL-Net model used for the reported evaluation.
- `requirements.txt` — software dependencies.
- `split_manifest.csv` — anonymized identifiers and dataset assignments for the training, validation, internal test, and external test cohorts.
- `internal_test_predictions.csv` — patient-level labels and prediction probabilities for the held-out internal test cohort.
- `external_test_predictions.csv` — patient-level labels and prediction probabilities for the independent external test cohort.

## Installation

```bash
pip install -r requirements.txt
```

## Input format

Each case is stored as one `.npz` file containing the following `float32` arrays:

| Key | Shape |
| --- | --- |
| `axial` | `[N_a, 1, H, W]` |
| `coronal` | `[N_c, 1, H, W]` |
| `sagittal` | `[N_s, 1, H, W]` |
| `patches` | `[N_p, 1, D, H, W]` |

Training and validation CSV files use the columns:

```text
case_id,path,label
```

where `label=0` indicates adenosis and `label=1` indicates carcinoma.

## Training

```bash
python train.py --train train.csv --val val.csv --out runs --pretrained
```

## Inference

```bash
python infer.py --input test.csv --weights weights.pt --out predictions.csv
python infer.py --input case.npz --weights weights.pt --out prediction.csv
```

## Visualization

```bash
python heatmap.py --input case.npz --weights weights.pt --out heatmaps
```

## Data availability

The original ABUS images cannot be publicly redistributed because of institutional and ethical restrictions governing clinical imaging data. All released identifiers are anonymized and cannot be linked to the original institutional patient identifiers.
