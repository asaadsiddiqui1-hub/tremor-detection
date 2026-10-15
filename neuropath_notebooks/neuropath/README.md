# NeuroPath AI — Model Training Guide

## Folder structure to create on your machine

```
neuropath/
├── notebooks/          ← all 4 .ipynb files go here
├── models/             ← created automatically when you run notebooks
└── data/
    ├── sisfall/        ← from your archive.zip (Three Classes/ contents)
    │   ├── x_train_3
    │   ├── y_train_3
    │   ├── x_val_3
    │   ├── y_val_3
    │   ├── x_test_3
    │   ├── y_test_3
    │   └── weights_3.txt
    ├── imu-hand-tremor-parkinsons.csv   ← directly in data/
    ├── wesad/          ← OPTIONAL: PhysioNet WESAD subject .pkl files
    └── pt_exercise/    ← OPTIONAL: physical therapy CSV files
```

---

## Step 0 — Install requirements (run once)

```bash
pip install tensorflow scikit-learn numpy pandas matplotlib seaborn scipy
```

---

## Step 1 — Data setup

### SisFall (Fall Detection)
1. Open your `archive.zip`
2. Inside it find the folder `Three Classes/`
3. Copy all 7 files from that folder into `neuropath/data/sisfall/`

### Tremor dataset
- Your `imu-hand-tremor-parkinsons.csv` goes directly in `neuropath/data/`

### Stress/HRV (optional WESAD)
- Without WESAD: notebook auto-generates synthetic HRV data — works fine for demo
- With WESAD: download from https://physionet.org/content/wesad/1.0.0/
  Place the S2.pkl, S3.pkl... files in `neuropath/data/wesad/`

### PT Form + Atrophy
- No external data needed — synthetic generation built in

---

## Step 2 — Run notebooks in order

| Notebook | What it trains | Output files |
|----------|---------------|--------------|
| `01_fall_detection_train.ipynb` | 1D-CNN fall detector | `fall_detection_cnn.h5`, `fall_model_config.json` |
| `02_tremor_classifier_train.ipynb` | SVM tremor classifier | `tremor_classifier.pkl`, `tremor_model_config.json` |
| `03_stress_hrv_train.ipynb` | RF stress predictor | `stress_classifier.pkl`, `stress_model_config.json` |
| `04_atrophy_and_pt_form_train.ipynb` | GBR atrophy + LSTM PT form | `atrophy_regressor.pkl`, `pt_form_lstm.h5`, configs |

**Training times (CPU, no GPU):**
- Notebook 01: ~2–3 hours (14 epochs × ~10 min)
- Notebook 02: ~5 minutes
- Notebook 03: ~3 minutes
- Notebook 04: ~15 minutes

**With GPU:** Notebook 01 drops to ~20 minutes.

---

## Step 3 — Verify outputs

After running all notebooks, `models/` should contain:

```
models/
├── fall_detection_cnn.h5
├── fall_model_config.json
├── fall_training_curves.png
├── fall_evaluation.png
├── tremor_classifier.pkl
├── tremor_scaler.pkl
├── tremor_label_encoder.pkl
├── tremor_model_config.json
├── stress_classifier.pkl
├── stress_label_encoder.pkl
├── stress_model_config.json
├── atrophy_regressor.pkl
├── atrophy_scaler.pkl
├── atrophy_model_config.json
├── pt_form_lstm.h5
├── pt_form_scaler.pkl
└── pt_form_model_config.json
```

---

## Step 4 — Next step

Once all model files exist, we build:
- `05_gesture_data_collection.py` — records your own hand gestures to CSV
- `06_gesture_classifier_train.ipynb` — trains gesture translator on your recordings
- `app/main.py` — the PyQt5 application with all 7 screens

---

## Notebook 01 troubleshooting

**"Missing files" error:**
→ Make sure you copied from `Three Classes/` not from the zip root

**"CUDA errors" on startup:**
→ Safe to ignore — these are TensorFlow GPU warnings, training continues on CPU

**Training very slow:**
→ Normal on CPU. Reduce `epochs=15` to `epochs=5` for a quick test run first.
→ Or reduce `batch_size=512` to `batch_size=256` to use less RAM.

**`weights_3.txt` loads 3 values but we use index 0 and 1:**
→ This is correct. The dataset has a 3-class version; we use it as binary (Normal vs Fall).
   Index 2 is unused and discarded automatically.
