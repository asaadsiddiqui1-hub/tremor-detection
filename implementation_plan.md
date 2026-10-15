# Tremor Classifier — Binary Retrain Plan (Extreme Tremor vs. Normal)

## Problem Summary

The current tremor model is a **3-class SVM** (mild / moderate / high), trained entirely on Parkinson's patient data — with **no "normal" class at all**. When deployed, even a resting healthy hand gets forced into one of those three tremor buckets, making it impossible to distinguish "no tremor" from "tremor present."

The goal is to retrain a **binary classifier** (tremor vs. normal) that can correctly say "no tremor" when the hand is calm, and "tremor" when pathological oscillation is detected.

---

## Design Decisions (confirmed with you)

| Question | Answer |
|---|---|
| Classification goal | Binary: **tremor vs. normal** |
| Tremor class data | **Merge all 3 labels** (mild + moderate + high) → single "tremor" class |
| Normal class data | **Record ourselves** wearing the MPU6050 glove (~15–20 min of varied daily movements) |
| Classifier | **Try both SVM and Random Forest**, pick the winner |
| Priority metric | **Balanced accuracy** (equal weight to precision and recall) |
| Model filename | **New file** — do NOT overwrite `tremor_classifier.pkl` |

---

## Why This Fixes the Root Cause

The MPU6050 is extremely sensitive and picks up micro-vibrations even in healthy hands. The original model had **no reference for "calm"** — it learned only the differences *between* tremor severities. A binary model trained with real normal data gives the model a learned baseline for what "calm" looks like at your specific sensor sensitivity.

> [!IMPORTANT]
> **~15–20 minutes of normal recording at 50 Hz = ~45,000–60,000 samples.** That matches the tremor dataset size (4,502 rows), especially after windowing. This is enough data for a robust SVM/RF binary classifier.

---

## How Much Data to Record

| Activity | Duration | Why |
|---|---|---|
| Resting hand on table, completely still | 3–4 min | Tests pure sensor baseline noise |
| Slow hand rotation (supination/pronation) | 2–3 min | Voluntary slow movement |
| Normal typing on keyboard | 3–4 min | High-frequency voluntary motion |
| Pointing, reaching, picking up objects | 3–4 min | Purposeful arm movements |
| Walking with arm swinging naturally | 3–4 min | Locomotion-coupled hand motion |

**Target: 15–20 minutes total.** You can split across 2–3 people if available to get diversity.

---

## Proposed Changes

### Phase 1 — Data Collection Script

#### [NEW] `record_normal_data.py` (in `neuropath_notebooks/neuropath/`)

A standalone Python script that:
- Connects to the live ESP32/MPU6050 via serial (same `SerialReader` the app uses)
- Records a timed session of raw `xAcc, yAcc, zAcc, xGyro, yGyro, zGyro` at 50 Hz
- Appends `label = 0` to every row (0 = normal)
- Saves to `data/normal_hand_data.csv`
- Shows a real-time countdown so you know how long to hold each pose

---

### Phase 2 — Retrain Notebook

#### [MODIFY] `notebooks/02_tremor_classifier_train.ipynb`

We will add **new cells at the end** (keeping all existing cells intact) that:

1. **Load and merge data**
   - Load existing `imu-hand-tremor-parkinsons.csv` (labels 1/2/3 → remap to `1 = tremor`)
   - Load new `normal_hand_data.csv` (label `0 = normal`)
   - Concatenate into a single binary dataset

2. **Window and extract features**
   - Reuse the existing `prepare_tremor_input()` / FFT feature pipeline (no changes to feature engineering)
   - Window size stays at 100 samples (2 s at 50 Hz) with 50% overlap

3. **Train SVM (RBF) vs. Random Forest**
   - Both trained with `StandardScaler`
   - Evaluate with 5-fold stratified cross-validation
   - Report: accuracy, precision, recall, F1, confusion matrix

4. **Pick the winner** and save to:
   - `models/tremor_binary_classifier.pkl`
   - `models/tremor_binary_scaler.pkl`
   - `models/tremor_binary_label_encoder.pkl`
   - `models/tremor_binary_model_config.json`

5. **Plot evaluation charts** saved to `models/tremor_binary_evaluation.png`

---

### Phase 3 — Backend Integration

#### [MODIFY] `screens/backend/model_loader.py`

Add a new `load_tremor_binary()` loader function that loads the four new binary model files.

#### [NEW] `screens/backend/ws_routes/tremor_binary.py`

A new WebSocket route that:
- Uses `load_tremor_binary()` 
- Outputs a simple `{ "result": "tremor" | "normal", "confidence": <int> }` payload
- Keeps the same motion-gate logic (skip inference when `axis_motion < 0.005`)
- The label map is simply `{0: "normal", 1: "tremor"}`

#### [MODIFY] `screens/backend/main.py`

Register the new `/ws/tremor-binary` WebSocket endpoint.

---

## Verification Plan

### Step 1 — Data Quality Check
After recording, run a quick plot in the notebook to confirm:
- Normal data has low gyro variance (< 5 deg/s RMS)
- Tremor data has visible oscillation peaks in the 3–12 Hz band

### Step 2 — Model Evaluation
- Cross-validation accuracy ≥ 90% (SVM and/or RF)
- Confusion matrix shows < 10% false positives on the normal class
- Test on a held-out 20% split

### Step 3 — Live App Test
- Connect the glove, open the new binary tremor screen
- With hand at rest → should show "normal" consistently
- Simulate tremor (rapid voluntary shaking) → should show "tremor"

---

## Open Questions

> [!NOTE]
> **Column format for recording:** Your glove streams 7 channels (`accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z, ppg_raw`), but the tremor model uses only 6 (`xAcc, yAcc, zAcc, xGyro, yGyro, zGyro`). The recording script will drop `ppg_raw` automatically to match the training CSV format.

> [!NOTE]
> **COM port:** The recording script will need your glove's COM port (e.g., `COM3`). It will default to what's in `config.py` but can be overridden via a command-line argument.
