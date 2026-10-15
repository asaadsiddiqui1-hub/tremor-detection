"""
NeuroPath AI — Central Configuration
======================================
Adjust SAMPLE_RATE_HZ to change the acquisition rate globally.
All window sizes, feature extractors, and WebSocket tick rates
are derived from this single value.
"""

import os

# ── Hardware / Sampling ────────────────────────────────────────────────────────
SAMPLE_RATE_HZ: int = 50          # ← CHANGE THIS to adjust the Hz rate globally
DEFAULT_COM_PORT: str = "COM3"
BAUD_RATE: int = 115200
BUFFER_SECONDS: float = 65.0      # rolling buffer (needs ≥ 60 s for stress HRV)

# ── Derived window sizes (samples) ────────────────────────────────────────────
WIN_FALL   = 512                   # ~10.24 s at 50 Hz  (fixed by CNN input shape)
WIN_TREMOR = int(SAMPLE_RATE_HZ * 2)     # 2-second window
WIN_STRESS = int(SAMPLE_RATE_HZ * 60)    # 60-second window for HRV
WIN_PT     = int(SAMPLE_RATE_HZ * 2)     # 2-second window
WIN_SENSOR = int(SAMPLE_RATE_HZ * 4)     # display last 4 s on sensor screen
WIN_GESTURE= int(SAMPLE_RATE_HZ * 2)     # 2-second gesture window

# ── Inference intervals (milliseconds) ────────────────────────────────────────
INFER_INTERVAL_FALL    = 500
INFER_INTERVAL_TREMOR  = 500
INFER_INTERVAL_STRESS  = 2000
INFER_INTERVAL_ATROPHY = 99999    # button-triggered only; live plot still ticks
INFER_INTERVAL_PT      = 500
INFER_INTERVAL_SENSOR  = 100      # raw sensor stream, fast
INFER_INTERVAL_GESTURE = 500

# ── Paths ─────────────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))

MODELS_DIR = os.path.join(
    _PROJECT_ROOT, "neuropath_notebooks", "neuropath", "models"
)

# Path to hardware core (serial_reader + feature_adapter)
HARDWARE_CORE = os.path.join(
    _PROJECT_ROOT, "neuropath_notebooks", "neuropath", "app", "core"
)

# Evaluation images served to frontend
EVAL_IMAGES = {
    "fall":    os.path.join(MODELS_DIR, "fall_evaluation.png"),
    "tremor":  os.path.join(MODELS_DIR, "tremor_evaluation.png"),
    "stress":  os.path.join(MODELS_DIR, "stress_evaluation.png"),
    "atrophy": os.path.join(MODELS_DIR, "atrophy_evaluation.png"),
    "ptform":  os.path.join(MODELS_DIR, "pt_form_evaluation.png"),
}
