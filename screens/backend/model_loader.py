"""
NeuroPath AI — Model Loader (lazy, cached)
==========================================
All models are loaded once and cached. WebSocket routes import
from here — no model is loaded until its screen is first opened.
"""

import os
import sys
import json
import pickle

from config import MODELS_DIR, HARDWARE_CORE

sys.path.insert(0, HARDWARE_CORE)

_cache: dict = {}


def _pkl(name: str):
    key = f"pkl:{name}"
    if key not in _cache:
        with open(os.path.join(MODELS_DIR, name), "rb") as f:
            _cache[key] = pickle.load(f)
    return _cache[key]


def _keras(name: str):
    key = f"h5:{name}"
    if key not in _cache:
        import tensorflow as tf
        tf.get_logger().setLevel("ERROR")
        from tensorflow.keras.models import load_model
        _cache[key] = load_model(os.path.join(MODELS_DIR, name))
    return _cache[key]


def _json(name: str) -> dict:
    key = f"json:{name}"
    if key not in _cache:
        with open(os.path.join(MODELS_DIR, name)) as f:
            _cache[key] = json.load(f)
    return _cache[key]


# ── Public loaders ─────────────────────────────────────────────────────────────

def load_fall():
    return {
        "model":     _keras("fall_detection_cnn.h5"),
        "config":    _json("fall_model_config.json"),
        "threshold": _json("fall_model_config.json").get("threshold", 0.65),
    }

def load_tremor():
    return {
        "clf":    _pkl("tremor_classifier.pkl"),
        "scaler": _pkl("tremor_scaler.pkl"),
        "le":     _pkl("tremor_label_encoder.pkl"),
        "config": _json("tremor_model_config.json"),
    }

def load_stress():
    return {
        "clf":    _pkl("stress_classifier.pkl"),
        "le":     _pkl("stress_label_encoder.pkl"),
        "config": _json("stress_model_config.json"),
    }

def load_atrophy():
    return {
        "gbr":    _pkl("atrophy_regressor.pkl"),
        "scaler": _pkl("atrophy_scaler.pkl"),
        "config": _json("atrophy_model_config.json"),
    }

def load_ptform():
    return {
        "model":  _keras("pt_form_lstm.h5"),
        "scaler": _pkl("pt_form_scaler.pkl"),
        "config": _json("pt_form_model_config.json"),
    }

def load_gesture():
    return {
        "clf": _pkl("gesture_brain.pkl"),
        "config": {
            "classes":              ["goodbye", "thanks", "welcome"],
            "motion_threshold":     1.5,
            "confidence_threshold": 0.75,
            "cooldown_s":           2.0,
        }
    }

def load_gesture_v2():
    return {
        "clf":    _pkl("gesture_brain_v2.pkl"),
        "scaler": _pkl("gesture_scaler_v2.pkl"),
        "le":     _pkl("gesture_label_encoder_v2.pkl"),
        "config": _json("gesture_model_config_v2.json"),
    }

def load_tremor_binary():
    return {
        "clf":    _pkl("tremor_binary_classifier.pkl"),
        "scaler": _pkl("tremor_binary_scaler.pkl"),
        "le":     _pkl("tremor_binary_label_encoder.pkl"),
        "config": _json("tremor_binary_model_config.json"),
    }

def load_fall_binary():
    return {
        "clf":    _pkl("fall_binary_classifier.pkl"),
        "scaler": _pkl("fall_binary_scaler.pkl"),
        "config": _json("fall_binary_model_config.json"),
    }
