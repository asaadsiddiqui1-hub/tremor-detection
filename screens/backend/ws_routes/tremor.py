"""
WebSocket — Tremor Severity
Streams: severity label + confidence + band-power metrics + accel chart.
"""

import sys
import asyncio
import numpy as np
from fastapi import WebSocket, WebSocketDisconnect

sys.path.insert(0, "..")
from config import WIN_TREMOR, INFER_INTERVAL_TREMOR, HARDWARE_CORE, SAMPLE_RATE_HZ
sys.path.insert(0, HARDWARE_CORE)
from feature_adapter import (                      # noqa: E402
    prepare_tremor_input, extract_fft_features, dominant_axis,
    _TREMOR_FEATURE_KEYS,
)

import serial_bridge as bridge
import model_loader as ml

SEVERITY_MAP = {1: "mild", 2: "moderate", 3: "high"}
SEVERITY_COLOR = {
    "none":     "#6e7681",
    "mild":     "#3fb950",
    "moderate": "#d29922",
    "high":     "#f85149",
}
CONFIDENCE_THRESHOLD = 45   # below this %, report "none" (model has no healthy class)


async def tremor_ws(websocket: WebSocket):
    await websocket.accept()
    await websocket.send_json({
        "type": "meta",
        "sample_rate": SAMPLE_RATE_HZ,
        "win_size": WIN_TREMOR,
        "infer_ms": INFER_INTERVAL_TREMOR,
    })

    models = None
    try:
        models = ml.load_tremor()
    except Exception as e:
        await websocket.send_json({"type": "error", "msg": str(e)})
        await websocket.close()
        return

    clf    = models["clf"]
    scaler = models["scaler"]
    le     = models["le"]
    cfg    = models["config"]
    feature_names = cfg.get("feature_names", _TREMOR_FEATURE_KEYS)
    interval = INFER_INTERVAL_TREMOR / 1000.0

    try:
        while True:
            reader  = bridge.get_reader()
            buf     = reader.get_window(WIN_TREMOR / SAMPLE_RATE_HZ)

            severity   = "—"
            confidence = 0
            dom_freq   = 0.0
            p_park     = 0.0
            p_ess      = 0.0
            proba_list = []

            if buf.shape[0] >= 10:
                try:
                    axis        = dominant_axis(buf)
                    axis_motion = float(np.std(axis))

                    if axis_motion < 0.005:
                        # sensor is static — below tremor noise floor (~0.002 g)
                        severity   = "none"
                        confidence = 0
                        proba_list = []
                    else:
                        feat   = prepare_tremor_input(buf, feature_names)
                        feat_s = scaler.transform(feat)
                        idx    = clf.predict(feat_s)[0]
                        proba  = clf.predict_proba(feat_s)[0].tolist()
                        class_name = str(le.classes_[idx])
                        confidence = int(max(proba) * 100)
                        # When there IS motion, always show the model's best class
                        # (no confidence gate — the model has no "healthy" class,
                        #  so gating on confidence here just hides real predictions)
                        severity = SEVERITY_MAP.get(int(class_name), class_name)
                        proba_list = [round(p * 100, 1) for p in proba]

                        # band-power metrics (reuse axis already computed above)
                        feats  = extract_fft_features(axis, fs=SAMPLE_RATE_HZ)
                        dom_freq = feats["dominant_freq"]
                        total_p  = (feats["power_0_3hz"] + feats["power_3_6hz"] +
                                    feats["power_6_12hz"] + feats["power_12plus"] + 1e-10)
                        p_park   = feats["power_3_6hz"] / total_p * 100
                        p_ess    = feats["power_6_12hz"] / total_p * 100
                except Exception:
                    pass

            # Chart: dominant axis (last 100 samples)
            display = reader.get_window(2)
            axis_g  = dominant_axis(display).tolist() if display.shape[0] > 0 else []

            await websocket.send_json({
                "type":         "data",
                "severity":     severity,
                "confidence":   confidence,
                "color":        SEVERITY_COLOR.get(severity, "#e6edf3"),
                "dominantFreq": round(dom_freq, 2),
                "parkinsonPct": round(p_park, 1),
                "essentialPct": round(p_ess, 1),
                "probabilities": proba_list,
                "classes":      [str(c) for c in le.classes_],
                "chart":        axis_g[-100:],
            })

            await asyncio.sleep(interval)

    except WebSocketDisconnect:
        pass
    except Exception:
        pass
