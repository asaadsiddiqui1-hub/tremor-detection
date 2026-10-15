"""
WebSocket — Binary Tremor Classifier (normal vs tremor)
Uses real recorded normal data + Parkinson's dataset.
Same FFT feature pipeline as the 3-class model.
"""

import sys
import asyncio
import numpy as np
from fastapi import WebSocket, WebSocketDisconnect

sys.path.insert(0, "..")
from config import WIN_TREMOR, INFER_INTERVAL_TREMOR, HARDWARE_CORE, SAMPLE_RATE_HZ
sys.path.insert(0, HARDWARE_CORE)
from feature_adapter import prepare_tremor_input, extract_fft_features, dominant_axis, _TREMOR_FEATURE_KEYS

import serial_bridge as bridge
import model_loader as ml

RESULT_COLORS = {
    "normal": "#3fb950",
    "tremor": "#f85149",
}

MOTION_GATE = 0.005   # g — below this the hand is static, skip inference


async def tremor_binary_ws(websocket: WebSocket):
    await websocket.accept()
    await websocket.send_json({
        "type":        "meta",
        "sample_rate": SAMPLE_RATE_HZ,
        "win_size":    WIN_TREMOR,
        "infer_ms":    INFER_INTERVAL_TREMOR,
        "model":       "binary (normal vs tremor)",
    })

    try:
        models = ml.load_tremor_binary()
    except Exception as e:
        await websocket.send_json({"type": "error", "msg": str(e)})
        await websocket.close()
        return

    clf    = models["clf"]
    scaler = models["scaler"]
    le     = models["le"]
    cfg    = models["config"]
    feature_names = cfg.get("feature_names", list(_TREMOR_FEATURE_KEYS))
    interval = INFER_INTERVAL_TREMOR / 1000.0

    try:
        while True:
            reader = bridge.get_reader()
            buf    = reader.get_window(WIN_TREMOR / SAMPLE_RATE_HZ)

            result     = "—"
            confidence = 0
            dom_freq   = 0.0
            p_park     = 0.0
            p_ess      = 0.0

            if buf.shape[0] >= 10:
                try:
                    axis        = dominant_axis(buf)
                    axis_motion = float(np.std(axis))

                    if axis_motion >= MOTION_GATE:
                        feat   = prepare_tremor_input(buf, feature_names)
                        feat_s = scaler.transform(feat)
                        idx    = clf.predict(feat_s)[0]
                        proba  = clf.predict_proba(feat_s)[0]
                        class_int  = int(le.classes_[idx])
                        result     = "normal" if class_int == 0 else "tremor"
                        confidence = int(proba.max() * 100)

                        feats    = extract_fft_features(axis, fs=SAMPLE_RATE_HZ)
                        dom_freq = feats["dominant_freq"]
                        total_p  = (feats["power_0_3hz"] + feats["power_3_6hz"] +
                                    feats["power_6_12hz"] + feats["power_12plus"] + 1e-10)
                        p_park   = feats["power_3_6hz"] / total_p * 100
                        p_ess    = feats["power_6_12hz"] / total_p * 100

                        # Frequency fallback: Parkinson's band dominant + right frequency range
                        # Real Parkinson's tremor: 3-6 Hz with >60% band power
                        if result == "normal" and p_park > 60 and 3.0 <= dom_freq <= 6.5:
                            result     = "tremor"
                            confidence = int(p_park)
                    else:
                        result     = "normal"
                        confidence = 0
                except Exception:
                    pass

            # Chart: dominant axis last 2 s
            display = reader.get_window(2)
            chart   = dominant_axis(display).tolist() if display.shape[0] > 0 else []

            await websocket.send_json({
                "type":         "data",
                "result":       result,
                "confidence":   confidence,
                "color":        RESULT_COLORS.get(result, "#8b949e"),
                "dominantFreq": round(dom_freq, 2),
                "parkinsonPct": round(p_park, 1),
                "essentialPct": round(p_ess, 1),
                "chart":        chart[-100:],
            })

            await asyncio.sleep(interval)

    except WebSocketDisconnect:
        pass
    except Exception:
        pass
