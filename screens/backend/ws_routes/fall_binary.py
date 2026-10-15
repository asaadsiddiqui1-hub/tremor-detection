"""
WebSocket — Fall Binary Classifier v2
Isolation Forest anomaly detection — trained on glove normal data only.
Score below threshold = anomaly = FALL.
"""

import asyncio
import time
from collections import deque

import numpy as np
from fastapi import WebSocket, WebSocketDisconnect

import sys
sys.path.insert(0, "..")
from config import SAMPLE_RATE_HZ
import serial_bridge as bridge
import model_loader as ml

# ── Constants ──────────────────────────────────────────────────────────────────
WIN            = 100   # 2 s at 50 Hz
INFER_INTERVAL = 0.5   # run inference every 500 ms
DEBOUNCE_N     = 5     # rolling window for debounce vote
DEBOUNCE_K     = 3     # require ≥ 3 anomalies in last 5 windows to declare FALL


def _extract_7(win: np.ndarray) -> np.ndarray:
    """7-feature vector from a (100, 3) accel window.
    Computes acceleration magnitude then extracts stats.
    Must match training notebook extract_7 exactly."""
    mag = np.linalg.norm(win, axis=1)   # (100,) scalar magnitude in m/s²
    sc  = mag - mag.mean()              # AC component
    zcr = np.sum(np.diff(np.sign(sc)) != 0) / WIN
    return np.array([
        mag.mean(), mag.std(), mag.min(), mag.max(),
        mag.max() - mag.min(), float(np.sum(sc**2)), zcr
    ], dtype=np.float64)


async def fall_binary_ws(websocket: WebSocket):
    await websocket.accept()

    try:
        m = ml.load_fall_binary()
    except Exception as e:
        await websocket.send_json({"type": "error", "msg": str(e)})
        await websocket.close()
        return

    iso       = m["clf"]       # IsolationForest instance
    scaler    = m["scaler"]
    cfg       = m["config"]
    threshold = cfg.get("threshold", -0.05)   # anomaly score threshold (from training)

    await websocket.send_json({"type": "meta", "model_ready": True,
                               "threshold": threshold})

    recent_preds: deque = deque(maxlen=DEBOUNCE_N)
    last_score   = 0.0
    last_infer   = 0.0

    try:
        while True:
            reader = bridge.get_reader()

            # Chart: accel magnitude of last 200 samples
            disp = reader.get_window(4)
            if disp.shape[0] > 0:
                chart_mag = np.linalg.norm(
                    disp[:, :3].astype(np.float64), axis=1
                ).tolist()[-200:]
            else:
                chart_mag = []

            prediction = "NORMAL"
            confidence = 0
            state      = "MONITORING"

            now = time.time()
            if (now - last_infer) >= INFER_INTERVAL:
                last_infer = now
                buf = reader.get_window(WIN / SAMPLE_RATE_HZ)

                if buf.shape[0] >= WIN:
                    win    = buf[-WIN:, :3].astype(np.float64)   # accel (m/s²)
                    feat   = _extract_7(win).reshape(1, -1)
                    feat_s = scaler.transform(feat)

                    # decision_function: more negative = more anomalous
                    try:
                        last_score = float(iso.decision_function(feat_s)[0])
                        is_anomaly = 1 if last_score < threshold else 0
                        recent_preds.append(is_anomaly)
                    except AttributeError:
                        # Wrong model type loaded — user needs to run IsolationForest notebook
                        pass

            # Debounce: FALL only if ≥ DEBOUNCE_K anomaly windows in last DEBOUNCE_N
            fall_votes = sum(recent_preds)
            if len(recent_preds) >= DEBOUNCE_N and fall_votes >= DEBOUNCE_K:
                prediction = "FALL"
                state      = "FALL_DETECTED"
                # Confidence: how far below threshold (more negative = more confident)
                confidence = min(100, int(abs(last_score - threshold) * 200))
            else:
                prediction = "NORMAL"
                state      = "MONITORING"
                confidence = min(100, int((last_score - threshold) * 100)) if last_score > threshold else 0

            await websocket.send_json({
                "model_ready": True,
                "prediction":  prediction,
                "confidence":  confidence,
                "anomaly_score": round(last_score, 4),
                "threshold":   round(threshold, 4),
                "fall_votes":  fall_votes,
                "state":       state,
                "chart_mag":   chart_mag,
            })

            await asyncio.sleep(0.2)

    except WebSocketDisconnect:
        pass
    except Exception:
        pass
