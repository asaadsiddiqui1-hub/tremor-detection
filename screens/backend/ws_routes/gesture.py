"""
WebSocket — Gesture Translator
Random Forest: goodbye | thanks | welcome
Window: 100 samples x 6 axes (accel_xyz + gyro_xyz), 12 features (mean+std per axis).
"""

import sys
import time
import asyncio
import numpy as np
from fastapi import WebSocket, WebSocketDisconnect

sys.path.insert(0, "..")
from config import WIN_GESTURE, INFER_INTERVAL_GESTURE, SAMPLE_RATE_HZ

import serial_bridge as bridge
import model_loader as ml

GESTURE_COLORS = {
    "goodbye": "#58a6ff",
    "thanks":  "#3fb950",
    "welcome": "#ff9800",
}
COLUMNS = ['accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z']


async def gesture_ws(websocket: WebSocket):
    await websocket.accept()

    models = None
    try:
        models = ml.load_gesture()
    except Exception as e:
        await websocket.send_json({"type": "error", "msg": str(e)})
        await websocket.close()
        return

    clf              = models["clf"]
    cfg              = models["config"]
    classes          = cfg["classes"]
    motion_threshold = 1.5    # v1 has no idle class — high bar prevents still-hand triggers
    conf_threshold   = 0.80   # v1 has no idle class — require strong confidence
    cooldown_s       = cfg["cooldown_s"]

    await websocket.send_json({
        "type":        "meta",
        "sample_rate": SAMPLE_RATE_HZ,
        "win_size":    WIN_GESTURE,
        "columns":     COLUMNS,
        "infer_ms":    INFER_INTERVAL_GESTURE,
        "model_ready": True,
        "classes":     classes,
    })

    interval          = INFER_INTERVAL_GESTURE / 1000.0
    last_gesture_time = 0.0
    last_gesture      = "—"
    last_confidence   = 0

    try:
        while True:
            reader = bridge.get_reader()
            buf    = reader.get_window(WIN_GESTURE / SAMPLE_RATE_HZ)

            accel_x = buf[:, 0].tolist() if buf.shape[0] > 0 else []
            accel_y = buf[:, 1].tolist() if buf.shape[0] > 0 else []
            accel_z = buf[:, 2].tolist() if buf.shape[0] > 0 else []
            gyro_x  = buf[:, 3].tolist() if buf.shape[0] > 0 else []

            gesture    = "—"
            confidence = 0
            state      = "LISTENING"

            if buf.shape[0] >= 50:   # need at least 50 samples; take last 100
                window    = buf[-100:, :6] if buf.shape[0] >= 100 else buf[:, :6]
                mean_vals = np.mean(window, axis=0)
                std_vals  = np.std(window, axis=0)
                features  = np.concatenate([mean_vals, std_vals]).reshape(1, -1)
                motion    = float(np.max(std_vals[:3]))

                now         = time.time()
                in_cooldown = (now - last_gesture_time) < cooldown_s

                if motion < motion_threshold:
                    state = "LISTENING"
                elif in_cooldown:
                    gesture    = last_gesture
                    confidence = last_confidence
                    state      = "COOLDOWN"
                else:
                    try:
                        proba    = clf.predict_proba(features)[0]
                        max_prob = float(proba.max())
                        best_idx = int(proba.argmax())
                        if max_prob >= conf_threshold:
                            gesture           = classes[best_idx]
                            confidence        = int(max_prob * 100)
                            state             = "DETECTED"
                            last_gesture_time = now
                            last_gesture      = gesture
                            last_confidence   = confidence
                        else:
                            state      = "UNCLEAR"
                            confidence = int(max_prob * 100)
                    except Exception:
                        pass

            await websocket.send_json({
                "type":        "data",
                "model_ready": True,
                "gesture":     gesture,
                "confidence":  confidence,
                "state":       state,
                "accel_x":     accel_x[-100:],
                "accel_y":     accel_y[-100:],
                "accel_z":     accel_z[-100:],
                "gyro_x":      gyro_x[-100:],
            })

            await asyncio.sleep(interval)

    except WebSocketDisconnect:
        pass
    except Exception:
        pass
