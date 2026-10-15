"""
WebSocket — Fall Detection
Streams: accel magnitude chart data + FALL/NORMAL prediction every 500 ms.
"""

import sys
import asyncio
import numpy as np
from fastapi import WebSocket, WebSocketDisconnect

sys.path.insert(0, "..")
from config import WIN_FALL, INFER_INTERVAL_FALL, HARDWARE_CORE, SAMPLE_RATE_HZ
sys.path.insert(0, HARDWARE_CORE)
from feature_adapter import prepare_fall_input   # noqa: E402

import serial_bridge as bridge
import model_loader as ml


async def fall_ws(websocket: WebSocket):
    await websocket.accept()
    models = None

    # Send handshake metadata so frontend can configure itself
    await websocket.send_json({
        "type": "meta",
        "sample_rate": SAMPLE_RATE_HZ,
        "win_size": WIN_FALL,
        "infer_ms": INFER_INTERVAL_FALL,
    })

    try:
        models = ml.load_fall()
    except Exception as e:
        await websocket.send_json({"type": "error", "msg": str(e)})
        await websocket.close()
        return

    model     = models["model"]
    threshold = models["threshold"]
    interval  = INFER_INTERVAL_FALL / 1000.0

    # Debounce: require 3/5 recent predictions to be FALL before displaying FALL
    recent_preds = []
    DEBOUNCE_N   = 5

    try:
        while True:
            reader = bridge.get_reader()
            buf    = reader.get_window(WIN_FALL / SAMPLE_RATE_HZ)
            have   = reader.buffer_length() if hasattr(reader, "buffer_length") else len(reader._buffer)

            # Accel magnitude for chart (last 200 points)
            display_buf = reader.get_window(4)
            accel       = display_buf[:, :3]
            mag         = np.sqrt(np.sum(accel ** 2, axis=1))
            mu, s       = mag.mean(), mag.std()
            if (mag.max() - mag.min()) < 0.5:
                mag_z = [0.0] * len(mag)
            elif s > 1e-6:
                mag_z = ((mag - mu) / s).tolist()
            else:
                mag_z = [0.0] * len(mag)

            # Inference
            raw_pred   = "NORMAL"
            confidence = 0
            prob       = 0.0
            if buf.shape[0] >= 64:
                try:
                    x    = prepare_fall_input(buf, target_len=WIN_FALL)
                    prob = float(model.predict(x, verbose=0)[0][0])
                    raw_pred   = "FALL" if prob >= threshold else "NORMAL"
                    confidence = int(prob * 100)
                except Exception:
                    pass

            # Debounce: smooth rapid FALL/NORMAL oscillation
            recent_preds.append(1 if raw_pred == "FALL" else 0)
            if len(recent_preds) > DEBOUNCE_N:
                recent_preds.pop(0)
            prediction = "FALL" if sum(recent_preds) >= 3 else "NORMAL"

            await websocket.send_json({
                "type":       "data",
                "prediction": prediction,
                "confidence": confidence,
                "prob":       round(prob, 4),
                "threshold":  threshold,
                "buffer_pct": min(100, int(have * 100 / WIN_FALL)),
                "chart":      mag_z[-200:],
            })

            await asyncio.sleep(interval)

    except WebSocketDisconnect:
        pass
    except Exception:
        pass
