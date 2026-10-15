"""
WebSocket — Muscle Atrophy
Streams: live accel magnitude chart continuously.
Inference is button-triggered: frontend sends JSON { age, immobilisedDays }.
"""

import sys
import asyncio
import json
import numpy as np
from fastapi import WebSocket, WebSocketDisconnect

sys.path.insert(0, "..")
from config import WIN_STRESS, HARDWARE_CORE, SAMPLE_RATE_HZ
sys.path.insert(0, HARDWARE_CORE)
from feature_adapter import prepare_atrophy_input    # noqa: E402

import serial_bridge as bridge
import model_loader as ml

PLOT_INTERVAL = 0.2     # 200 ms live chart refresh


async def atrophy_ws(websocket: WebSocket):
    await websocket.accept()
    await websocket.send_json({
        "type":        "meta",
        "sample_rate": SAMPLE_RATE_HZ,
        "infer_ms":    "button-triggered",
    })

    models = None
    try:
        models = ml.load_atrophy()
    except Exception as e:
        await websocket.send_json({"type": "error", "msg": str(e)})
        await websocket.close()
        return

    gbr    = models["gbr"]
    scaler = models["scaler"]
    cfg    = models["config"]
    feat_names = cfg.get("feature_names",
                         ["spatial_variance","resting_hr","hr_variability",
                          "immobilised_days","age","activity_level"])

    async def send_chart():
        reader  = bridge.get_reader()
        disp    = reader.get_window(4)
        accel   = disp[:, :3]
        mag     = np.sqrt(np.sum(accel ** 2, axis=1)).tolist() if disp.shape[0] > 0 else []
        await websocket.send_json({
            "type":  "chart",
            "accel": mag[-200:],
        })

    # Separate task to listen for inference requests from the frontend
    async def infer_loop():
        while True:
            try:
                raw = await asyncio.wait_for(websocket.receive_text(), timeout=0.05)
                msg = json.loads(raw)
                if msg.get("cmd") == "infer":
                    age   = float(msg.get("age", 35))
                    imm   = float(msg.get("immobilisedDays", 0))
                    reader = bridge.get_reader()
                    buf    = reader.get_window(WIN_STRESS / SAMPLE_RATE_HZ)
                    feat   = prepare_atrophy_input(buf, imm, age, feat_names)
                    feat_s = scaler.transform(feat)
                    pred   = float(gbr.predict(feat_s)[0])
                    pred   = float(np.clip(pred, 0, 25))
                    color  = ("#3fb950" if pred < 5 else
                              "#d29922" if pred < 15 else "#f85149")
                    await websocket.send_json({
                        "type":         "result",
                        "atrophyPct":   round(pred, 1),
                        "color":        color,
                        "spatialVar":   round(float(feat[0, 0]), 3),
                        "restingHr":    round(float(feat[0, 1]), 0),
                        "hrVariability":round(float(feat[0, 2]), 1),
                        "activityLevel":round(float(feat[0, 5]), 2),
                    })
            except asyncio.TimeoutError:
                pass
            except Exception:
                break

    try:
        infer_task = asyncio.create_task(infer_loop())
        while True:
            await send_chart()
            await asyncio.sleep(PLOT_INTERVAL)
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        infer_task.cancel()
