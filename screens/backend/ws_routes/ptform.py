"""
WebSocket — PT Form Assessment
Streams: joint angle chart + form classification.
"""

import sys
import asyncio
import numpy as np
from fastapi import WebSocket, WebSocketDisconnect

sys.path.insert(0, "..")
from config import WIN_PT, INFER_INTERVAL_PT, HARDWARE_CORE, SAMPLE_RATE_HZ
sys.path.insert(0, HARDWARE_CORE)
from feature_adapter import evaluate_pt_form, compute_joint_angle    # noqa: E402

import serial_bridge as bridge
import model_loader as ml

CLASSES = ["correct", "overextension", "insufficient_range", "wrong_plane"]
COLORS  = {
    "correct":            "#3fb950",
    "overextension":      "#f85149",
    "insufficient_range": "#d29922",
    "wrong_plane":        "#bc8cff",
}


async def ptform_ws(websocket: WebSocket):
    await websocket.accept()
    await websocket.send_json({
        "type":        "meta",
        "sample_rate": SAMPLE_RATE_HZ,
        "win_size":    WIN_PT,
        "infer_ms":    INFER_INTERVAL_PT,
        "classes":     CLASSES,
    })

    interval = INFER_INTERVAL_PT / 1000.0

    try:
        while True:
            reader  = bridge.get_reader()
            buf     = reader.get_window(WIN_PT / SAMPLE_RATE_HZ)

            label      = "—"
            confidence = 0
            peak_angle = 0.0
            rom        = 0.0
            proba_dict = {c: 0 for c in CLASSES}
            angles     = []

            if buf.shape[0] >= 20:
                try:
                    result = evaluate_pt_form(buf)
                    label  = result["label"]
                    peak_angle = result["peak_angle"]
                    rom        = result["rom"]

                    # Rule-based → assign 90% confidence to the decided class
                    confidence = 90
                    proba_dict = {c: 5 for c in CLASSES}
                    proba_dict[label] = 85

                    angles = compute_joint_angle(buf).tolist()
                except Exception:
                    pass

            await websocket.send_json({
                "type":         "data",
                "label":        label,
                "confidence":   confidence,
                "color":        COLORS.get(label, "#e6edf3"),
                "peakAngle":    round(peak_angle, 1),
                "rom":          round(rom, 1),
                "probabilities": proba_dict,
                "jointAngles":  angles[-100:],
            })

            await asyncio.sleep(interval)

    except WebSocketDisconnect:
        pass
    except Exception:
        pass
