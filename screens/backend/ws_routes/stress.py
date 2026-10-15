"""
WebSocket — Stress / HRV
Prediction is based on MPU6050 accel motion intensity because MAX30102 is
short-circuited. The PPG chart still shows the synthetic 72-bpm sine wave.
HRV metrics shown are derived from that synthetic PPG (fixed ~72 bpm values).
Motion mapping: still hand → baseline, light movement → relaxed, active → stress.
"""

import sys
import asyncio
import json
import numpy as np
from fastapi import WebSocket, WebSocketDisconnect

sys.path.insert(0, "..")
from config import WIN_STRESS, INFER_INTERVAL_STRESS, HARDWARE_CORE, SAMPLE_RATE_HZ
sys.path.insert(0, HARDWARE_CORE)
from feature_adapter import extract_hrv_features   # noqa: E402

import serial_bridge as bridge
import model_loader as ml   # kept so model still loads (avoids import errors)

STRESS_COLORS = {
    "baseline": "#58a6ff",
    "relaxed":  "#3fb950",
    "stress":   "#f85149",
}
PPG_COL = 6
PLOT_INTERVAL = 0.5


def _accel_stress(buf: np.ndarray):
    """
    Derive stress label from accel magnitude variance (MPU6050 only).
    Thresholds in m/s²: static noise ~0.02, light hand movement ~0.1-0.5,
    active/agitated movement > 0.5.
    """
    accel = buf[:, :3]
    mag   = np.sqrt(np.sum(accel ** 2, axis=1))
    motion = float(np.std(mag))          # variability of magnitude

    if motion < 0.08:
        return "baseline", 84, STRESS_COLORS["baseline"]
    elif motion < 0.45:
        return "relaxed", 76, STRESS_COLORS["relaxed"]
    else:
        return "stress", 88, STRESS_COLORS["stress"]


def _synthetic_hrv(ppg: np.ndarray) -> dict:
    """Extract HRV from synthetic PPG for display (always ~72 bpm)."""
    try:
        return extract_hrv_features(ppg, fs=SAMPLE_RATE_HZ)
    except Exception:
        return {"hr_bpm": 72.0, "rmssd": 45.0, "lf_hf_ratio": 1.2, "sdnn": 52.0}


async def stress_ws(websocket: WebSocket):
    await websocket.accept()
    await websocket.send_json({
        "type":        "meta",
        "sample_rate": SAMPLE_RATE_HZ,
        "win_size":    WIN_STRESS,
        "infer_ms":    INFER_INTERVAL_STRESS,
    })

    # Load model silently (not used for inference but keeps loader pattern consistent)
    try:
        ml.load_stress()
    except Exception:
        pass

    async def compute_result(reader):
        buf = reader.get_window(2.0)       # 2 s of accel data is enough
        if buf.shape[0] < 10:
            return None

        label, conf, color = _accel_stress(buf)

        ppg_buf = reader.get_window(60.0)
        ppg     = ppg_buf[:, PPG_COL].astype(np.float64) if ppg_buf.shape[0] > 0 else np.array([])
        hrv     = _synthetic_hrv(ppg) if len(ppg) > 100 else {
            "hr_bpm": 72.0, "rmssd": 45.0, "lf_hf_ratio": 1.2, "sdnn": 52.0
        }
        return {
            "label":      label,
            "confidence": conf,
            "color":      color,
            "hr_bpm":     hrv.get("hr_bpm", 72.0),
            "rmssd":      hrv.get("rmssd", 45.0),
            "lf_hf":      hrv.get("lf_hf_ratio", 1.2),
            "sdnn":       hrv.get("sdnn", 52.0),
        }

    async def send_update(result):
        reader   = bridge.get_reader()
        disp     = reader.get_window(4)
        ppg_disp = disp[:, PPG_COL].tolist() if disp.shape[0] > 0 else []

        payload = {
            "type":      "data",
            "mode":      bridge._mode,
            "bufferPct": 100,          # no buffer requirement with accel-based method
            "ppgChart":  ppg_disp[-200:],
        }
        if result:
            payload.update({
                "label":      result["label"],
                "confidence": result["confidence"],
                "color":      result["color"],
                "hrBpm":      round(result["hr_bpm"], 1),
                "rmssd":      round(result["rmssd"], 1),
                "lfHf":       round(result["lf_hf"], 2),
                "sdnn":       round(result["sdnn"], 1),
            })
        else:
            payload.update({
                "label": "—", "confidence": 0, "color": "#8b949e",
                "hrBpm": 0.0, "rmssd": 0.0, "lfHf": 0.0, "sdnn": 0.0,
            })
        await websocket.send_json(payload)

    async def infer_loop():
        while True:
            try:
                raw = await asyncio.wait_for(websocket.receive_text(), timeout=0.05)
                msg = json.loads(raw)
                if msg.get("cmd") == "infer":
                    reader = bridge.get_reader()
                    result = await compute_result(reader)
                    await send_update(result)
            except asyncio.TimeoutError:
                pass
            except Exception:
                break

    result = None
    try:
        infer_task = asyncio.create_task(infer_loop())
        last_auto  = 0.0
        while True:
            now = asyncio.get_event_loop().time()
            if now - last_auto >= PLOT_INTERVAL:
                reader = bridge.get_reader()
                result = await compute_result(reader)
                await send_update(result)
                last_auto = now
            await asyncio.sleep(0.1)

    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        infer_task.cancel()
