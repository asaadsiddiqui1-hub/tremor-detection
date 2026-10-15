"""
WebSocket — Live Sensor Stream
Streams all 7 raw channels at high speed (no model inference).
"""

import sys
import asyncio
import numpy as np
from fastapi import WebSocket, WebSocketDisconnect

sys.path.insert(0, "..")
from config import WIN_SENSOR, INFER_INTERVAL_SENSOR, SAMPLE_RATE_HZ

import serial_bridge as bridge

COLUMNS = ['accel_x','accel_y','accel_z','gyro_x','gyro_y','gyro_z','ppg_raw']


async def sensor_ws(websocket: WebSocket):
    await websocket.accept()
    await websocket.send_json({
        "type":        "meta",
        "sample_rate": SAMPLE_RATE_HZ,
        "win_size":    WIN_SENSOR,
        "columns":     COLUMNS,
        "infer_ms":    INFER_INTERVAL_SENSOR,
    })

    interval = INFER_INTERVAL_SENSOR / 1000.0

    try:
        while True:
            reader = bridge.get_reader()
            buf    = reader.get_window(WIN_SENSOR / SAMPLE_RATE_HZ)

            payload = {"type": "data", "channels": {}}

            if buf.shape[0] > 0:
                for i, col in enumerate(COLUMNS):
                    payload["channels"][col] = buf[:, i].tolist()
            else:
                for col in COLUMNS:
                    payload["channels"][col] = []

            # Also send the latest single sample for numeric display
            last = reader.last_sample
            if last:
                payload["latest"] = {COLUMNS[i]: round(last[i], 4)
                                     for i in range(len(COLUMNS))}

            payload["sample_rate"] = getattr(reader, "actual_rate", SAMPLE_RATE_HZ)
            payload["lines_read"]  = reader.lines_read

            await websocket.send_json(payload)
            await asyncio.sleep(interval)

    except WebSocketDisconnect:
        pass
    except Exception:
        pass
