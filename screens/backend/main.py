"""
NeuroPath AI — FastAPI Backend
================================
Run:  uvicorn main:app --reload --port 8000
"""

import os
import sys

from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from config import EVAL_IMAGES, DEFAULT_COM_PORT, SAMPLE_RATE_HZ
import serial_bridge as bridge

# WebSocket route handlers
from ws_routes.fall    import fall_ws
from ws_routes.tremor  import tremor_ws
from ws_routes.stress  import stress_ws
from ws_routes.atrophy import atrophy_ws
from ws_routes.ptform  import ptform_ws
from ws_routes.sensor  import sensor_ws
from ws_routes.gesture       import gesture_ws
from ws_routes.gesture_v2    import gesture_v2_ws
from ws_routes.tremor_binary import tremor_binary_ws
from ws_routes.fall_binary   import fall_binary_ws

app = FastAPI(title="NeuroPath AI", version="1.0.0")

# ── CORS (allow React dev server) ─────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Start simulated reader on boot ────────────────────────────────────────────
@app.on_event("startup")
async def startup():
    bridge.connect_sim("normal")


# ═══════════════════════════════════════════════════════════════════════════════
# REST  —  Connection management
# ═══════════════════════════════════════════════════════════════════════════════

class ConnectBody(BaseModel):
    mode:     str = "sim"      # "sim" | "live"
    port:     str = DEFAULT_COM_PORT
    scenario: str = "normal"   # for sim: "normal"|"fall"|"tremor"|"stress"

class SampleRateBody(BaseModel):
    hz: int

@app.get("/api/ports")
def get_ports():
    return {"ports": bridge.list_ports()}

@app.get("/api/status")
def get_status():
    return bridge.get_status()

@app.post("/api/connect")
def connect(body: ConnectBody):
    if body.mode == "live":
        result = bridge.connect_live(body.port)
    else:
        result = bridge.connect_sim(body.scenario)
    return result

@app.post("/api/sample-rate")
def set_sample_rate(body: SampleRateBody):
    """Hot-update the sample rate. Reconnects the reader."""
    import config as cfg
    hz = max(1, min(500, body.hz))
    cfg.SAMPLE_RATE_HZ = hz
    # Reconnect sim with new rate
    bridge.connect_sim()
    return {"ok": True, "sample_rate": hz}

# ── Serve model evaluation images ─────────────────────────────────────────────

@app.get("/api/eval-image/{feature}")
def eval_image(feature: str):
    path = EVAL_IMAGES.get(feature)
    if path and os.path.exists(path):
        return FileResponse(path, media_type="image/png")
    return JSONResponse({"error": "not found"}, status_code=404)


# ═══════════════════════════════════════════════════════════════════════════════
# WebSocket endpoints
# ═══════════════════════════════════════════════════════════════════════════════

@app.websocket("/ws/fall")
async def ws_fall(ws: WebSocket):
    await fall_ws(ws)

@app.websocket("/ws/tremor")
async def ws_tremor(ws: WebSocket):
    await tremor_ws(ws)

@app.websocket("/ws/stress")
async def ws_stress(ws: WebSocket):
    await stress_ws(ws)

@app.websocket("/ws/atrophy")
async def ws_atrophy(ws: WebSocket):
    await atrophy_ws(ws)

@app.websocket("/ws/ptform")
async def ws_ptform(ws: WebSocket):
    await ptform_ws(ws)

@app.websocket("/ws/sensor")
async def ws_sensor(ws: WebSocket):
    await sensor_ws(ws)

@app.websocket("/ws/gesture")
async def ws_gesture(ws: WebSocket):
    await gesture_ws(ws)

@app.websocket("/ws/gesture-v2")
async def ws_gesture_v2(ws: WebSocket):
    await gesture_v2_ws(ws)

@app.websocket("/ws/tremor-binary")
async def ws_tremor_binary(ws: WebSocket):
    await tremor_binary_ws(ws)

@app.websocket("/ws/fall-binary")
async def ws_fall_binary(ws: WebSocket):
    await fall_binary_ws(ws)
