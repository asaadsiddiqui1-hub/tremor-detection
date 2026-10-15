"""
WebSocket — Gesture Translator v2
3-class (goodbye / thanks / welcome) · motion gate handles idle
32 orientation-invariant features · augmented training · TTS · consecutive-window voting
"""

import sys
import asyncio
import time
import threading
import numpy as np
from fastapi import WebSocket, WebSocketDisconnect

sys.path.insert(0, "..")
from config import SAMPLE_RATE_HZ, HARDWARE_CORE

import serial_bridge as bridge
import model_loader as ml

# ── TTS via Windows built-in speech (no pyttsx3 needed) ──────────────────────
import subprocess

def _speak(word: str):
    """Non-blocking TTS using Windows SAPI5 via PowerShell COM object."""
    def _run():
        subprocess.run(
            ['powershell', '-WindowStyle', 'Hidden', '-Command',
             f'(New-Object -ComObject SAPI.SpVoice).Speak("{word}")'],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
    threading.Thread(target=_run, daemon=True).start()

# ── Constants ──────────────────────────────────────────────────────────────────
WIN = 100
GESTURE_COLORS = {
    "goodbye": "#58a6ff",
    "thanks":  "#3fb950",
    "welcome": "#ff9800",
    "idle":    "#6e7681",
}


# ── Feature extractors (one per saved model generation) ───────────────────────

def _extract_42_legacy(win: np.ndarray) -> np.ndarray:
    """42-feature vector — generation 1 (pre-magnitude retrain)."""
    n      = win.shape[0]
    mean   = np.mean(win, axis=0)
    std    = np.std(win, axis=0)
    mn     = np.min(win, axis=0)
    mx     = np.max(win, axis=0)
    rng    = mx - mn
    energy = np.sum(win ** 2, axis=0)
    zcr    = np.array([
        np.sum(np.diff(np.sign(win[:, i])) != 0) / n
        for i in range(win.shape[1])
    ])
    return np.concatenate([mean, std, mn, mx, rng, energy, zcr])


def _extract_56_legacy(win: np.ndarray) -> np.ndarray:
    """56-feature vector — generation 2 (mean/min/max/energy + magnitudes).
    Kept for backward compat with models saved before the orientation-invariant switch.
    """
    n      = win.shape[0]
    mean   = np.mean(win, axis=0)
    std    = np.std(win, axis=0)
    mn     = np.min(win, axis=0)
    mx     = np.max(win, axis=0)
    rng    = mx - mn
    energy = np.sum(win ** 2, axis=0)
    zcr    = np.array([
        np.sum(np.diff(np.sign(win[:, i] - mean[i])) != 0) / n
        for i in range(win.shape[1])
    ])
    accel_mag = np.sqrt(np.sum(win[:, :3] ** 2, axis=1))
    gyro_mag  = np.sqrt(np.sum(win[:, 3:] ** 2, axis=1))

    def _mag_feats(mag: np.ndarray) -> np.ndarray:
        m_mean = np.mean(mag); m_std = np.std(mag)
        m_min  = np.min(mag);  m_max = np.max(mag)
        return np.array([m_mean, m_std, m_min, m_max, m_max - m_min,
                         float(np.sum(mag ** 2)),
                         np.sum(np.diff(np.sign(mag - m_mean)) != 0) / n])

    return np.concatenate([mean, std, mn, mx, rng, energy, zcr,
                           _mag_feats(accel_mag), _mag_feats(gyro_mag)])


def _extract_32(win: np.ndarray) -> np.ndarray:
    """32 orientation-invariant features — generation 3 (current).

    Per-axis (3 × 6 = 18):  std, range, ZCR (mean-centred)
    Magnitude    (7 × 2 = 14): mean, std, min, max, range, energy, ZCR

    Excluded (orientation-sensitive, biased by gravity direction):
      mean, min, max, energy per axis.
    """
    n    = win.shape[0]
    mean = np.mean(win, axis=0)
    std  = np.std(win,  axis=0)
    mn   = np.min(win,  axis=0)
    mx   = np.max(win,  axis=0)
    rng  = mx - mn
    zcr  = np.array([
        np.sum(np.diff(np.sign(win[:, i] - mean[i])) != 0) / n
        for i in range(win.shape[1])
    ])
    accel_mag = np.sqrt(np.sum(win[:, :3] ** 2, axis=1))
    gyro_mag  = np.sqrt(np.sum(win[:, 3:6] ** 2, axis=1))

    def _mag_feats(mag: np.ndarray) -> np.ndarray:
        m_mean = np.mean(mag); m_std = np.std(mag)
        m_min  = np.min(mag);  m_max = np.max(mag)
        return np.array([m_mean, m_std, m_min, m_max, m_max - m_min,
                         float(np.sum(mag ** 2)),
                         np.sum(np.diff(np.sign(mag - m_mean)) != 0) / n])

    return np.concatenate([std, rng, zcr,
                           _mag_feats(accel_mag), _mag_feats(gyro_mag)])


_EXTRACTOR_MAP = {
    42: _extract_42_legacy,
    56: _extract_56_legacy,
    32: _extract_32,
}


async def gesture_v2_ws(websocket: WebSocket):
    await websocket.accept()

    try:
        m = ml.load_gesture_v2()
    except Exception as e:
        await websocket.send_json({"type": "error", "msg": str(e)})
        await websocket.close()
        return

    clf    = m["clf"]
    scaler = m["scaler"]
    le     = m["le"]
    cfg    = m["config"]

    classes          = cfg.get("classes", ["goodbye", "thanks", "welcome"])
    motion_threshold = cfg.get("motion_threshold", 1.0)
    cooldown_s       = cfg.get("cooldown_s", 2.0)
    use_scaler       = cfg.get("needs_scaler",
                               cfg.get("model_type", "RF") in ("SVM", "GradientBoosting"))
    has_idle = (len(classes) == 4 and classes[-1] == "idle")
    idle_idx = 3 if has_idle else None

    feat_count = cfg.get("feature_count", 42)
    _extractor = _EXTRACTOR_MAP.get(feat_count, _extract_42_legacy)
    print(f"[gesture_v2] model={cfg.get('model_type')}  features={feat_count}  "
          f"extractor={_extractor.__name__}")

    # Thresholds: goodbye tends to be over-predicted (most training data, most generic
    # motion), so we require a higher bar for it than thanks/welcome.
    CLASS_THRESHOLDS = {
        "goodbye": 0.75,
        "thanks":  0.60,
        "welcome": 0.60,
    }

    await websocket.send_json({"type": "meta", "model_ready": True,
                               "classes": classes, "features": feat_count,
                               "model_type": cfg.get("model_type", "SVM")})

    last_gesture_time = 0.0
    last_gesture      = "—"
    last_confidence   = 0
    _vote_cls         = None
    _vote_count       = 0

    try:
        while True:
            reader = bridge.get_reader()
            buf    = reader.get_window(WIN / SAMPLE_RATE_HZ)

            gesture    = "—"
            confidence = 0
            state      = "LISTENING"

            if buf.shape[0] >= WIN:
                win = buf[-WIN:, :6].astype(np.float64)

                # Motion gate: max std across accel axes (m/s²)
                motion = float(np.max(np.std(win[:, :3], axis=0)))

                if motion >= motion_threshold:
                    try:
                        feat = _extractor(win).reshape(1, -1)
                        if use_scaler:
                            feat = scaler.transform(feat)
                        proba = clf.predict_proba(feat)[0]
                    except Exception as exc:
                        print(f"[gesture_v2] inference error: {exc}")
                        _vote_cls, _vote_count = None, 0
                        continue

                    if has_idle and int(np.argmax(proba)) == idle_idx:
                        state      = "UNCLEAR"
                        confidence = int(proba[idle_idx] * 100)
                        _vote_cls, _vote_count = None, 0
                    else:
                        gesture_proba = proba[:idle_idx] if has_idle else proba
                        best_i   = int(np.argmax(gesture_proba))
                        best_p   = float(gesture_proba[best_i])
                        pred_cls = classes[best_i]
                        threshold = CLASS_THRESHOLDS.get(pred_cls, 0.60)

                        now         = time.time()
                        in_cooldown = (now - last_gesture_time) < cooldown_s

                        if in_cooldown:
                            state      = "COOLDOWN"
                            gesture    = last_gesture
                            confidence = last_confidence
                            _vote_cls, _vote_count = None, 0

                        elif best_p >= threshold:
                            if pred_cls == _vote_cls:
                                _vote_count += 1
                            else:
                                _vote_cls   = pred_cls
                                _vote_count = 1

                            if _vote_count >= 2:
                                state             = "DETECTED"
                                gesture           = pred_cls
                                confidence        = int(best_p * 100)
                                last_gesture_time = now
                                last_gesture      = gesture
                                last_confidence   = confidence
                                _vote_cls, _vote_count = None, 0
                                _speak(gesture)
                            else:
                                state      = "UNCLEAR"
                                confidence = int(best_p * 100)
                        else:
                            state      = "UNCLEAR"
                            confidence = int(best_p * 100)
                            _vote_cls, _vote_count = None, 0
                else:
                    _vote_cls, _vote_count = None, 0

            display = reader.get_window(2)
            d = display[:, :6] if display.shape[0] > 0 else np.zeros((1, 6))

            await websocket.send_json({
                "model_ready": True,
                "gesture":     gesture,
                "confidence":  confidence,
                "state":       state,
                "color":       GESTURE_COLORS.get(gesture, "#6e7681"),
                "features":    feat_count,
                "model_type":  cfg.get("model_type", "SVM"),
                "accel_x":     d[:, 0].tolist()[-100:],
                "accel_y":     d[:, 1].tolist()[-100:],
                "accel_z":     d[:, 2].tolist()[-100:],
                "gyro_x":      d[:, 3].tolist()[-100:],
            })

            await asyncio.sleep(0.1)

    except WebSocketDisconnect:
        pass
    except Exception:
        pass
