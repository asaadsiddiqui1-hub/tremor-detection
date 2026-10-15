"""
Feature extraction for all 7 NeuroPath models.

CRITICAL RULE: Every function here must stay byte-for-byte identical to the
corresponding feature extraction code in the training notebooks.  If a notebook
changes its feature code, change this file too — and vice versa.

Unit notes
----------
- ESP32 sends accel in m/s²; tremor model was trained on data in g-units.
  adapt_tremor() divides by G before extraction to match training.
- Stress model config says ppg_fs_hz=64 but serial stream runs at FS=50 Hz.
  adapt_stress() passes fs=50 explicitly so R-R intervals are computed correctly.
"""

import numpy as np
import pandas as pd
from scipy import signal as sp_signal
from scipy.fft import fft, fftfreq

# ── Constants ─────────────────────────────────────────────────────────────────
FS   = 50       # serial stream sample rate (Hz)
G    = 9.81     # m/s² → g conversion

# Column indices in the 7-column serial frame
IDX_AX, IDX_AY, IDX_AZ = 0, 1, 2
IDX_GX, IDX_GY, IDX_GZ = 3, 4, 5
IDX_PPG                 = 6

# Window sizes (samples at 50 Hz unless noted)
WIN_FALL   = 512    # 10.24 s
WIN_TREMOR = 100    #  2.00 s
WIN_STRESS = 3000   # 60.00 s
WIN_PT     = 100    #  2.00 s
WIN_SENSOR = 200    #  4.00 s  (display only)

TREMOR_FEATURES = [
    "dominant_freq", "peak_magnitude",
    "power_0_3hz", "power_3_6hz", "power_6_12hz", "power_12plus",
    "ratio_parkinson", "ratio_essential",
    "mean", "std", "rms", "max_abs", "skewness", "kurtosis", "jerk_mean",
]

HRV_FEATURES = [
    "rmssd", "sdnn", "pnn50", "mean_rr", "std_rr",
    "hr_bpm", "lf_hf_ratio", "cv_rr",
]

ATROPHY_FEATURES = [
    "spatial_variance", "resting_hr", "hr_variability",
    "immobilised_days", "age", "activity_level",
]


# ─────────────────────────────────────────────────────────────────────────────
# 1. Fall detection  (1D-CNN, window=512, input shape (512,1))
# ─────────────────────────────────────────────────────────────────────────────

def adapt_fall(buf: np.ndarray) -> np.ndarray:
    """
    buf : (512, 7)  →  (512, 1) float32 for the 1D-CNN.

    Uses z-scored accelerometer magnitude as the single channel.
    The SisFall binary training data was preprocessed to a single-channel
    accelerometer signal — magnitude + z-score is the closest match we can
    derive from the live 3-axis stream without the original preprocessing script.
    """
    accel = buf[:, IDX_AX : IDX_AZ + 1]                         # (512, 3)
    mag   = np.sqrt(np.sum(accel ** 2, axis=1)).astype(np.float32)
    mu, sigma = mag.mean(), mag.std()
    # Range-based static detection: MPU6050 noise gives sigma ~0.02 m/s²
    # which is >> 1e-6 but still produces z-scored ±2 oscillations that
    # confuse the CNN into predicting FALL. Use magnitude range instead.
    if (mag.max() - mag.min()) < 0.5:
        mag = np.zeros_like(mag)
    elif sigma > 1e-6:
        mag = (mag - mu) / sigma
    else:
        mag = np.zeros_like(mag)
    return mag.reshape(WIN_FALL, 1)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Tremor severity  (RandomForest, 15 FFT features)
# ─────────────────────────────────────────────────────────────────────────────

def _dominant_axis(accel_block: np.ndarray) -> np.ndarray:
    """Return the axis (column) of (N,3) accel with the highest AC variance."""
    var = [np.var(accel_block[:, i] - np.mean(accel_block[:, i])) for i in range(3)]
    return accel_block[:, int(np.argmax(var))]


def _fft_features(signal_1d: np.ndarray, fs: float = FS) -> np.ndarray:
    """
    15 FFT features — identical to notebook 02 extract_fft_features().
    Input signal is expected to be in g-units (see adapt_tremor).
    """
    x = np.asarray(signal_1d, dtype=np.float64)
    N = len(x)
    if N < 4:
        return np.zeros(len(TREMOR_FEATURES), dtype=np.float64)

    x_ac      = x - np.mean(x)
    freqs     = fftfreq(N, d=1.0 / fs)[: N // 2]
    magnitude = np.abs(fft(x_ac))[: N // 2]
    di        = int(np.argmax(magnitude))

    def bp(lo, hi):
        mask = (freqs >= lo) & (freqs < hi)
        return float(np.sum(magnitude[mask] ** 2))

    p03, p36, p612, p12 = bp(0, 3), bp(3, 6), bp(6, 12), bp(12, 25)
    tot = p03 + p36 + p612 + p12 + 1e-10

    s = pd.Series(x)
    return np.array([
        float(freqs[di]),                         # dominant_freq
        float(magnitude[di]),                     # peak_magnitude
        p03, p36, p612, p12,                      # band powers
        p36  / tot,                               # ratio_parkinson  (3-6 Hz)
        p612 / tot,                               # ratio_essential  (6-12 Hz)
        float(np.mean(x)),                        # mean
        float(np.std(x)),                         # std
        float(np.sqrt(np.mean(x ** 2))),          # rms
        float(np.max(np.abs(x))),                 # max_abs
        float(s.skew()),                          # skewness
        float(s.kurt()),                          # kurtosis
        float(np.mean(np.abs(np.diff(x)))),       # jerk_mean
    ], dtype=np.float64)


def adapt_tremor(buf: np.ndarray) -> np.ndarray:
    """
    buf : (100, 7)  →  15-element float64 feature vector.

    Converts accel m/s² → g before FFT because the tremor CSV was in g-units.
    Returns raw feature vector (caller must apply tremor_scaler.pkl).
    """
    accel_g = buf[:, IDX_AX : IDX_AZ + 1] / G     # m/s² → g
    axis    = _dominant_axis(accel_g)
    return _fft_features(axis, fs=FS)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Stress / HRV  (RandomForest, 8 HRV features)
# ─────────────────────────────────────────────────────────────────────────────

def _hrv_features(rr_or_bvp: np.ndarray, fs_bvp: float = FS) -> dict:
    """
    Identical to notebook 03 extract_hrv_features().
    Accepts raw BVP signal OR pre-extracted R-R intervals (ms).
    fs_bvp is passed as FS=50 for live data (not 64 — serial stream is 50 Hz).
    """
    rr = np.array(rr_or_bvp, dtype=np.float64)

    if len(rr) > 200 or (len(rr) > 0 and np.max(rr) < 200):
        peaks, _ = sp_signal.find_peaks(rr, distance=int(fs_bvp * 0.4))
        if len(peaks) < 2:
            return {k: 0.0 for k in HRV_FEATURES}
        rr = np.diff(peaks) * (1000.0 / fs_bvp)

    rr = rr[(rr > 300) & (rr < 2000)]
    if len(rr) < 3:
        return {k: 0.0 for k in HRV_FEATURES}

    diff_rr = np.diff(rr)
    rmssd   = float(np.sqrt(np.mean(diff_rr ** 2)))
    sdnn    = float(np.std(rr))
    pnn50   = float(np.sum(np.abs(diff_rr) > 50) / len(diff_rr))
    mean_rr = float(np.mean(rr))
    std_rr  = float(np.std(rr))
    hr_bpm  = float(60000.0 / mean_rr) if mean_rr > 0 else 0.0
    cv_rr   = float(std_rr  / mean_rr) if mean_rr > 0 else 0.0

    try:
        t_orig   = np.cumsum(rr) / 1000.0
        t_interp = np.arange(t_orig[0], t_orig[-1], 0.25)
        rr_interp = np.interp(t_interp, t_orig, rr)
        freqs, psd = sp_signal.welch(rr_interp, fs=4.0,
                                     nperseg=min(64, len(rr_interp)))
        lf  = float(np.trapz(psd[(freqs >= 0.04) & (freqs < 0.15)],
                              freqs[(freqs >= 0.04) & (freqs < 0.15)]))
        hf  = float(np.trapz(psd[(freqs >= 0.15) & (freqs < 0.40)],
                              freqs[(freqs >= 0.15) & (freqs < 0.40)]))
        lf_hf = float(lf / (hf + 1e-10))
    except Exception:
        lf_hf = 1.5

    return {
        "rmssd": rmssd, "sdnn": sdnn, "pnn50": pnn50,
        "mean_rr": mean_rr, "std_rr": std_rr,
        "hr_bpm": hr_bpm, "lf_hf_ratio": lf_hf, "cv_rr": cv_rr,
    }


def adapt_stress(buf: np.ndarray) -> np.ndarray:
    """
    buf : (3000, 7) — 60 s at 50 Hz.
    Returns 8 HRV features as float64 array (same order as HRV_FEATURES).
    """
    ppg   = buf[:, IDX_PPG].astype(np.float64)
    feats = _hrv_features(ppg, fs_bvp=FS)
    return np.array([feats[k] for k in HRV_FEATURES], dtype=np.float64)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Muscle atrophy  (GradientBoostingRegressor, 6 features)
# ─────────────────────────────────────────────────────────────────────────────

def adapt_atrophy(buf: np.ndarray,
                  immobilised_days: float,
                  age: float) -> np.ndarray:
    """
    buf : (N, 7) — uses all available samples in buffer.
    immobilised_days and age come from the UI input fields (not derivable from sensor).
    Returns 6-element float64 feature vector (caller must apply atrophy_scaler.pkl).
    """
    accel = buf[:, IDX_AX : IDX_AZ + 1]
    mag   = np.sqrt(np.sum(accel ** 2, axis=1))

    spatial_variance = float(np.var(mag))
    activity_level   = float(np.clip(spatial_variance * 10.0, 0.0, 10.0))

    ppg = buf[:, IDX_PPG].astype(np.float64)
    peaks, _ = sp_signal.find_peaks(ppg, distance=int(FS * 0.4))
    if len(peaks) > 2:
        rr_ms        = np.diff(peaks) * (1000.0 / FS)
        rr_ms        = rr_ms[(rr_ms > 300) & (rr_ms < 2000)]
        resting_hr   = float(60000.0 / np.mean(rr_ms)) if len(rr_ms) > 0 else 72.0
        hr_variability = float(np.std(rr_ms)) if len(rr_ms) > 0 else 30.0
    else:
        resting_hr, hr_variability = 72.0, 30.0

    return np.array([
        spatial_variance, resting_hr, hr_variability,
        float(immobilised_days), float(age), activity_level,
    ], dtype=np.float64)


# ─────────────────────────────────────────────────────────────────────────────
# 5. PT form  (LSTM, raw 6-axis, window=100)
# ─────────────────────────────────────────────────────────────────────────────

def adapt_pt_form(buf: np.ndarray) -> np.ndarray:
    """
    buf : (100, 7)  →  (100, 6) float32 raw IMU data.
    Caller must apply pt_form_scaler.pkl before passing to LSTM.
    """
    return buf[:, :6].astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# 7. Live sensor display  (no model)
# ─────────────────────────────────────────────────────────────────────────────

SENSOR_CHANNEL_LABELS = [
    "Accel X (m/s²)", "Accel Y (m/s²)", "Accel Z (m/s²)",
    "Gyro X (rad/s)", "Gyro Y (rad/s)", "Gyro Z (rad/s)",
    "PPG (raw)",
]

def adapt_sensor(buf: np.ndarray) -> dict:
    """buf : (N, 7)  →  dict mapping channel label → 1-D array."""
    return {label: buf[:, i] for i, label in enumerate(SENSOR_CHANNEL_LABELS)}


# ─────────────────────────────────────────────────────────────────────────────
# 6. PT form assessment (rule-based, no trained model yet)
# ─────────────────────────────────────────────────────────────────────────────

def compute_joint_angle(buf: np.ndarray) -> np.ndarray:
    """
    Estimate joint angle from gyroscope integration.
    buf : (N, 7).  Returns 1-D float32 array of angles in degrees, mean-centred.
    """
    gyro_x = buf[:, IDX_GX].astype(np.float64)
    dt     = 1.0 / FS
    angles = np.cumsum(np.degrees(gyro_x) * dt)
    return (angles - angles.mean()).astype(np.float32)


def evaluate_pt_form(buf: np.ndarray) -> dict:
    """
    Rule-based PT form classification from raw IMU window.
    buf : (N, 7).  Returns {"label", "peak_angle", "rom"}.

    Rules (degrees):
      ROM > 80  → overextension
      ROM < 20  → insufficient_range
      mean |gyro_z| > 1 rad/s  → wrong_plane (lateral drift)
      else      → correct
    """
    angles     = compute_joint_angle(buf)
    peak_angle = float(np.max(np.abs(angles)))
    rom        = float(np.ptp(angles))                    # peak-to-peak

    lateral    = float(np.abs(buf[:, IDX_GZ]).mean())

    if rom > 80.0:
        label = "overextension"
    elif rom < 20.0:
        label = "insufficient_range"
    elif lateral > 1.0:
        label = "wrong_plane"
    else:
        label = "correct"

    return {"label": label, "peak_angle": peak_angle, "rom": rom}


# ─────────────────────────────────────────────────────────────────────────────
# Backward-compatible aliases — ws_routes import these names
# ─────────────────────────────────────────────────────────────────────────────

# Tremor aliases
_TREMOR_FEATURE_KEYS = TREMOR_FEATURES


def dominant_axis(buf: np.ndarray) -> np.ndarray:
    """Return dominant accel axis (g-converted) from (N, 7) buffer."""
    accel_g = buf[:, IDX_AX : IDX_AZ + 1] / G
    return _dominant_axis(accel_g)


def extract_fft_features(signal_1d: np.ndarray, fs: float = FS) -> dict:
    """Return dict of FFT band-power metrics (used by tremor ws_route)."""
    x = np.asarray(signal_1d, dtype=np.float64)
    N = len(x)
    if N < 4:
        return {"dominant_freq": 0.0, "power_0_3hz": 0.0, "power_3_6hz": 0.0,
                "power_6_12hz": 0.0, "power_12plus": 0.0}
    x_ac      = x - np.mean(x)
    freqs     = fftfreq(N, d=1.0 / fs)[: N // 2]
    magnitude = np.abs(fft(x_ac))[: N // 2]
    di        = int(np.argmax(magnitude))

    def bp(lo, hi):
        mask = (freqs >= lo) & (freqs < hi)
        return float(np.sum(magnitude[mask] ** 2))

    return {
        "dominant_freq": float(freqs[di]),
        "power_0_3hz":   bp(0, 3),
        "power_3_6hz":   bp(3, 6),
        "power_6_12hz":  bp(6, 12),
        "power_12plus":  bp(12, 25),
    }


# Fall
def prepare_fall_input(buf: np.ndarray, target_len: int = WIN_FALL) -> np.ndarray:
    """(N, 7) → (1, target_len, 1) float32 ready for CNN.predict()."""
    feat = adapt_fall(buf)                                    # (WIN_FALL, 1)
    if feat.shape[0] != target_len:
        if feat.shape[0] > target_len:
            feat = feat[-target_len:]
        else:
            pad  = np.zeros((target_len - feat.shape[0], 1), dtype=np.float32)
            feat = np.vstack([pad, feat])
    return feat.reshape(1, target_len, 1)


# Tremor
def prepare_tremor_input(buf: np.ndarray, feature_names=None) -> np.ndarray:
    """(N, 7) → (1, 15) float64 feature array."""
    return adapt_tremor(buf).reshape(1, -1)


# Stress
def extract_hrv_features(ppg_signal: np.ndarray, fs: float = FS) -> dict:
    """Return HRV feature dict from raw PPG signal."""
    return _hrv_features(ppg_signal, fs_bvp=fs)


def prepare_stress_input(buf: np.ndarray, feat_names=None) -> np.ndarray:
    """(N, 7) → (1, 8) float64 HRV feature array."""
    return adapt_stress(buf).reshape(1, -1)


# Atrophy
def prepare_atrophy_input(buf: np.ndarray,
                          immobilised_days: float,
                          age: float,
                          feat_names=None) -> np.ndarray:
    """(N, 7) → (1, 6) float64 feature array."""
    return adapt_atrophy(buf, immobilised_days=immobilised_days, age=age).reshape(1, -1)
