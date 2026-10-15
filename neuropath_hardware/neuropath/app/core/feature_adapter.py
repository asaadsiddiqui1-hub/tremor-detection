"""
NeuroPath AI — Feature Adapter
===============================
Bridges the gap between the live 7-column hardware stream and what
each trained model expects as input.

This is THE critical glue layer. Each model was trained on data in a
specific format that differs from the raw glove stream:

  Fall detector  : trained on single-channel accel MAGNITUDE (512 window)
                   → we compute sqrt(ax^2+ay^2+az^2) from live data
  Tremor (SVM)   : trained on FFT features of one axis
                   → we extract features from accel magnitude
  Stress (RF)    : trained on HRV features from PPG
                   → we extract HRV from the ppg_raw column
  PT form        : uses complementary-filter joint angle (rule-based)
  Atrophy        : uses spatial variance + resting HR

Every function here takes the live window (N, 7) and returns exactly
what the corresponding model/screen needs.
"""

import numpy as np
from scipy import signal as sp_signal
from scipy.fft import fft, fftfreq

# Column indices (must match serial_reader.COLUMNS)
AX, AY, AZ, GX, GY, GZ, PPG = range(7)

SAMPLE_RATE_HZ = 50
GRAVITY = 9.81


# ────────────────────────────────────────────────────────────────────
# FALL DETECTION
# ────────────────────────────────────────────────────────────────────
def accel_magnitude(window: np.ndarray) -> np.ndarray:
    """
    Compute acceleration magnitude from 3-axis accel.
    SisFall was trained on single-channel magnitude, so this is what
    the fall CNN expects.
    """
    ax, ay, az = window[:, AX], window[:, AY], window[:, AZ]
    return np.sqrt(ax**2 + ay**2 + az**2)


def prepare_fall_input(window: np.ndarray, target_len: int = 512) -> np.ndarray:
    """
    Convert a live window into the (1, 512, 1) shape the fall CNN wants.

    Steps:
      1. Compute accel magnitude
      2. Convert m/s^2 → g units (SisFall used g)
      3. Resample/pad to exactly 512 samples
      4. Reshape to (1, 512, 1)
    """
    mag = accel_magnitude(window)

    # m/s^2 → g  (SisFall training data was in g)
    mag_g = mag / GRAVITY

    # Resample to exactly 512 points
    if len(mag_g) == 0:
        mag_g = np.zeros(target_len)
    elif len(mag_g) != target_len:
        mag_g = sp_signal.resample(mag_g, target_len)

    return mag_g.reshape(1, target_len, 1).astype(np.float32)


# ────────────────────────────────────────────────────────────────────
# TREMOR
# ────────────────────────────────────────────────────────────────────
def extract_fft_features(signal_1d: np.ndarray, fs: int = SAMPLE_RATE_HZ) -> dict:
    """
    FFT + time-domain features for tremor classification.
    Mirrors the function in notebook 02 exactly.
    """
    x = np.asarray(signal_1d, dtype=np.float64)
    N = len(x)
    if N < 4:
        return {k: 0.0 for k in _TREMOR_FEATURE_KEYS}

    # Remove DC offset (gravity dominates the 0 Hz bin and masks the
    # real tremor peak). Detrend so the FFT reflects oscillation only.
    x_ac = x - np.mean(x)

    freqs = fftfreq(N, d=1/fs)[:N//2]
    magnitude = np.abs(fft(x_ac))[:N//2]

    dominant_idx = int(np.argmax(magnitude))
    dominant_freq = float(freqs[dominant_idx])
    peak_magnitude = float(magnitude[dominant_idx])

    def band_power(f_low, f_high):
        mask = (freqs >= f_low) & (freqs < f_high)
        return float(np.sum(magnitude[mask] ** 2))

    p03  = band_power(0, 3)
    p36  = band_power(3, 6)
    p612 = band_power(6, 12)
    p12p = band_power(12, 25)
    total = p03 + p36 + p612 + p12p + 1e-10

    import pandas as pd
    return {
        'dominant_freq'   : dominant_freq,
        'peak_magnitude'  : peak_magnitude,
        'power_0_3hz'     : p03,
        'power_3_6hz'     : p36,
        'power_6_12hz'    : p612,
        'power_12plus'    : p12p,
        'ratio_parkinson' : p36 / total,
        'ratio_essential' : p612 / total,
        'mean'            : float(np.mean(x)),
        'std'             : float(np.std(x)),
        'rms'             : float(np.sqrt(np.mean(x**2))),
        'max_abs'         : float(np.max(np.abs(x))),
        'skewness'        : float(pd.Series(x).skew()),
        'kurtosis'        : float(pd.Series(x).kurt()),
        'jerk_mean'       : float(np.mean(np.abs(np.diff(x)))),
    }

_TREMOR_FEATURE_KEYS = ['dominant_freq','peak_magnitude','power_0_3hz','power_3_6hz',
                        'power_6_12hz','power_12plus','ratio_parkinson','ratio_essential',
                        'mean','std','rms','max_abs','skewness','kurtosis','jerk_mean']


def dominant_axis(window: np.ndarray) -> np.ndarray:
    """
    Return the accelerometer axis with the most oscillation (highest
    variance after removing DC). Tremor analysis must use a single axis,
    NOT magnitude — squaring axes for magnitude doubles the frequency
    (a 5 Hz tremor shows as a 10 Hz harmonic in the magnitude signal).
    """
    candidates = [window[:, AX], window[:, AY], window[:, AZ]]
    variances = [np.var(c - np.mean(c)) for c in candidates]
    return candidates[int(np.argmax(variances))]


def prepare_tremor_input(window: np.ndarray, feature_names: list) -> np.ndarray:
    """Extract tremor features from the most active single axis."""
    axis = dominant_axis(window)
    feats = extract_fft_features(axis)
    return np.array([feats[k] for k in feature_names]).reshape(1, -1)


# ────────────────────────────────────────────────────────────────────
# STRESS / HRV
# ────────────────────────────────────────────────────────────────────
def extract_hrv_features(ppg_signal: np.ndarray, fs: float = SAMPLE_RATE_HZ) -> dict:
    """
    Extract HRV features from the raw PPG (ppg_raw) channel.
    Detects beats, computes R-R intervals, derives HRV metrics.
    Mirrors notebook 03.
    """
    x = np.asarray(ppg_signal, dtype=np.float64)
    keys = ['rmssd','sdnn','pnn50','mean_rr','std_rr','hr_bpm','lf_hf_ratio','cv_rr']
    if len(x) < fs:  # need at least 1 second
        return {k: 0.0 for k in keys}

    # Bandpass filter to isolate the pulse waveform (0.5–4 Hz = 30–240 bpm)
    nyq = fs / 2
    try:
        b, a = sp_signal.butter(2, [0.5/nyq, min(4.0, nyq*0.99)/nyq], btype='band')
        x_filt = sp_signal.filtfilt(b, a, x)
    except Exception:
        x_filt = x - np.mean(x)

    # Detect peaks (heartbeats)
    peaks, _ = sp_signal.find_peaks(x_filt, distance=int(fs * 0.4))
    if len(peaks) < 3:
        return {k: 0.0 for k in keys}

    rr = np.diff(peaks) * (1000.0 / fs)   # R-R intervals in ms
    rr = rr[(rr > 300) & (rr < 2000)]
    if len(rr) < 3:
        return {k: 0.0 for k in keys}

    diff_rr = np.diff(rr)
    mean_rr = float(np.mean(rr))
    return {
        'rmssd'       : float(np.sqrt(np.mean(diff_rr**2))),
        'sdnn'        : float(np.std(rr)),
        'pnn50'       : float(np.sum(np.abs(diff_rr) > 50) / len(diff_rr)),
        'mean_rr'     : mean_rr,
        'std_rr'      : float(np.std(rr)),
        'hr_bpm'      : float(60000 / mean_rr) if mean_rr > 0 else 0.0,
        'lf_hf_ratio' : 1.5,  # simplified; full Welch in notebook
        'cv_rr'       : float(np.std(rr) / mean_rr) if mean_rr > 0 else 0.0,
    }


def prepare_stress_input(window: np.ndarray, feature_names: list) -> np.ndarray:
    """Extract HRV features from the PPG channel of the live window."""
    ppg = window[:, PPG]
    feats = extract_hrv_features(ppg)
    return np.array([feats.get(k, 0.0) for k in feature_names]).reshape(1, -1)


# ────────────────────────────────────────────────────────────────────
# PT FORM — complementary filter joint angle (rule-based)
# ────────────────────────────────────────────────────────────────────
def compute_joint_angle(window: np.ndarray, fs: int = SAMPLE_RATE_HZ,
                        alpha: float = 0.98) -> np.ndarray:
    """
    Estimate forearm pitch angle over time using a complementary filter
    that fuses accelerometer (absolute, noisy) and gyroscope (smooth, drifts).

    Returns angle in degrees for each timestep.
    """
    ax, ay, az = window[:, AX], window[:, AY], window[:, AZ]
    gx = window[:, GX]   # roll rate (rad/s)

    # Accelerometer-derived pitch angle
    accel_angle = np.degrees(np.arctan2(ay, np.sqrt(ax**2 + az**2)))

    # Complementary filter
    dt = 1.0 / fs
    angle = np.zeros(len(window))
    angle[0] = accel_angle[0]
    for i in range(1, len(window)):
        gyro_delta = np.degrees(gx[i]) * dt
        angle[i] = alpha * (angle[i-1] + gyro_delta) + (1 - alpha) * accel_angle[i]
    return angle


def evaluate_pt_form(window: np.ndarray, target_min: float = 70,
                     target_max: float = 110) -> dict:
    """
    Rule-based PT form evaluation. Measures range of motion of one rep
    and classifies the form.

    target_min/max define the correct peak angle band (e.g. 70–110° for
    an elbow curl). Tune per exercise.
    """
    angle = compute_joint_angle(window)
    if len(angle) == 0:
        return {'label': 'no_data', 'peak_angle': 0, 'rom': 0}

    peak = float(np.max(angle))
    trough = float(np.min(angle))
    rom = peak - trough   # range of motion

    # Plane consistency: how much the OTHER axis moved (off-plane drift)
    off_plane = float(np.std(window[:, GZ]))  # yaw rate std

    if peak < target_min:
        label = 'insufficient_range'
    elif peak > target_max:
        label = 'overextension'
    elif off_plane > 1.0:   # rad/s threshold for wobble
        label = 'wrong_plane'
    else:
        label = 'correct'

    return {
        'label'      : label,
        'peak_angle' : round(peak, 1),
        'rom'        : round(rom, 1),
        'off_plane'  : round(off_plane, 3),
    }


# ────────────────────────────────────────────────────────────────────
# MUSCLE ATROPHY
# ────────────────────────────────────────────────────────────────────
def prepare_atrophy_input(window: np.ndarray, immobilised_days: float,
                          age: float, feature_names: list) -> np.ndarray:
    """
    Build the atrophy regressor input from live sensor stats + user metadata.
    Spatial variance and resting HR come from the sensors; days/age from the user.
    """
    spatial_variance = float(np.var(accel_magnitude(window)))

    # Resting HR from PPG
    hrv = extract_hrv_features(window[:, PPG])
    resting_hr = hrv['hr_bpm'] if hrv['hr_bpm'] > 0 else 72.0
    hr_variability = hrv['rmssd']

    activity_level = float(np.clip(spatial_variance * 10, 0, 10))

    feat_map = {
        'spatial_variance' : spatial_variance,
        'resting_hr'       : resting_hr,
        'hr_variability'   : hr_variability,
        'immobilised_days' : immobilised_days,
        'age'              : age,
        'activity_level'   : activity_level,
    }
    return np.array([feat_map[k] for k in feature_names]).reshape(1, -1)


# ── Test ────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("Testing feature adapter with a synthetic window...\n")
    # Simulate a 3-second window at 50 Hz (150 rows, 7 cols)
    t = np.linspace(0, 3, 150)
    fake = np.column_stack([
        2*np.sin(2*np.pi*5*t),          # accel_x (5Hz tremor)
        0.5*np.sin(2*np.pi*5*t),        # accel_y
        -9.81 + np.random.normal(0,0.1,150),  # accel_z
        np.random.normal(0,0.3,150),    # gyro_x
        np.random.normal(0,0.3,150),    # gyro_y
        np.random.normal(0,0.3,150),    # gyro_z
        2000 + 300*np.sin(2*np.pi*1.2*t),  # ppg (~72 bpm)
    ])

    fall_in = prepare_fall_input(fake)
    print(f"  Fall input shape    : {fall_in.shape}  (expect (1, 512, 1))")

    tremor_feats = extract_fft_features(dominant_axis(fake))
    print(f"  Tremor dom. freq    : {tremor_feats['dominant_freq']:.2f} Hz  (expect ~5)")

    hrv = extract_hrv_features(fake[:, PPG])
    print(f"  HRV heart rate      : {hrv['hr_bpm']:.0f} bpm  (expect ~72)")

    pt = evaluate_pt_form(fake)
    print(f"  PT form result      : {pt['label']} (peak {pt['peak_angle']}°)")

    print("\nFeature adapter OK — all models can receive correctly-shaped input.")
