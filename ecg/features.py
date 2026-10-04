"""Vorverarbeitung und Merkmalsextraktion je Herzschlag.

Drei Merkmalsgruppen beschreiben einen Schlag:
  1. Rhythmus:    RR-Intervalle und ihr Verhältnis zum lokalen Rhythmus (Vorzeitigkeit, Pause danach)
  2. QRS-Breite:  absolut und relativ zu den vorangegangenen Schlägen desselben Patienten
  3. Morphologie: die Schlagform als 26 Mittelwerte über je 25 ms
"""
import numpy as np
import pandas as pd
from scipy import signal as sps

from . import config
from .data import Record, load_record

RR_FEATURES = ["rr_pre", "rr_post", "rr_local", "rr_ratio_pre", "rr_ratio_post"]
SHAPE_FEATURES = ["qrs_width_ms", "qrs_width_rel", "amp_rel"]


def n_bins(fs: int = config.FS) -> int:
    """Anzahl der Morphologie-Abschnitte im Schlagfenster."""
    return (int(config.WIN_PRE_S * fs) + int(config.WIN_POST_S * fs)) // int(config.BIN_S * fs)


def feature_names(fs: int = config.FS) -> list:
    return RR_FEATURES + SHAPE_FEATURES + [f"m{i:02d}" for i in range(n_bins(fs))]


def bandpass(x: np.ndarray, fs: int) -> np.ndarray:
    """Butterworth-Bandpass, vorwärts und rückwärts angewendet (keine Phasenverschiebung)."""
    sos = sps.butter(config.FILTER_ORDER, config.BANDPASS_HZ, btype="bandpass", fs=fs, output="sos")
    return sps.sosfiltfilt(sos, x)


def qrs_width_ms(x: np.ndarray, center: int, fs: int) -> tuple:
    """Schätzt die QRS-Breite als Halbwertsbreite des größten Ausschlags um die R-Zacke.

    Rückgabe: (Breite in ms, Amplitude des Ausschlags in mV).
    """
    half = int(config.WIDTH_HALF_S * fs)
    seg = x[center - half:center + half + 1]
    dev = np.abs(seg - np.median(seg))            # Abstand zur Grundlinie des Fensters
    search = int(0.05 * fs)                        # Annotation kann leicht neben dem Maximum liegen
    peak = half - search + int(np.argmax(dev[half - search:half + search + 1]))
    threshold = 0.5 * dev[peak]
    left = right = peak
    while left > 0 and dev[left] > threshold:
        left -= 1
    while right < len(dev) - 1 and dev[right] > threshold:
        right += 1
    return (right - left) / fs * 1000, float(dev[peak])


def rolling_median_prev(values: np.ndarray, n: int) -> np.ndarray:
    """Median der n vorangegangenen Werte. Der aktuelle Wert zählt nicht mit (kausal).

    Fehlende Werte (NaN) werden übersprungen. Gibt es noch keinen gültigen Vorgänger,
    dient der Wert selbst als Referenz.
    """
    out = np.empty(len(values))
    for i in range(len(values)):
        prev = values[max(0, i - n):i]
        prev = prev[~np.isnan(prev)]
        out[i] = np.median(prev) if len(prev) else values[i]
    return out


def beat_features(rec: Record) -> pd.DataFrame:
    """Berechnet eine Merkmalszeile je annotiertem Herzschlag eines Records."""
    fs = rec.fs
    x = bandpass(rec.signal, fs)
    pre, post = int(config.WIN_PRE_S * fs), int(config.WIN_POST_S * fs)
    bin_len = int(config.BIN_S * fs)
    bins = (pre + post) // bin_len

    # Nur Schlagannotationen behalten; Rhythmus- und Rauschmarker fallen weg.
    is_beat = np.array([s in config.AAMI for s in rec.ann_symbols])
    samples = rec.ann_samples[is_beat]
    symbols = np.array(rec.ann_symbols)[is_beat]
    aami = np.array([config.AAMI[s] for s in symbols])

    # --- Rhythmus: RR-Intervalle in Sekunden
    rr = np.diff(samples) / fs
    rr_pre = np.r_[rr[0], rr]                      # erster Schlag hat keinen Vorgänger
    rr_post = np.r_[rr, rr[-1]]                    # letzter Schlag hat keinen Nachfolger
    rr_local = rolling_median_prev(rr_pre, config.RR_LOCAL_BEATS)

    # --- QRS-Breite und Morphologie
    margin = max(pre, int(config.WIDTH_HALF_S * fs))
    complete = (samples - margin >= 0) & (samples + max(post, margin) < len(x))
    width = np.full(len(samples), np.nan)          # NaN: Schlag liegt zu nah am Rand des Records
    amp = np.full(len(samples), np.nan)
    shape = np.zeros((len(samples), bins))
    for i, s in enumerate(samples):
        if not complete[i]:
            continue
        width[i], amp[i] = qrs_width_ms(x, s, fs)
        window = x[s - pre:s + post][:bins * bin_len]
        shape[i] = (window - np.median(window)).reshape(bins, bin_len).mean(axis=1)
    shape /= np.median(amp[complete])              # Verstärkung des Records herausrechnen

    width_ref = np.maximum(rolling_median_prev(width, config.REF_BEATS), 1.0)
    amp_ref = np.maximum(rolling_median_prev(amp, config.REF_BEATS), 1e-6)

    df = pd.DataFrame({
        "record": rec.name, "sample": samples, "symbol": symbols, "aami": aami,
        "rr_pre": rr_pre, "rr_post": rr_post, "rr_local": rr_local,
        "rr_ratio_pre": rr_pre / np.maximum(rr_local, 1e-3),    # < 1: Schlag kommt zu früh
        "rr_ratio_post": rr_post / np.maximum(rr_pre, 1e-3),    # > 1: Pause nach dem Schlag
        "qrs_width_ms": width,
        "qrs_width_rel": width / width_ref,                     # > 1: breiter als gewohnt
        "amp_rel": amp / amp_ref,
    })
    df[[f"m{i:02d}" for i in range(bins)]] = shape

    # Randschläge ohne vollständiges Fenster und die Klasse Q (hier nur 15 Schläge) entfallen.
    keep = complete & (aami != "Q")
    return df[keep].reset_index(drop=True)


def build_dataset(data_dir, records, progress=None) -> tuple:
    """Merkmalstabelle für mehrere Records. Rückgabe: (Tabelle, Aufnahmedauer in Stunden)."""
    frames, hours = [], 0.0
    for name in records:
        if progress:
            progress(name)
        rec = load_record(data_dir, name)
        frames.append(beat_features(rec))
        hours += rec.hours
    return pd.concat(frames, ignore_index=True), hours
