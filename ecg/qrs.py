"""R-Zacken-Detektion mit NeuroKit2 und Prüfung gegen die Referenzannotationen.

Die Schlagklassifikation in dieser Baseline arbeitet auf den Referenzpositionen
(üblich in der Literatur). Dieses Modul misst getrennt davon, wie zuverlässig ein
Detektor die Schläge überhaupt findet. Das ist die Voraussetzung für jeden
Einsatz ohne manuelle Annotation.
"""
import warnings

import neurokit2 as nk
import numpy as np
import pandas as pd

from . import config
from .data import load_record
from .evaluate import match_peaks


def detect_r_peaks(signal: np.ndarray, fs: int) -> np.ndarray:
    """Positionen der R-Zacken als aufsteigend sortierte Abtastindizes."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cleaned = nk.ecg_clean(signal, sampling_rate=fs, method="neurokit")
        _, info = nk.ecg_peaks(cleaned, sampling_rate=fs, method="neurokit")
    return np.sort(np.asarray(info["ECG_R_Peaks"], dtype=int))


def evaluate_qrs(data_dir, records, progress=None) -> pd.DataFrame:
    """Sensitivität und positive Prädiktivität der R-Zacken-Detektion je Record."""
    rows = []
    for name in records:
        if progress:
            progress(name)
        rec = load_record(data_dir, name)
        reference = np.array([s for s, sym in zip(rec.ann_samples, rec.ann_symbols)
                              if sym in config.AAMI])
        detected = detect_r_peaks(rec.signal, rec.fs)
        tp, fp, fn = match_peaks(reference, detected, int(config.QRS_TOL_S * rec.fs))
        rows.append({"record": name, "tp": tp, "fp": fp, "fn": fn,
                     "sensitivity": tp / max(tp + fn, 1), "ppv": tp / max(tp + fp, 1)})
    return pd.DataFrame(rows)
