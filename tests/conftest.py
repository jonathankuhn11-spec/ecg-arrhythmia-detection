"""Gemeinsame Testbausteine: ein synthetischer EKG-Record mit bekannten Eigenschaften.

Die Tests brauchen keine MIT-BIH-Daten und laufen deshalb auch ohne Download.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ecg.data import Record  # noqa: E402

FS = 360


def synthetic_record(n_beats: int = 60, veb_every: int = 6, first_beat_s: float = 1.0) -> Record:
    """EKG aus Gauß-Impulsen: normale Schläge schmal und regelmäßig, VEB vorzeitig und breit."""
    times, symbols = [], []
    t = first_beat_s
    for i in range(n_beats):
        veb = veb_every and i > 0 and i % veb_every == 0
        times.append(t - 0.3 if veb else t)          # VEB fällt 300 ms zu früh ein
        symbols.append("V" if veb else "N")
        t += 0.8
    samples = (np.array(times) * FS).astype(int)
    grid = np.arange(int((t + 1.0) * FS)) / FS
    signal = np.zeros(len(grid))
    for time_s, sym in zip(times, symbols):
        sigma = 0.035 if sym == "V" else 0.012       # VEB etwa dreimal so breit
        signal += np.exp(-0.5 * ((grid - time_s) / sigma) ** 2)
    signal += 0.3 * np.sin(2 * np.pi * 0.15 * grid)  # langsame Grundlinienschwankung
    return Record(name=999, fs=FS, signal=signal, ann_samples=samples, ann_symbols=symbols)


@pytest.fixture
def record():
    return synthetic_record()
