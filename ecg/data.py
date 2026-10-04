"""Datenzugriff: MIT-BIH Arrhythmia Database über WFDB laden."""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import wfdb

from . import config


@dataclass
class Record:
    """Ein EKG-Record: eine Ableitung plus die Referenzannotationen der Kardiologen."""
    name: int
    fs: int
    signal: np.ndarray        # Rohsignal der Ableitung in mV
    ann_samples: np.ndarray   # Abtastindex jeder Annotation
    ann_symbols: list         # MIT-BIH-Symbol jeder Annotation, etwa "N" oder "V"

    @property
    def hours(self) -> float:
        return len(self.signal) / self.fs / 3600


def download(data_dir: Path, records=None) -> None:
    """Lädt Signal-, Header- und Annotationsdateien von PhysioNet (rund 90 MB)."""
    records = records or (config.DS1 + config.DS2)
    data_dir.mkdir(parents=True, exist_ok=True)
    wfdb.dl_database("mitdb", str(data_dir),
                     records=[str(r) for r in records], annotators=["atr"])


def missing_records(data_dir: Path, records) -> list:
    """Gibt die Records zurück, für die lokal Dateien fehlen."""
    return [r for r in records
            if not all((data_dir / f"{r}.{ext}").exists() for ext in ("hea", "dat", "atr"))]


def load_record(data_dir: Path, name: int, lead: str = config.LEAD) -> Record:
    """Liest einen Record und wählt die Ableitung über ihren Namen aus.

    Die Auswahl über den Namen ist nötig, weil die Kanalreihenfolge nicht in
    allen Records gleich ist (in Record 114 steht MLII an zweiter Stelle).
    """
    path = str(data_dir / str(name))
    rec = wfdb.rdrecord(path)
    ann = wfdb.rdann(path, "atr")
    if lead not in rec.sig_name:
        raise ValueError(f"Record {name} enthält die Ableitung {lead} nicht: {rec.sig_name}")
    signal = rec.p_signal[:, rec.sig_name.index(lead)]
    return Record(name=name, fs=int(rec.fs), signal=signal,
                  ann_samples=np.asarray(ann.sample), ann_symbols=list(ann.symbol))
