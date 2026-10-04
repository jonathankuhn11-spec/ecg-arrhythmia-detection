import neurokit2 as nk
import numpy as np
import wfdb

from conftest import FS
from ecg import data, qrs
from ecg.evaluate import match_peaks


def _write_wfdb_record(folder, name="900"):
    """Schreibt ein simuliertes EKG im echten MIT-BIH-Dateiformat (Header, Signal, Annotation)."""
    ecg = nk.ecg_simulate(duration=30, sampling_rate=FS, heart_rate=70, noise=0.01, random_state=1)
    peaks = qrs.detect_r_peaks(ecg, FS)
    two_leads = np.column_stack([-ecg, ecg])                  # MLII absichtlich an zweiter Stelle
    wfdb.wrsamp(name, fs=FS, units=["mV", "mV"], sig_name=["V5", "MLII"], p_signal=two_leads,
                fmt=["16", "16"], write_dir=str(folder))
    wfdb.wrann(name, "atr", sample=peaks, symbol=["N"] * len(peaks), write_dir=str(folder))
    return ecg, peaks


def test_load_record_selects_lead_by_name(tmp_path):
    ecg, peaks = _write_wfdb_record(tmp_path)
    rec = data.load_record(tmp_path, 900)
    assert rec.fs == FS
    assert np.corrcoef(rec.signal, ecg)[0, 1] > 0.999         # MLII, nicht der erste Kanal
    assert rec.ann_samples.tolist() == peaks.tolist()
    assert set(rec.ann_symbols) == {"N"}
    assert np.isclose(rec.hours, 30 / 3600)


def test_missing_records_reports_incomplete_downloads(tmp_path):
    _write_wfdb_record(tmp_path)
    assert data.missing_records(tmp_path, [900]) == []
    assert data.missing_records(tmp_path, [900, 901]) == [901]


def test_r_peak_detection_finds_the_simulated_heartbeats():
    ecg = nk.ecg_simulate(duration=60, sampling_rate=FS, heart_rate=70, noise=0.01, random_state=2)
    peaks = qrs.detect_r_peaks(ecg, FS)
    assert 66 <= len(peaks) <= 72                             # 70 Schläge pro Minute
    assert np.all(np.diff(peaks) > 0)
    rr = np.diff(peaks) / FS
    assert np.isclose(np.median(rr), 60 / 70, atol=0.03)


def test_evaluate_qrs_on_a_record_with_known_annotations(tmp_path):
    _write_wfdb_record(tmp_path)
    table = qrs.evaluate_qrs(tmp_path, [900])
    assert table.loc[0, "sensitivity"] == 1.0 and table.loc[0, "ppv"] == 1.0
    assert match_peaks(np.array([10]), np.array([400]), tolerance=54) == (0, 1, 1)
