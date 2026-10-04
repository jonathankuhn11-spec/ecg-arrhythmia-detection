import numpy as np

from conftest import FS, synthetic_record
from ecg import config
from ecg.features import bandpass, beat_features, feature_names, qrs_width_ms, rolling_median_prev


def test_split_follows_de_chazal_and_excludes_paced_records():
    assert len(config.DS1) == len(config.DS2) == 22
    assert not set(config.DS1) & set(config.DS2)
    assert not {102, 104, 107, 217} & set(config.DS1 + config.DS2)
    assert set(config.AAMI.values()) == {"N", "S", "V", "F", "Q"}


def test_bandpass_removes_baseline_wander_and_keeps_qrs_band():
    t = np.arange(20 * FS) / FS
    drift = np.sin(2 * np.pi * 0.1 * t)
    qrs_band = np.sin(2 * np.pi * 10 * t)
    core = slice(2 * FS, -2 * FS)                    # Ränder des Filters ausblenden
    assert np.std(bandpass(drift, FS)[core]) < 0.1 * np.std(drift[core])
    assert np.std(bandpass(qrs_band, FS)[core]) > 0.9 * np.std(qrs_band[core])


def test_rolling_median_uses_only_previous_values_and_skips_nan():
    values = np.array([np.nan, 2.0, 4.0, 6.0, 100.0])
    out = rolling_median_prev(values, n=3)
    assert out[1] == 2.0                             # kein gültiger Vorgänger: Wert selbst
    assert out[2] == 2.0
    assert out[3] == 3.0
    assert out[4] == 4.0                             # der Ausreißer selbst zählt nicht mit


def test_qrs_width_grows_with_pulse_width():
    t = np.arange(2 * FS) / FS
    narrow = np.exp(-0.5 * ((t - 1.0) / 0.012) ** 2)
    wide = np.exp(-0.5 * ((t - 1.0) / 0.035) ** 2)
    w_narrow, amp = qrs_width_ms(narrow, FS, FS)
    w_wide, _ = qrs_width_ms(wide, FS, FS)
    assert np.isclose(amp, 1.0, atol=0.01)
    assert 25 < w_narrow < 40                        # Halbwertsbreite = 2,355 * Sigma = 28 ms
    assert 70 < w_wide < 95                          # 2,355 * 35 ms = 82 ms
    assert w_wide > 2 * w_narrow


def test_beat_features_describe_premature_wide_beats(record):
    df = beat_features(record)
    assert list(df.columns[4:]) == feature_names(FS)
    assert not df[feature_names(FS)].isna().any().any()
    veb, normal = df[df["aami"] == "V"], df[df["aami"] == "N"]
    assert len(veb) == 9
    assert (veb["rr_ratio_pre"] < 0.75).all()        # 0,5 s statt 0,8 s: klar vorzeitig
    assert (veb["rr_ratio_post"] > 1.5).all()        # danach Pause von 1,1 s
    assert (veb["qrs_width_rel"] > 2.0).all()
    assert np.isclose(normal["rr_ratio_pre"].median(), 1.0, atol=0.01)
    assert np.isclose(normal["qrs_width_rel"].median(), 1.0, atol=0.05)


def test_beat_at_record_edge_is_dropped_without_distorting_its_neighbours():
    rec = synthetic_record(first_beat_s=0.1)         # erster Schlag ohne vollständiges Fenster
    df = beat_features(rec)
    assert df["sample"].iloc[0] > int(0.25 * FS)
    first = df.iloc[0]
    assert np.isclose(first["qrs_width_rel"], 1.0)
    assert np.isclose(first["amp_rel"], 1.0)


def test_class_q_is_excluded_from_the_feature_table(record):
    record.ann_symbols[3] = "Q"
    df = beat_features(record)
    assert "Q" not in set(df["aami"])
    assert len(df) == len(record.ann_symbols) - 1
