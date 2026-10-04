import numpy as np
import pandas as pd

from conftest import FS, synthetic_record
from ecg import detectors
from ecg.features import beat_features, feature_names


def _table(rows):
    cols = ["aami", "rr_ratio_pre", "qrs_width_rel", "qrs_width_ms"]
    return pd.DataFrame(rows, columns=cols)


def test_rule_needs_premature_and_wide_or_very_wide():
    params = {"feature": "qrs_width_rel", "premature": 0.9, "wide": 1.2, "very_wide": 2.0}
    df = _table([
        ("V", 0.7, 1.5, 60.0),    # vorzeitig und verbreitert        -> VEB
        ("N", 0.7, 1.0, 30.0),    # nur vorzeitig                    -> kein VEB
        ("N", 1.0, 1.5, 60.0),    # nur verbreitert                  -> kein VEB
        ("V", 1.0, 2.5, 90.0),    # stark verbreitert, nicht vorzeitig -> VEB
    ])
    assert detectors.rule_predict(df, params).tolist() == [True, False, False, True]


def test_calibration_finds_thresholds_that_separate_clean_data():
    rng = np.random.default_rng(0)
    normal = [("N", rng.uniform(0.95, 1.05), rng.uniform(0.9, 1.1), 30.0) for _ in range(300)]
    veb = [("V", rng.uniform(0.5, 0.8), rng.uniform(1.5, 1.7), 60.0) for _ in range(30)]
    df = _table(normal + veb)
    for variant in detectors.RULE_VARIANTS:
        params = detectors.calibrate_rule(df, variant)
        assert params["variant"] == variant
        assert params["f1_train"] == 1.0
        assert detectors.rule_predict(df, params).tolist() == detectors.is_veb(df).tolist()


def test_model_learns_to_detect_veb_on_synthetic_records():
    train = pd.concat([beat_features(synthetic_record(n_beats=120, veb_every=k)) for k in (5, 7)])
    test = beat_features(synthetic_record(n_beats=90, veb_every=6))
    model = detectors.train_model(train)
    scores = detectors.model_scores(model, test)
    assert scores.shape == (len(test),)
    assert ((scores >= 0.5) == detectors.is_veb(test)).mean() > 0.98
    assert len(model.feature_importances_) == len(feature_names(FS))
