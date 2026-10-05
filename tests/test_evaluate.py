import numpy as np

from ecg.evaluate import best_f1_threshold, binary_metrics, match_peaks, threshold_for_sensitivity


def test_binary_metrics_counts_and_rates():
    y_true = np.array([1, 1, 1, 0, 0, 0, 0, 0], bool)
    y_pred = np.array([1, 1, 0, 1, 0, 0, 0, 0], bool)
    m = binary_metrics(y_true, y_pred, hours=2.0)
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (2, 1, 1, 4)
    assert np.isclose(m["sensitivity"], 2 / 3)
    assert np.isclose(m["ppv"], 2 / 3)
    assert np.isclose(m["fpr"], 1 / 5)
    assert np.isclose(m["false_alarms_per_hour"], 0.5)


def test_binary_metrics_without_positives_does_not_divide_by_zero():
    m = binary_metrics(np.zeros(4, bool), np.zeros(4, bool), hours=1.0)
    assert m["sensitivity"] == 0 and m["ppv"] == 0


def test_match_peaks_one_to_one_within_tolerance():
    reference = np.array([100, 200, 300, 400])
    detected = np.array([102, 250, 298, 399, 500])   # 250 und 500 passen zu keinem Schlag
    tp, fp, fn = match_peaks(reference, detected, tolerance=5)
    assert (tp, fp, fn) == (3, 2, 1)


def test_match_peaks_does_not_count_a_reference_beat_twice():
    tp, fp, fn = match_peaks(np.array([100]), np.array([98, 101]), tolerance=5)
    assert (tp, fp, fn) == (1, 1, 0)


def test_threshold_for_sensitivity_hits_requested_number_of_true_positives():
    scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4])
    y_true = np.array([1, 0, 1, 1, 0, 1], bool)
    threshold = threshold_for_sensitivity(scores, y_true, target_tp=2)
    assert threshold == 0.7
    assert np.sum((scores >= threshold) & y_true) == 2


def test_best_f1_threshold_separates_clean_scores():
    scores = np.array([0.9, 0.8, 0.3, 0.2, 0.1, 0.05])
    y_true = np.array([1, 1, 0, 0, 0, 0], bool)
    thr = best_f1_threshold(scores, y_true)
    assert 0.3 < thr <= 0.8
    assert np.array_equal(scores >= thr, y_true)
