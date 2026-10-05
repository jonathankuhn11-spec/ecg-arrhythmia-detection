"""Kennzahlen für Detektoren.

Begriffe:
  Sensitivität (Se)            Anteil der echten Ereignisse, die erkannt wurden
  positive Prädiktivität (+P)  Anteil der Meldungen, die echte Ereignisse sind
  Fehlalarme pro Stunde        falsch-positive Meldungen je Stunde EKG
"""
import numpy as np


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray, hours: float) -> dict:
    """Kennzahlen eines Ja/Nein-Detektors gegen die Referenz."""
    y_true, y_pred = np.asarray(y_true, bool), np.asarray(y_pred, bool)
    tp = int(np.sum(y_true & y_pred))
    fp = int(np.sum(~y_true & y_pred))
    fn = int(np.sum(y_true & ~y_pred))
    tn = int(np.sum(~y_true & ~y_pred))
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "sensitivity": tp / max(tp + fn, 1),
        "ppv": tp / max(tp + fp, 1),
        "fpr": fp / max(fp + tn, 1),
        "false_alarms_per_hour": fp / hours,
    }


def threshold_for_sensitivity(scores: np.ndarray, y_true: np.ndarray, target_tp: int) -> float:
    """Schwelle, bei der ein Modell genau `target_tp` echte Ereignisse erkennt.

    Dient dem Vergleich zweier Detektoren bei gleicher Sensitivität.
    """
    positives = np.sort(np.asarray(scores)[np.asarray(y_true, bool)])[::-1]
    target_tp = int(np.clip(target_tp, 1, len(positives)))
    return float(positives[target_tp - 1])


def match_peaks(reference: np.ndarray, detected: np.ndarray, tolerance: int) -> tuple:
    """Ordnet detektierte R-Zacken den Referenzschlägen eins zu eins zu.

    Beide Folgen müssen aufsteigend sortiert sein. Rückgabe: (TP, FP, FN).
    """
    i = j = tp = 0
    while i < len(reference) and j < len(detected):
        diff = detected[j] - reference[i]
        if abs(diff) <= tolerance:
            tp += 1
            i += 1
            j += 1
        elif diff < 0:
            j += 1          # Detektion ohne Referenzschlag in Reichweite
        else:
            i += 1          # Referenzschlag wurde verpasst
    return tp, len(detected) - tp, len(reference) - tp


def best_f1_threshold(scores: np.ndarray, y_true: np.ndarray) -> float:
    """Schwelle mit dem höchsten F1-Wert auf einer Validierungsmenge (nie auf der Testmenge wählen)."""
    scores, y_true = np.asarray(scores), np.asarray(y_true, bool)
    order = np.argsort(-scores)
    tp = np.cumsum(y_true[order])
    fp = np.cumsum(~y_true[order])
    f1 = 2 * tp / (tp + fp + y_true.sum())
    k = int(np.argmax(f1))
    return float(scores[order][k])
