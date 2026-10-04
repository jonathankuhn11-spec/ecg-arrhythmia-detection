"""Zwei Detektoren für ventrikuläre Extrasystolen (VEB): Regelwerk und gelerntes Modell.

Beide beantworten dieselbe Frage je Herzschlag: VEB ja oder nein.
Beide werden ausschließlich auf DS1 eingestellt und auf DS2 geprüft.
"""
import itertools

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from . import config
from .features import feature_names

# Zwei Varianten desselben Regelwerks. Sie unterscheiden sich nur darin, wie "breit" gemessen wird:
# relativ zu den vorangegangenen Schlägen des Patienten oder absolut in Millisekunden.
RULE_VARIANTS = {
    "relativ": {
        "feature": "qrs_width_rel",
        "wide": np.round(np.arange(1.00, 1.80, 0.05), 2),
        "very_wide": np.array([1.4, 1.6, 1.8, 2.0, 2.5, 3.0]),
    },
    "absolut": {
        "feature": "qrs_width_ms",
        "wide": np.arange(30.0, 125.0, 5.0),
        "very_wide": np.array([80.0, 100.0, 120.0, 140.0, 160.0, 400.0]),   # 400 ms = praktisch aus
    },
}
PREMATURE_GRID = np.round(np.arange(0.70, 1.00, 0.02), 2)   # rr_ratio_pre unterhalb = vorzeitig


def is_veb(df: pd.DataFrame) -> np.ndarray:
    """Zielgröße: gehört der Schlag zur AAMI-Klasse V?"""
    return (df["aami"] == "V").to_numpy()


def rule_predict(df: pd.DataFrame, params: dict) -> np.ndarray:
    """Regel: (vorzeitig UND verbreitert) ODER stark verbreitert.

    Das bildet die klinische Beschreibung einer ventrikulären Extrasystole nach:
    Sie fällt zu früh ein und hat einen breiten, deformierten QRS-Komplex.
    """
    premature = df["rr_ratio_pre"].to_numpy() < params["premature"]
    width = df[params["feature"]].to_numpy()
    return (premature & (width > params["wide"])) | (width > params["very_wide"])


def calibrate_rule(train: pd.DataFrame, variant: str) -> dict:
    """Wählt die drei Schwellen per Gittersuche so, dass der F1-Wert auf DS1 maximal wird."""
    grid = RULE_VARIANTS[variant]
    y = is_veb(train)
    best = {"f1_train": -1.0}
    for a, b, c in itertools.product(PREMATURE_GRID, grid["wide"], grid["very_wide"]):
        params = {"variant": variant, "feature": grid["feature"],
                  "premature": float(a), "wide": float(b), "very_wide": float(c)}
        pred = rule_predict(train, params)
        f1 = 2 * np.sum(y & pred) / max(np.sum(y) + np.sum(pred), 1)
        if f1 > best["f1_train"]:
            best = {**params, "f1_train": float(f1)}
    return best


def train_model(train: pd.DataFrame) -> RandomForestClassifier:
    """Random Forest auf allen Merkmalen. Die Klassengewichtung gleicht aus, dass VEB selten sind."""
    model = RandomForestClassifier(
        n_estimators=200, min_samples_leaf=3, class_weight="balanced_subsample",
        n_jobs=-1, random_state=config.SEED)
    model.fit(train[feature_names()], is_veb(train))
    return model


def model_scores(model: RandomForestClassifier, df: pd.DataFrame) -> np.ndarray:
    """Geschätzte VEB-Wahrscheinlichkeit je Schlag (0 bis 1)."""
    return model.predict_proba(df[feature_names()])[:, 1]
