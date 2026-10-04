"""EKG-Arrhythmie-Erkennung: komplette Baseline in einem Lauf.

Ablauf:
  1. Daten prüfen (mit --download zuerst von PhysioNet laden)
  2. Merkmale je Herzschlag berechnen (DS1 = Training, DS2 = Test)
  3. Regelwerk auf DS1 kalibrieren, Random Forest auf DS1 trainieren
  4. Beide Detektoren auf DS2 auswerten und vergleichen
  5. R-Zacken-Detektion gegen die Referenz prüfen
  6. Kennzahlen, Bericht und Abbildungen nach results/ schreiben

Aufruf:
  python run_pipeline.py --download     (erster Lauf)
  python run_pipeline.py                (danach)
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from ecg import config, data, detectors, evaluate, features, qrs, report


def parse_args():
    p = argparse.ArgumentParser(description="Baseline zur Erkennung ventrikulärer Extrasystolen auf MIT-BIH")
    p.add_argument("--data-dir", type=Path, default=Path("data/mitdb"), help="Ordner mit den MIT-BIH-Dateien")
    p.add_argument("--out-dir", type=Path, default=Path("results"), help="Zielordner für Bericht und Abbildungen")
    p.add_argument("--download", action="store_true", help="Daten vor dem Lauf von PhysioNet laden")
    p.add_argument("--skip-qrs", action="store_true", help="Prüfung der R-Zacken-Detektion überspringen")
    return p.parse_args()


def progress(label):
    def show(name):
        print(f"\r  {label}: Record {name}   ", end="", flush=True)
    return show


def main():
    args = parse_args()
    t0 = time.time()
    all_records = config.DS1 + config.DS2

    # 1. Daten
    if args.download:
        print("Lade MIT-BIH Arrhythmia Database von PhysioNet ...")
        data.download(args.data_dir, all_records)
    missing = data.missing_records(args.data_dir, all_records)
    if missing:
        sys.exit(f"Es fehlen Dateien für {len(missing)} Records in {args.data_dir} (z. B. {missing[0]}).\n"
                 f"Starte einmal mit:  python run_pipeline.py --download")

    # 2. Merkmale
    print("Berechne Merkmale ...")
    train, train_hours = features.build_dataset(args.data_dir, config.DS1, progress("DS1"))
    test, test_hours = features.build_dataset(args.data_dir, config.DS2, progress("DS2"))
    y_train, y_test = detectors.is_veb(train), detectors.is_veb(test)
    print(f"\r  {len(train)} Schläge in DS1, {len(test)} Schläge in DS2" + " " * 20)

    # 3. Detektoren einstellen (nur DS1)
    print("Kalibriere Regelwerk auf DS1 ...")
    rules = {variant: detectors.calibrate_rule(train, variant) for variant in detectors.RULE_VARIANTS}
    primary = max(rules, key=lambda v: rules[v]["f1_train"])     # Auswahl nur anhand von DS1
    print("Trainiere Random Forest auf DS1 ...")
    model = detectors.train_model(train)

    # 4. Auswertung auf DS2
    scores = detectors.model_scores(model, test)
    model_pred = scores >= 0.5
    m_model = evaluate.binary_metrics(y_test, model_pred, test_hours)

    rule_results, rule_preds = {}, {}
    for variant, params in rules.items():
        pred = detectors.rule_predict(test, params)
        m_test = evaluate.binary_metrics(y_test, pred, test_hours)
        # Vergleich bei gleicher Sensitivität: Modellschwelle so legen, dass es gleich viele VEB findet
        iso_threshold = evaluate.threshold_for_sensitivity(scores, y_test, m_test["tp"])
        m_iso = evaluate.binary_metrics(y_test, scores >= iso_threshold, test_hours)
        rule_preds[variant] = pred
        rule_results[variant] = {
            "params": params,
            "train": evaluate.binary_metrics(y_train, detectors.rule_predict(train, params), train_hours),
            "test": m_test,
            "model_iso": m_iso, "iso_threshold": iso_threshold,
            "false_alarm_reduction_iso": 1 - m_iso["fp"] / max(m_test["fp"], 1),
        }
    rule_pred = rule_preds[primary]

    classes = test["aami"].to_numpy()
    fp_by_class = {
        key: {c: int(np.sum(pred & ~y_test & (classes == c))) for c in ("N", "S", "F")}
        for key, pred in (("rule", rule_pred), ("model", model_pred))
    }
    per_record = (
        test.assign(veb=y_test, fp_rule=rule_pred & ~y_test, fp_model=model_pred & ~y_test,
                    fn_rule=~rule_pred & y_test, fn_model=~model_pred & y_test)
        .groupby("record")[["veb", "fp_rule", "fp_model", "fn_rule", "fn_model"]].sum()
        .astype(int).reset_index()
    )
    names = features.feature_names()
    order = np.argsort(model.feature_importances_)[::-1][:8]

    metrics = {
        "data": {
            "train_records": len(config.DS1), "test_records": len(config.DS2),
            "train_beats": len(train), "test_beats": len(test),
            "train_veb": int(y_train.sum()), "test_veb": int(y_test.sum()),
            "train_hours": train_hours, "test_hours": test_hours,
        },
        "primary_rule": primary,
        "rules": rule_results,
        "model": {
            "type": "RandomForestClassifier", "n_features": len(names), "threshold": 0.5,
            "test": m_model,
            "false_alarm_reduction_vs_primary_rule": 1 - m_model["fp"] / max(rule_results[primary]["test"]["fp"], 1),
            "top_features": [{"feature": names[i], "importance": float(model.feature_importances_[i])}
                             for i in order],
        },
        "errors_test": {"fp_by_class": fp_by_class, "per_record": per_record.to_dict(orient="records")},
    }

    # 5. R-Zacken-Detektion
    if not args.skip_qrs:
        print("Prüfe R-Zacken-Detektion (NeuroKit2) ...")
        q = qrs.evaluate_qrs(args.data_dir, all_records, progress("QRS"))
        tp, fp, fn = int(q["tp"].sum()), int(q["fp"].sum()), int(q["fn"].sum())
        worst = q.assign(score=q["sensitivity"] + q["ppv"]).sort_values("score").head(5)
        metrics["qrs"] = {
            "records": len(q), "tp": tp, "fp": fp, "fn": fn,
            "sensitivity": tp / (tp + fn), "ppv": tp / (tp + fp),
            "worst": worst.drop(columns="score").to_dict(orient="records"),
        }
        print("\r" + " " * 40, end="\r")

    # 6. Ausgabe
    fig_dir = args.out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    report.plot_strip(args.data_dir, test, rule_pred, model_pred, fig_dir / "ekg_ausschnitt.png")
    report.plot_precision_recall(y_test, scores, metrics, fig_dir / "veb_precision_recall.png")
    report.plot_false_alarms(per_record, fig_dir / "fehlalarme_je_record.png")
    (args.out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    report.write_report(metrics, args.out_dir / "report.md")

    print(f"\nVEB-Erkennung auf DS2 ({len(config.DS2)} Records, {test_hours:.1f} h)")
    print(f"  {'Detektor':<34}{'Se':>8}{'+P':>8}{'Fehlalarme/h':>14}")
    rows = [(f"Regelwerk ({v})" + (" *" if v == primary else ""), r["test"]) for v, r in rule_results.items()]
    rows.append(("Modell, Schwelle 0,5", m_model))
    for label, m in rows:
        print(f"  {label:<34}{100 * m['sensitivity']:>7.1f}%{100 * m['ppv']:>7.1f}%{m['false_alarms_per_hour']:>14.1f}")
    print("  * auf DS1 ausgewählte Variante")
    print(f"\nFertig in {time.time() - t0:.0f} s. Bericht: {args.out_dir / 'report.md'}")


if __name__ == "__main__":
    main()
