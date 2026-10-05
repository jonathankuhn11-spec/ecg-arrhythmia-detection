"""Deep-Learning-Vergleich: 1D-CNN auf rohen Schlagfenstern gegen Regelwerk und Random Forest.

  python run_cnn.py                  (nach einem Lauf von run_pipeline.py; nutzt dieselben Daten)
Ergebnis: results/cnn.json, results/figures/cnn_training.png
"""
import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import average_precision_score

from ecg import cnn, config, data, detectors, evaluate, features


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=Path("data/mitdb"))
    ap.add_argument("--out-dir", type=Path, default=Path("results"))
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--seeds", type=int, default=3, help="Wiederholungen mit verschiedenen Startwerten")
    args = ap.parse_args(argv)
    missing = data.missing_records(args.data_dir, config.DS1 + config.DS2)
    if missing:
        sys.exit(f"Es fehlen Daten in {args.data_dir}. Erst: python run_pipeline.py --download")

    t0 = time.time()
    print("Lade Schlagfenster ...")
    Xtr, Rtr, ytr, rtr = cnn.beat_windows(args.data_dir, config.DS1)
    Xte, Rte, yte, rte = cnn.beat_windows(args.data_dir, config.DS2)
    hours_test = 22 * 30 / 60
    print(f"  {len(Xtr)} Schläge in DS1, {len(Xte)} in DS2, Fenster {Xtr.shape[-1]} Abtastwerte, {Xtr.shape[1]} Kanäle")
    val_records = config.DS1[-5:]                                  # patientenweise Validierung innerhalb DS1

    # Random Forest auf denselben Schlägen, für schwellenfreie Vergleiche (Average Precision, Iso-Sensitivität)
    print("Random Forest als Referenz ...")
    train_df, _ = features.build_dataset(args.data_dir, config.DS1)
    test_df, _ = features.build_dataset(args.data_dir, config.DS2)
    rf_scores = detectors.model_scores(detectors.train_model(train_df), test_df)
    rf_y = detectors.is_veb(test_df)
    assert len(rf_y) == len(yte) and (rf_y == yte).all(), "Schlagmengen von CNN und Modell stimmen nicht überein"
    rf_ap = float(average_precision_score(yte, rf_scores))
    rf_m = evaluate.binary_metrics(yte, rf_scores >= 0.5, hours_test)

    ref = json.loads((args.out_dir / "metrics.json").read_text(encoding="utf-8")) \
        if (args.out_dir / "metrics.json").exists() else None
    runs, all_scores = [], []
    val = np.isin(rtr, val_records)
    for seed in range(args.seeds):
        print(f"Training, Startwert {seed} ...")
        trained = cnn.train(Xtr, Rtr, ytr, rtr, epochs=args.epochs, seed=seed, val_records=val_records, log=print)
        # Entscheidungsschwelle auf den Validierungsrecords wählen (bester F1), nie auf DS2
        val_scores = cnn.predict_proba(trained.params, Xtr[val], Rtr[val])
        thr_val = evaluate.best_f1_threshold(val_scores, ytr[val])
        scores = cnn.predict_proba(trained.params, Xte, Rte)
        m = evaluate.binary_metrics(yte, scores >= thr_val, hours_test)
        thr_iso = evaluate.threshold_for_sensitivity(scores, yte, rf_m["tp"])
        run = {"seed": seed, "best_epoch": trained.best_epoch, "seconds": round(trained.seconds),
               "history": trained.history, "threshold_val": thr_val, "test": m,
               "test_at_0_5": evaluate.binary_metrics(yte, scores >= 0.5, hours_test),
               "average_precision": float(average_precision_score(yte, scores)),
               "iso_rf": evaluate.binary_metrics(yte, scores >= thr_iso, hours_test), "iso_rf_threshold": thr_iso}
        all_scores.append(scores.astype(np.float32))
        if ref:
            tp_rule = ref["rules"][ref["primary_rule"]]["test"]["tp"]
            thr = evaluate.threshold_for_sensitivity(scores, yte, tp_rule)
            run["iso_rule"] = evaluate.binary_metrics(yte, scores >= thr, hours_test)
            run["iso_threshold"] = thr
        runs.append(run)
        print(f"  DS2 (Schwelle {thr_val:.2f} aus Validierung): Se {100 * m['sensitivity']:.1f} %  "
              f"+P {100 * m['ppv']:.1f} %  Fehlalarme/h {m['false_alarms_per_hour']:.1f}  "
              f"(beste Epoche {trained.best_epoch}, {trained.seconds:.0f} s)")

    se = [r["test"]["sensitivity"] for r in runs]
    ppv = [r["test"]["ppv"] for r in runs]
    fa = [r["test"]["false_alarms_per_hour"] for r in runs]
    summary = {"model": "1D-CNN (JAX)", "n_params": int(sum(int(np.prod(np.asarray(p).shape)) for leaf in
                                                           cnn.init_params(cnn.jax.random.PRNGKey(0)).values()
                                                           for p in leaf)),
               "epochs": args.epochs, "seeds": args.seeds, "val_records": val_records,
               "sensitivity_mean": float(np.mean(se)), "sensitivity_min": float(np.min(se)), "sensitivity_max": float(np.max(se)),
               "ppv_mean": float(np.mean(ppv)), "ppv_min": float(np.min(ppv)), "ppv_max": float(np.max(ppv)),
               "false_alarms_per_hour_mean": float(np.mean(fa)), "false_alarms_per_hour_min": float(np.min(fa)),
               "false_alarms_per_hour_max": float(np.max(fa)),
               "average_precision_mean": float(np.mean([r["average_precision"] for r in runs])),
               "iso_rf_ppv_mean": float(np.mean([r["iso_rf"]["ppv"] for r in runs])),
               "random_forest": {"test": rf_m, "average_precision": rf_ap}, "runs": runs}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "cnn.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    np.savez_compressed(args.out_dir / "cnn_scores.npz", y=yte, cnn=np.stack(all_scores), rf=rf_scores.astype(np.float32))

    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    for r in runs:
        ep = [h[0] for h in r["history"]]
        ax.plot(ep, [h[1] for h in r["history"]], color="#1F7A8C", alpha=0.6, linewidth=1.2)
        ax.plot(ep, [h[2] for h in r["history"]], color="#C0392B", alpha=0.6, linewidth=1.2)
    ax.plot([], [], color="#1F7A8C", label="Training (DS1 ohne Validierungsrecords)")
    ax.plot([], [], color="#C0392B", label="Validierung (5 Records aus DS1)")
    ax.set_xlabel("Epoche"); ax.set_ylabel("gewichteter Verlust")
    ax.set_title("1D-CNN: Verlauf über drei Startwerte", fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=9); ax.spines[["top", "right"]].set_visible(False); ax.grid(alpha=0.25)
    fig.tight_layout(); (args.out_dir / "figures").mkdir(exist_ok=True)
    fig.savefig(args.out_dir / "figures" / "cnn_training.png", dpi=150); plt.close(fig)

    print(f"\n1D-CNN auf DS2, {args.seeds} Startwerte: Se {100 * summary['sensitivity_mean']:.1f} % "
          f"({100 * summary['sensitivity_min']:.1f}–{100 * summary['sensitivity_max']:.1f}), "
          f"+P {100 * summary['ppv_mean']:.1f} %, Fehlalarme/h {summary['false_alarms_per_hour_mean']:.1f}")
    print(f"  Average Precision CNN {summary['average_precision_mean']:.3f}, bei Sensitivität des Random Forest: "
          f"+P {100 * summary['iso_rf_ppv_mean']:.1f} %")
    print(f"Random Forest: Se {100 * rf_m['sensitivity']:.1f} %, +P {100 * rf_m['ppv']:.1f} %, "
          f"Fehlalarme/h {rf_m['false_alarms_per_hour']:.1f}, Average Precision {rf_ap:.3f}")
    print(f"Fertig in {time.time() - t0:.0f} s.")
    return summary


if __name__ == "__main__":
    main()
