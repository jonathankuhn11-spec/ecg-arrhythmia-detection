"""Ergebnisse als Abbildungen und Markdown-Bericht ausgeben."""
import matplotlib

matplotlib.use("Agg")   # ohne Fenster rendern, läuft so auch auf Servern
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve

from . import config
from .data import load_record
from .features import bandpass

C_SIGNAL, C_VEB, C_RULE, C_MODEL, C_GREY = "#1F3A5F", "#C0392B", "#E08E0B", "#1F7A8C", "#8A8A8A"

FEATURE_TEXT = {
    "rr_ratio_pre": "Vorzeitigkeit: RR-Intervall vor dem Schlag relativ zum lokalen Rhythmus",
    "rr_ratio_post": "Pause danach: RR-Intervall nach dem Schlag relativ zum Intervall davor",
    "qrs_width_rel": "QRS-Breite relativ zu den letzten 30 Schlägen",
    "qrs_width_ms": "QRS-Breite absolut (Halbwertsbreite in ms)",
    "rr_pre": "RR-Intervall vor dem Schlag in s",
    "rr_post": "RR-Intervall nach dem Schlag in s",
    "rr_local": "lokaler Rhythmus: Median der letzten 10 RR-Intervalle",
    "amp_rel": "Amplitude relativ zu den letzten 30 Schlägen",
}


def pct(x: float) -> str:
    return f"{100 * x:.1f}".replace(".", ",") + " %"


def num(x: float, digits: int = 1) -> str:
    return f"{x:.{digits}f}".replace(".", ",")


def _style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.25, linewidth=0.6)


def plot_strip(data_dir, test, rule_pred, model_pred, out_path, seconds: float = 8.0):
    """EKG-Ausschnitt aus der Testmenge mit Referenz und Detektor-Entscheidungen."""
    veb_per_record = test[test["aami"] == "V"].groupby("record").size()
    name = int(veb_per_record.idxmax())
    rec = load_record(data_dir, name)
    x = bandpass(rec.signal, rec.fs)
    part = test["record"].to_numpy() == name
    beats = test[part]
    rule_hit, model_hit = rule_pred[part], model_pred[part]

    # erstes Fenster, in dem mindestens zwei VEB liegen
    veb_pos = beats.loc[beats["aami"] == "V", "sample"].to_numpy()
    start = veb_pos[0]
    for a, b in zip(veb_pos[:-1], veb_pos[1:]):
        if b - a < (seconds - 2) * rec.fs:
            start = a
            break
    lo = max(0, int(start - rec.fs))
    hi = min(len(x), lo + int(seconds * rec.fs))
    t = (np.arange(lo, hi) - lo) / rec.fs

    fig, ax = plt.subplots(figsize=(11, 3.4))
    ax.plot(t, x[lo:hi], color=C_SIGNAL, linewidth=1.0)
    top = x[lo:hi].max()
    span = top - x[lo:hi].min()
    inside = (beats["sample"].to_numpy() >= lo) & (beats["sample"].to_numpy() < hi)
    for (_, beat), r_hit, m_hit in zip(beats[inside].iterrows(), rule_hit[inside], model_hit[inside]):
        bt = (beat["sample"] - lo) / rec.fs
        veb = beat["aami"] == "V"
        ax.text(bt, top + 0.10 * span, beat["aami"], ha="center", va="bottom", fontsize=10,
                fontweight="bold" if veb else "normal", color=C_VEB if veb else C_GREY)
        if r_hit:
            ax.plot(bt, top + 0.34 * span, marker="s", color=C_RULE, markersize=6)
        if m_hit:
            ax.plot(bt, top + 0.50 * span, marker="v", color=C_MODEL, markersize=7)
    ax.plot([], [], marker="s", color=C_RULE, linestyle="", label="Regelwerk meldet VEB")
    ax.plot([], [], marker="v", color=C_MODEL, linestyle="", label="Modell meldet VEB")
    ax.set_ylim(x[lo:hi].min() - 0.08 * span, top + 0.68 * span)
    ax.set_xlim(0, seconds)
    ax.set_xlabel("Zeit in s")
    ax.set_ylabel(f"{config.LEAD} in mV")
    ax.set_title(f"Record {name} (Testmenge): Referenzklasse und Meldungen je Schlag", fontsize=11, loc="left")
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), frameon=False, ncol=2, fontsize=9)
    _style(ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return name


def plot_precision_recall(y_true, scores, metrics: dict, out_path):
    """Precision-Recall-Kurve des Modells mit den Betriebspunkten aller Detektoren."""
    precision, recall, _ = precision_recall_curve(y_true, scores)
    primary = metrics["primary_rule"]
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    ax.plot(100 * recall, 100 * precision, color=C_MODEL, linewidth=1.8, label="Modell (alle Schwellen)")
    m = metrics["model"]["test"]
    ax.plot(100 * m["sensitivity"], 100 * m["ppv"], marker="v", color=C_MODEL, markersize=10,
            linestyle="", markeredgecolor="white", label="Modell, Schwelle 0,5")
    for variant, result in metrics["rules"].items():
        m = result["test"]
        filled = variant == primary
        ax.plot(100 * m["sensitivity"], 100 * m["ppv"], marker="s", markersize=8, linestyle="",
                color=C_RULE if filled else "white", markeredgecolor=C_RULE, markeredgewidth=1.6,
                label=f"Regelwerk ({variant})" + (", auf DS1 gewählt" if filled else ""))
    ax.set_xlim(60, 100.5)
    ax.set_ylim(80, 100.5)
    ax.set_yticks(range(80, 101, 5))
    ax.set_xlabel("Sensitivität in %")
    ax.set_ylabel("positive Prädiktivität in %")
    ax.set_title("VEB-Erkennung auf der Testmenge (DS2)", fontsize=11, loc="left")
    ax.legend(loc="lower left", frameon=False, fontsize=9)
    _style(ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_false_alarms(per_record, out_path):
    """Fehlalarme je Record der Testmenge: Regelwerk gegen Modell."""
    idx = np.arange(len(per_record))
    fig, ax = plt.subplots(figsize=(11, 3.6))
    ax.bar(idx - 0.2, per_record["fp_rule"], width=0.4, color=C_RULE, label="Regelwerk")
    ax.bar(idx + 0.2, per_record["fp_model"], width=0.4, color=C_MODEL, label="Modell")
    ax.set_xticks(idx)
    ax.set_xticklabels(per_record["record"].astype(str))
    ax.set_xlabel("Record (Testmenge DS2, je 30 Minuten)")
    ax.set_ylabel("falsch-positive VEB-Meldungen")
    ax.set_title("Wo entstehen die Fehlalarme?", fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=9)
    _style(ax)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _metric_row(label: str, m: dict) -> str:
    return (f"| {label} | {pct(m['sensitivity'])} | {pct(m['ppv'])} | {m['fp']} | "
            f"{num(m['false_alarms_per_hour'])} | {m['fn']} |")


def _rule_text(params: dict) -> str:
    unit = " ms" if params["feature"] == "qrs_width_ms" else ""
    digits = 0 if unit else 2
    return (f"vorzeitig bei `rr_ratio_pre < {num(params['premature'], 2)}`, "
            f"verbreitert bei `{params['feature']} > {num(params['wide'], digits)}{unit}`, "
            f"stark verbreitert bei `{params['feature']} > {num(params['very_wide'], digits)}{unit}`")


def write_report(metrics: dict, out_path):
    """Schreibt den Markdown-Bericht aus den berechneten Kennzahlen."""
    d, model, errors = metrics["data"], metrics["model"], metrics["errors_test"]
    primary = metrics["primary_rule"]
    main_rule = metrics["rules"][primary]
    lines = [
        "# Ergebnisbericht",
        "",
        "Automatisch erzeugt von `run_pipeline.py`. Regelwerk und Modell wurden ausschließlich auf DS1",
        "eingestellt. Alle Kennzahlen stammen von der Testmenge DS2, sofern nicht anders angegeben.",
        "",
        "## Daten",
        "",
        "| Menge | Records | Schläge | davon VEB | Dauer in h |",
        "| --- | --- | --- | --- | --- |",
        f"| DS1 (Training) | {d['train_records']} | {d['train_beats']} | {d['train_veb']} | {num(d['train_hours'], 2)} |",
        f"| DS2 (Test) | {d['test_records']} | {d['test_beats']} | {d['test_veb']} | {num(d['test_hours'], 2)} |",
        "",
        "## VEB-Erkennung auf DS2",
        "",
        "| Detektor | Sensitivität | pos. Prädiktivität | Fehlalarme | Fehlalarme pro h | verpasste VEB |",
        "| --- | --- | --- | --- | --- | --- |",
        _metric_row(f"Regelwerk ({primary})", main_rule["test"]),
        _metric_row("Modell, Schwelle 0,5", model["test"]),
        "",
        f"Das Modell meldet {pct(model['false_alarm_reduction_vs_primary_rule'])} weniger Fehlalarme als das Regelwerk",
        f"und erkennt zugleich mehr VEB ({pct(model['test']['sensitivity'])} gegenüber {pct(main_rule['test']['sensitivity'])}).",
        "",
        "## Regelvarianten und Vergleich bei gleicher Sensitivität",
        "",
        "Beide Varianten haben dieselbe Struktur: (vorzeitig UND verbreitert) ODER stark verbreitert.",
        "Sie messen die QRS-Breite relativ zu den letzten 30 Schlägen oder absolut in Millisekunden.",
        f"Die Schwellen stammen aus einer Gittersuche auf DS1 (Zielgröße F1). Als Hauptvariante gilt `{primary}`,",
        "weil sie auf DS1 den höheren F1-Wert erreicht.",
        "",
        "| Variante | F1 auf DS1 | DS1: Se / +P | DS2: Se / +P | Fehlalarme DS2 | Modell bei gleicher Se: Fehlalarme | Differenz |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for variant, r in metrics["rules"].items():
        lines.append(
            f"| {variant} | {num(r['params']['f1_train'], 3)} | "
            f"{pct(r['train']['sensitivity'])} / {pct(r['train']['ppv'])} | "
            f"{pct(r['test']['sensitivity'])} / {pct(r['test']['ppv'])} | {r['test']['fp']} | "
            f"{r['model_iso']['fp']} (Schwelle {num(r['iso_threshold'], 3)}) | "
            f"{pct(-r['false_alarm_reduction_iso']).replace('-', '−')} |")
    lines += [
        "",
        "Die Modellschwelle für den Vergleich bei gleicher Sensitivität wurde auf DS2 abgelesen.",
        "Das ist eine Beschreibung der Trennschärfe, kein vorab festgelegter Betriebspunkt.",
        "",
        "Kalibrierte Schwellen:",
        "",
    ]
    for variant, r in metrics["rules"].items():
        lines.append(f"- `{variant}`: {_rule_text(r['params'])}")
    lines += [
        "",
        f"## Fehlalarme nach wahrer Klasse (Regelwerk `{primary}` und Modell)",
        "",
        "| Detektor | N (normal) | S (supraventrikulär) | F (Fusion) |",
        "| --- | --- | --- | --- |",
        f"| Regelwerk | {errors['fp_by_class']['rule']['N']} | {errors['fp_by_class']['rule']['S']} | {errors['fp_by_class']['rule']['F']} |",
        f"| Modell | {errors['fp_by_class']['model']['N']} | {errors['fp_by_class']['model']['S']} | {errors['fp_by_class']['model']['F']} |",
        "",
        "## Fehler je Record",
        "",
        "| Record | VEB | Fehlalarme Regelwerk | Fehlalarme Modell | verpasst Regelwerk | verpasst Modell |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in errors["per_record"]:
        lines.append(f"| {row['record']} | {row['veb']} | {row['fp_rule']} | {row['fp_model']} | "
                     f"{row['fn_rule']} | {row['fn_model']} |")
    lines += [
        "",
        "## Wichtigste Merkmale des Modells",
        "",
        "| Merkmal | Bedeutung | Anteil |",
        "| --- | --- | --- |",
    ]
    for item in model["top_features"]:
        name = item["feature"]
        meaning = FEATURE_TEXT.get(name, "Schlagform: Mittelwert eines 25-ms-Abschnitts")
        lines.append(f"| `{name}` | {meaning} | {pct(item['importance'])} |")
    lines += ["", "Anteil: mittlere Verringerung der Unreinheit im Random Forest (Summe über alle Merkmale 100 %).", ""]

    if "qrs" in metrics:
        q = metrics["qrs"]
        lines += [
            "## R-Zacken-Detektion (NeuroKit2)",
            "",
            f"Über alle {q['records']} Records: Sensitivität {pct(q['sensitivity'])}, "
            f"positive Prädiktivität {pct(q['ppv'])} "
            f"({q['tp']} Treffer, {q['fp']} Fehldetektionen, {q['fn']} verpasste Schläge, Toleranz 150 ms).",
            "",
            "Die fünf schwächsten Records:",
            "",
            "| Record | Sensitivität | pos. Prädiktivität | Fehldetektionen | verpasst |",
            "| --- | --- | --- | --- | --- |",
        ]
        for row in q["worst"]:
            lines.append(f"| {row['record']} | {pct(row['sensitivity'])} | {pct(row['ppv'])} | {row['fp']} | {row['fn']} |")
        lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")
