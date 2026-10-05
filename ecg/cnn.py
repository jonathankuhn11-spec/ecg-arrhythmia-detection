"""Deep-Learning-Vergleich: 1D-CNN auf den rohen Schlagfenstern (JAX, nur CPU nötig).

Das Netz sieht dasselbe wie der Random Forest, nur ungefiltert durch Merkmale: das bandpassgefilterte
Schlagfenster von 250 ms vor bis 400 ms nach der R-Zacke (234 Abtastwerte bei 360 Hz) als Kanal 1 und, als
Kanal 2, die Abweichung dieses Schlags vom Median der 30 vorangegangenen Schläge desselben Patienten
(Vorlage). Das entspricht dem, was die relativen Merkmale des Modells leisten: "anders als sonst bei diesem
Patienten" statt "anders als der Durchschnitt aller Patienten". Dazu die beiden Rhythmusmerkmale
(Vorzeitigkeit, Pause danach). Training nur auf DS1, Auswertung nur auf DS2, wie beim Modell.

Architektur (klein, damit sie auf einer CPU in wenigen Minuten trainiert):
  Conv1d(2→16, k=7) ReLU MaxPool(2) → Conv1d(16→32, k=5) ReLU MaxPool(2) → Conv1d(32→32, k=3) ReLU
  → globales Mittel über die Zeit → Verkettung mit den 2 Rhythmusmerkmalen → Dense(34→32) ReLU → Dense(32→1)
Klassengewichtung in der Verlustfunktion, Adam, früher Abbruch per Validierungsanteil aus DS1 (patientenweise).
"""
import time
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from . import config
from .data import load_record
from .features import bandpass, rolling_median_prev

WIN_PRE, WIN_POST = int(config.WIN_PRE_S * config.FS), int(config.WIN_POST_S * config.FS)
WIN = WIN_PRE + WIN_POST          # 234 Abtastwerte
RHYTHM_FEATURES = ["rr_ratio_pre", "rr_ratio_post"]
TEMPLATE_BEATS = 30               # Vorlage: Median der letzten 30 Schläge (kausal, wie die Modellmerkmale)
CHANNELS = 2


def with_template(windows: np.ndarray, n: int = TEMPLATE_BEATS) -> np.ndarray:
    """Kanal 1: der Schlag; Kanal 2: Schlag minus Median der n vorangegangenen Schläge."""
    out = np.empty((len(windows), CHANNELS, windows.shape[1]), dtype=np.float32)
    out[:, 0] = windows
    for i in range(len(windows)):
        prev = windows[max(0, i - n):i]
        template = np.median(prev, axis=0) if len(prev) else windows[i]
        out[i, 1] = windows[i] - template
    return out


def beat_windows(data_dir, records, progress=None):
    """Rohe Schlagfenster, Rhythmusmerkmale, Ziel (VEB) und Record je Schlag."""
    X, R, y, rec_ids = [], [], [], []
    for name in records:
        if progress:
            progress(name)
        rec = load_record(data_dir, name)
        x = bandpass(rec.signal, rec.fs)
        is_beat = np.array([s in config.AAMI for s in rec.ann_symbols])
        samples, symbols = rec.ann_samples[is_beat], np.array(rec.ann_symbols)[is_beat]
        aami = np.array([config.AAMI[s] for s in symbols])
        rr = np.diff(samples) / rec.fs
        rr_pre, rr_post = np.r_[rr[0], rr], np.r_[rr, rr[-1]]
        rr_local = rolling_median_prev(rr_pre, config.RR_LOCAL_BEATS)
        ok = (samples - WIN_PRE >= 0) & (samples + WIN_POST < len(x)) & (aami != "Q")
        windows = np.stack([x[s - WIN_PRE:s + WIN_POST] for s in samples[ok]])
        windows -= np.median(windows, axis=1, keepdims=True)
        scale = np.median(np.abs(windows).max(axis=1))            # Verstärkung des Records herausrechnen
        X.append(with_template(windows / scale))
        R.append(np.c_[rr_pre / np.maximum(rr_local, 1e-3), rr_post / np.maximum(rr_pre, 1e-3)][ok])
        y.append(aami[ok] == "V")
        rec_ids.append(np.full(ok.sum(), name))
    return (np.concatenate(X).astype(np.float32), np.vstack(R).astype(np.float32),
            np.concatenate(y), np.concatenate(rec_ids))


def init_params(key, n_rhythm: int = 2):
    k = jax.random.split(key, 5)
    glorot = lambda kk, shape, fan_in, fan_out: jax.random.normal(kk, shape) * jnp.sqrt(2.0 / (fan_in + fan_out))
    return {
        "c1": (glorot(k[0], (16, CHANNELS, 7), CHANNELS * 7, 16 * 7), jnp.zeros(16)),
        "c2": (glorot(k[1], (32, 16, 5), 16 * 5, 32 * 5), jnp.zeros(32)),
        "c3": (glorot(k[2], (32, 32, 3), 32 * 3, 32 * 3), jnp.zeros(32)),
        "d1": (glorot(k[3], (32 + n_rhythm, 32), 32 + n_rhythm, 32), jnp.zeros(32)),
        "d2": (glorot(k[4], (32, 1), 32, 1), jnp.zeros(1)),
    }


def _conv(x, w, b):
    y = jax.lax.conv_general_dilated(x, w, (1,), "VALID", dimension_numbers=("NCH", "OIH", "NCH"))
    return jax.nn.relu(y + b[None, :, None])


def _pool(x):
    n = x.shape[-1] // 2 * 2
    return x[..., :n].reshape(x.shape[0], x.shape[1], -1, 2).max(axis=-1)


def forward(params, windows, rhythm):
    """Logit je Schlag. windows: (N, 2, 234), rhythm: (N, 2)."""
    h = windows
    h = _pool(_conv(h, *params["c1"]))
    h = _pool(_conv(h, *params["c2"]))
    h = _conv(h, *params["c3"])
    h = h.mean(axis=-1)                                           # globales Mittel über die Zeit
    h = jnp.concatenate([h, rhythm], axis=1)
    h = jax.nn.relu(h @ params["d1"][0] + params["d1"][1])
    return (h @ params["d2"][0] + params["d2"][1])[:, 0]


def loss_fn(params, windows, rhythm, y, pos_weight):
    logit = forward(params, windows, rhythm)
    w = jnp.where(y, pos_weight, 1.0)
    return jnp.mean(w * jnp.maximum(logit, 0) - w * logit * y + w * jnp.log1p(jnp.exp(-jnp.abs(logit))))


@dataclass
class Trained:
    params: dict
    history: list          # (epoch, train_loss, val_loss)
    best_epoch: int
    seconds: float


def _adam_update(params, grads, state, lr, b1=0.9, b2=0.999, eps=1e-8):
    m, v, t = state
    t = t + 1
    m = jax.tree_util.tree_map(lambda m_, g: b1 * m_ + (1 - b1) * g, m, grads)
    v = jax.tree_util.tree_map(lambda v_, g: b2 * v_ + (1 - b2) * g * g, v, grads)
    mhat = jax.tree_util.tree_map(lambda m_: m_ / (1 - b1 ** t), m)
    vhat = jax.tree_util.tree_map(lambda v_: v_ / (1 - b2 ** t), v)
    new = jax.tree_util.tree_map(lambda p, mh, vh: p - lr * mh / (jnp.sqrt(vh) + eps), params, mhat, vhat)
    return new, (m, v, t)


def train(X, R, y, records, epochs: int = 12, batch: int = 256, lr: float = 1e-3, seed: int = config.SEED,
          val_records=None, log=None) -> Trained:
    """Trainiert auf allen Records außer val_records, wählt die Epoche mit dem kleinsten Validierungsverlust."""
    rng = np.random.default_rng(seed)
    val_records = list(val_records or [])
    val = np.isin(records, val_records)
    Xtr, Rtr, ytr = X[~val], R[~val], y[~val]
    Xva, Rva, yva = X[val], R[val], y[val]
    pos_weight = float((~ytr).sum() / max(ytr.sum(), 1))

    params = init_params(jax.random.PRNGKey(seed))
    state = (jax.tree_util.tree_map(jnp.zeros_like, params), jax.tree_util.tree_map(jnp.zeros_like, params), 0)
    step = jax.jit(lambda p, s, xb, rb, yb: _adam_update(p, jax.grad(loss_fn)(p, xb, rb, yb, pos_weight), s, lr))
    evaluate = jax.jit(lambda p, xb, rb, yb: loss_fn(p, xb, rb, yb, pos_weight))

    t0, history, best = time.time(), [], (np.inf, None, 0)
    for epoch in range(1, epochs + 1):
        order = rng.permutation(len(Xtr))
        losses = []
        for i in range(0, len(order), batch):
            idx = order[i:i + batch]
            if len(idx) < batch:                                   # feste Batchgröße hält den JIT-Graphen stabil
                idx = np.r_[idx, order[:batch - len(idx)]]
            params, state = step(params, state, Xtr[idx], Rtr[idx], ytr[idx])
            losses.append(float(evaluate(params, Xtr[idx], Rtr[idx], ytr[idx])))
        val_loss = float(np.mean([float(evaluate(params, Xva[i:i + 2048], Rva[i:i + 2048], yva[i:i + 2048]))
                                  for i in range(0, len(Xva), 2048)])) if len(Xva) else float(np.mean(losses))
        history.append((epoch, float(np.mean(losses)), val_loss))
        if log:
            log(f"  Epoche {epoch:2d}: Training {np.mean(losses):.4f}  Validierung {val_loss:.4f}")
        if val_loss < best[0]:
            best = (val_loss, jax.tree_util.tree_map(lambda a: a.copy(), params), epoch)
    return Trained(best[1], history, best[2], time.time() - t0)


def predict_proba(params, X, R, batch: int = 4096) -> np.ndarray:
    f = jax.jit(forward)
    out = [np.asarray(jax.nn.sigmoid(f(params, X[i:i + batch], R[i:i + batch]))) for i in range(0, len(X), batch)]
    return np.concatenate(out)
