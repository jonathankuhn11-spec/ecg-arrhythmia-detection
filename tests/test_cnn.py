"""1D-CNN: Fenster, Vorwärtsrechnung, Gradienten, Training auf synthetischen Schlägen (ohne MIT-BIH)."""
import jax
import jax.numpy as jnp
import numpy as np

from conftest import FS, synthetic_record
from ecg import cnn


def _windows_from_record(rec):
    """beat_windows arbeitet auf Dateien; hier dieselbe Logik direkt auf einem Record."""
    from ecg.features import bandpass
    x = bandpass(rec.signal, rec.fs)
    samples = rec.ann_samples
    ok = (samples - cnn.WIN_PRE >= 0) & (samples + cnn.WIN_POST < len(x))
    W = np.stack([x[s - cnn.WIN_PRE:s + cnn.WIN_POST] for s in samples[ok]]).astype(np.float32)
    W -= np.median(W, axis=1, keepdims=True)
    W /= np.median(np.abs(W).max(axis=1))
    W = cnn.with_template(W)
    rr = np.diff(samples) / rec.fs
    rr_pre, rr_post = np.r_[rr[0], rr], np.r_[rr, rr[-1]]
    R = np.c_[rr_pre / np.median(rr_pre), rr_post / rr_pre][ok].astype(np.float32)
    y = (np.array(rec.ann_symbols)[ok] == "V")
    return W, R, y


def test_window_length_matches_feature_window():
    assert cnn.WIN == int(0.25 * FS) + int(0.40 * FS) == 234


def test_template_channel_is_deviation_from_previous_beats():
    w = np.zeros((40, 10), np.float32)
    w[35] = 1.0                                                   # ein abweichender Schlag
    out = cnn.with_template(w, n=30)
    assert out.shape == (40, 2, 10)
    assert np.allclose(out[35, 1], 1.0) and np.allclose(out[36, 1], 0.0)   # Median bleibt bei 0
    assert np.allclose(out[0, 1], 0.0)                            # erster Schlag: keine Vorlage, Abweichung 0


def test_forward_shapes_and_parameter_count():
    params = cnn.init_params(jax.random.PRNGKey(0))
    n = sum(int(np.prod(np.asarray(p).shape)) for leaf in params.values() for p in leaf)
    assert 5_000 < n < 20_000                                   # bewusst kleines Netz
    logit = cnn.forward(params, jnp.zeros((5, cnn.CHANNELS, cnn.WIN)), jnp.ones((5, 2)))
    assert logit.shape == (5,)


def test_loss_gradient_is_finite_and_weights_positives():
    params = cnn.init_params(jax.random.PRNGKey(1))
    W = jnp.asarray(np.random.default_rng(0).normal(size=(8, cnn.CHANNELS, cnn.WIN)).astype(np.float32))
    R = jnp.ones((8, 2)); y = jnp.array([True, False] * 4)
    g = jax.grad(cnn.loss_fn)(params, W, R, y, 5.0)
    assert all(bool(jnp.isfinite(a).all()) for leaf in g.values() for a in leaf)
    assert float(cnn.loss_fn(params, W, R, y, 5.0)) > float(cnn.loss_fn(params, W, R, y, 1.0)) * 0.9


def test_training_separates_synthetic_veb():
    recs = [synthetic_record(n_beats=200, veb_every=k) for k in (4, 5, 6)]
    parts = [_windows_from_record(r) for r in recs]
    W = np.vstack([p[0] for p in parts]); R = np.vstack([p[1] for p in parts]); y = np.concatenate([p[2] for p in parts])
    ids = np.concatenate([np.full(len(p[2]), i) for i, p in enumerate(parts)])
    trained = cnn.train(W, R, y, ids, epochs=6, batch=64, seed=0, val_records=[2])
    assert trained.best_epoch >= 1 and len(trained.history) == 6
    proba = cnn.predict_proba(trained.params, W[ids == 2], R[ids == 2])
    assert ((proba >= 0.5) == y[ids == 2]).mean() > 0.95
