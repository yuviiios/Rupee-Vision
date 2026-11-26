"""Fit from raw rates, measure generalization, and construct a candidate release."""
import argparse
import json
from pathlib import Path

import joblib
import keras
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from benchmark_sessions import REVISION, load_prices
from rupeevision.ml.session_models import STEPS, WINDOW, levels_from_returns, return_windows, ridge_features

FOLDS = pd.date_range("2024-06-01", "2026-06-01", freq="6MS")
HEADLINES = [0, 6, 29]
ALPHAS = {"Ridge100": 100, "Ridge10000": 10000}


def new_network():
    keras.backend.clear_session()
    keras.utils.set_random_seed(42)
    model = keras.Sequential([
        keras.Input((WINDOW, 1)),
        keras.layers.LSTM(8, kernel_regularizer=keras.regularizers.l2(0.0005)),
        keras.layers.Dense(len(STEPS), kernel_initializer="zeros", bias_initializer="zeros"),
    ])
    model.compile(optimizer=keras.optimizers.Adam(0.002, clipnorm=1), loss="mae")
    return model


def fit_models(x, y, epochs=None):
    models = {name: make_pipeline(StandardScaler(), Ridge(alpha=alpha)).fit(ridge_features(x), y)
              for name, alpha in ALPHAS.items()}
    net = new_network()
    if epochs is None:
        # Chronological inner validation; embargo overlapping 30-session labels.
        boundary = len(x) - 250
        log = net.fit(x[:boundary-30, :, None], y[:boundary-30],
                      validation_data=(x[boundary:, :, None], y[boundary:]),
                      epochs=16, batch_size=128, verbose=0, shuffle=False,
                      callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=3)])
        epochs = 1 + int(np.argmin(log.history["val_loss"]))
        net = new_network()
    net.fit(x[:, :, None], y, epochs=epochs, batch_size=128, verbose=0, shuffle=False)
    models["LSTM"] = net
    return models, epochs


def predict_all(models, x, prices):
    results = {"Last rate": np.repeat(prices[:, None], len(STEPS), axis=1)}
    for key, model in models.items():
        pred = model(x[:, :, None], training=False).numpy() if key == "LSTM" else model.predict(ridge_features(x))
        results[key] = levels_from_returns(prices, pred)
    return results


def train(prices, out):
    diagnostics, selected, epochs_used = [], {}, {}
    for code in prices:
        levels = prices[code].to_numpy()
        origins = np.arange(WINDOW, len(levels)-len(STEPS))
        x = return_windows(levels)[:len(origins)]
        actual = levels[origins[:, None] + STEPS]
        target = 100 * np.log(actual / levels[origins, None]) / np.sqrt(STEPS)
        history_epochs = []
        for i, start in enumerate(FOLDS):
            cut = prices.index.searchsorted(start)
            end = prices.index.searchsorted(FOLDS[i+1]) if i+1 < len(FOLDS) else len(prices)
            train_mask = origins + STEPS[-1] < cut
            eval_mask = (origins >= cut) & (origins + STEPS[-1] < end)
            models, epochs = fit_models(x[train_mask], target[train_mask])
            history_epochs.append(epochs)
            for split, mask in (("train", train_mask), ("evaluation", eval_mask)):
                # Recent fitted examples are a more comparable in-sample diagnostic.
                ids = np.flatnonzero(mask)
                if split == "train":
                    ids = ids[-250:]
                predictions = predict_all(models, x[ids], levels[origins[ids]])
                base_mae = np.abs(predictions["Last rate"] - actual[ids]).mean(axis=0)
                for name, pred in predictions.items():
                    error = pred - actual[ids]
                    for h in HEADLINES:
                        mae = np.abs(error[:, h]).mean()
                        diagnostics.append(dict(currency=code, start=start.date(), split=split, model=name,
                            horizon=h+1, count=len(ids), mae=mae, rmse=np.sqrt(np.mean(error[:,h]**2)),
                            gain_pct=100*(1-mae/base_mae[h]), epochs=epochs if name=="LSTM" else None))
            print(code, start.date(), "epochs", epochs, flush=True)

        results = pd.DataFrame(diagnostics).query("currency == @code and split == 'evaluation'")
        by_fold = results.groupby(["start","model"]).gain_pct.mean().unstack()
        validation = by_fold.iloc[:-1]
        # Choose on prior folds, then require confirmation on the previously examined audit period.
        eligible = validation.columns[(validation.mean() >= 2) & ((validation > 0).sum() >= 3)]
        choice = validation[eligible].mean().idxmax() if len(eligible) else "Last rate"
        audit = results[(results.start == FOLDS[-1].date()) & (results.model == choice)]
        if audit.gain_pct.mean() <= 0 or audit.gain_pct.min() < -2:
            choice = "Last rate"
        selected[code] = choice
        epochs_used[code] = int(np.median(history_epochs[:-1]))
        models, _ = fit_models(x, target, epochs_used[code])
        models["LSTM"].save(out / f"{code}.keras")
        for name in ALPHAS:
            joblib.dump(models[name], out / f"{code}-{name}.joblib")
    pd.DataFrame(diagnostics).to_csv(out / "diagnostics.csv", index=False)
    (out / "release.json").write_text(json.dumps(dict(selected=selected, epochs=epochs_used,
        trained_through=str(prices.index[-1].date()), snapshot_revision=REVISION,
        sessions=len(prices), horizons=STEPS.tolist()), indent=2))
    print(selected)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True, help="New release directory")
    out = parser.parse_args().output
    out.mkdir(parents=True, exist_ok=False)
    with threadpool_limits(limits=1):
        train(load_prices(), out)
