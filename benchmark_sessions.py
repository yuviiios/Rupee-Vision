"""Monthly FX backtest. Run from the repository root with PYTHONPATH=src."""
import hashlib
import json
from importlib.metadata import version
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from rupeevision.core.constants import CURRENCY_CODES
from rupeevision.ml.data import load_rate_history
from rupeevision.ml.registry import ModelRegistry
from summarize_benchmark import main as summarize

ROOT = Path(__file__).resolve().parent
REVISION = "13f8f41326c3677c8ebd715fc458a8b7dfb21f0c"
DATA = ROOT / "data" / f"rbi_fbil_{REVISION[:7]}.csv"
OUT = ROOT / "reports" / "fx_session_backtest"
HORIZONS = (1, 7, 30)
START = pd.Timestamp("2023-01-01")
TEST = pd.Timestamp("2026-06-01")
# Screen alternative window lengths and cross-currency inputs on validation only.
CONFIGS = ("own_all_years", "own_five_years", "all_fx_five_years")


def load_prices():
    expected_sha256 = "8140c2eade5a41fbfc3a48a85b467e404a5762e670a0f53ca12e05d432618121"
    if hashlib.sha256(DATA.read_bytes()).hexdigest() != expected_sha256:
        raise ValueError("Dataset does not match the pinned archive snapshot")
    return load_rate_history(DATA).frame.rename(columns=CURRENCY_CODES)


def features(prices):
    log_prices = np.log(prices)
    returns = log_prices.diff()
    parts = [log_prices.diff(lag).add_suffix(f"_move{lag}") for lag in (1, 2, 5, 10, 20)]
    parts.extend(returns.rolling(n).std().add_suffix(f"_vol{n}") for n in (5, 20))
    return pd.concat(parts, axis=1)


def estimator(model):
    if model == "Ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=100.0))
    return HistGradientBoostingRegressor(max_iter=50, max_leaf_nodes=7, min_samples_leaf=40,
                                         learning_rate=0.05, early_stopping=False, random_state=42)


def inputs(frame, code, config):
    return frame if config.startswith("all_fx") else frame.filter(regex=f"^{code}_")


def training_rows(index, first, h, config):
    # Labels for row i mature at i+h. The first forecast is made AFTER its fixing.
    end = first - h + 1
    start = 20
    if config.endswith("five_years"):
        start = max(start, index.searchsorted(index[first] - pd.DateOffset(years=5)))
    return slice(start, end)


def lstm_forecasts(bundle, levels, origins):
    from tensorflow import function
    lookback = bundle.seq_length
    windows = np.lib.stride_tricks.sliding_window_view(levels, lookback)[origins-lookback+1]
    scaler = bundle.scaler
    # Equivalent to the deployed MinMaxScaler transform, batched across origins.
    seq = (windows * scaler.scale_[0] + scaler.min_[0]).astype("float32")[:, :, None]
    forward = function(lambda x: bundle.model(x, training=False), reduce_retracing=True)
    output = {}
    for step in range(1, max(HORIZONS) + 1):
        pred = forward(seq).numpy().reshape(-1)
        if step in HORIZONS:
            output[step] = scaler.inverse_transform(pred[:, None]).ravel()
        seq = np.concatenate([seq[:, 1:], pred[:, None, None]], axis=1)
    return output


def run_currency(prices, frame, name, code, registry):
    idx = prices.index
    levels = prices[code].to_numpy()
    origins = np.flatnonzero(idx >= START)
    frozen = lstm_forecasts(registry.get(name), levels, origins)
    months = idx[origins].to_period("M")
    pieces = []
    selected = {}
    final_models = {}
    for phase, cutoff in (("validation", TEST), ("holdout", idx[-1] + pd.Timedelta(days=1))):
        for month in months.unique():
            current = origins[months == month]
            first = current[0]
            if (idx[first] >= TEST) != (phase == "holdout"):
                continue
            for h in HORIZONS:
                valid = current[current + h < idx.searchsorted(cutoff)]
                if not len(valid):
                    continue
                truth = levels[valid+h]
                predictions = {("Last rate", "-"): levels[valid],
                               ("LSTM", "-"): frozen[h][valid-origins[0]]}
                for model in ("Ridge", "Boosting"):
                    configs = CONFIGS if phase == "validation" else (selected[model],)
                    for config in configs:
                        x = inputs(frame, code, config)
                        train = training_rows(idx, first, h, config)
                        target = np.log(levels[h:] / levels[:-h])
                        fit = estimator(model).fit(x.iloc[train], target[train])
                        predictions[model, config] = levels[valid] * np.exp(fit.predict(x.iloc[valid]))
                for (model, config), pred in predictions.items():
                    pieces.append(pd.DataFrame(dict(currency=code, phase=phase, model=model,
                        config=config, horizon=h, origin=idx[valid], target_date=idx[valid+h],
                        actual=truth, predicted=pred)))
        if phase == "validation":
            val = pd.concat(pieces)
            val["ae"] = abs(val.predicted - val.actual)
            means = val.groupby(["model", "config", "horizon"]).ae.mean()
            base = means.loc["Last rate", "-"]
            for model in ("Ridge", "Boosting"):
                scores = means.loc[model].unstack().div(base).mean(axis=1)
                selected[model] = scores.idxmin()
            print(code, "validation selected", selected, flush=True)
    for model, config in selected.items():
        x = inputs(frame, code, config)
        for h in HORIZONS:
            train = training_rows(idx, len(idx)-1, h, config)
            target = np.log(levels[h:] / levels[:-h])
            final_models[model, h] = estimator(model).fit(x.iloc[train], target[train])
    joblib.dump(dict(models=final_models, configs=selected, trained_through=str(idx[-1].date()),
                     feature_columns=list(frame), revision=REVISION), OUT / f"{code}_candidates.joblib")
    all_rows = pd.concat(pieces, ignore_index=True)
    return all_rows[all_rows.config.eq(all_rows.model.map(selected).fillna("-"))], selected


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    prices = load_prices()
    frame = features(prices)
    registry = ModelRegistry(ROOT / "artifacts")
    runs, configs = [], {}
    with threadpool_limits(limits=1):
        for name, code in CURRENCY_CODES.items():
            rows, configs[code] = run_currency(prices, frame, name, code, registry)
            runs.append(rows)
    rows = pd.concat(runs)
    assert np.isfinite(rows.predicted).all()
    paired = rows.groupby(["currency","phase","origin","horizon"]).model.nunique()
    assert (paired == 4).all()
    rows.to_csv(OUT / "forecasts.csv.gz", index=False)
    rows["error"] = rows.predicted - rows.actual
    metrics = rows.groupby(["phase","currency","horizon","model"]).error.agg(
        n="size", mae=lambda x: x.abs().mean(), rmse=lambda x: np.sqrt(np.mean(x*x)),
        p90_abs=lambda x: x.abs().quantile(.9), bias="mean").reset_index()
    base = metrics.loc[metrics.model == "Last rate", ["phase","currency","horizon","mae"]]
    metrics = metrics.merge(base.rename(columns={"mae":"baseline_mae"}), on=["phase","currency","horizon"])
    metrics["gain_pct"] = 100 * (1 - metrics.mae / metrics.baseline_mae)
    metrics.to_csv(OUT / "scorecard.csv", index=False)
    manifest = dict(revision=REVISION, sha256=hashlib.sha256(DATA.read_bytes()).hexdigest(),
                    first=str(prices.index[0].date()), last=str(prices.index[-1].date()),
                    complete_sessions=len(prices), validation_start=str(START.date()),
                    versions={pkg: version(pkg) for pkg in ("numpy","pandas","scikit-learn","keras","tensorflow-cpu")},
                    holdout_start=str(TEST.date()), selected_configs=configs)
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(metrics.query("phase == 'holdout'").pivot(index=["currency","horizon"],columns="model",values="mae"))


if __name__ == "__main__":
    main()
    summarize()
