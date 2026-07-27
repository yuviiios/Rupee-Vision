"""Train multivariate LSTM forecasters and benchmark them against univariate ones.

The original ``train.py`` fits a univariate model per currency: the only input is the
currency's own 30-day history. This script adds the observable external drivers
sourced by :mod:`rupeevision.ml.features.macro_features` (gold, crude oil, S&P 500,
DXY, US 10-year yield) and trains one model per currency on all of them.

Both variants are trained on the identical row range and split so the reported MAE
delta is attributable to the extra features and not to differing data. The
univariate arm is refit here rather than read from ``artifacts/`` because the shipped
models were trained on a longer span (the macro join trims the pre-2000 rows) and on
a different preprocessing path, so their metrics are not comparable.

**Models are fit on daily returns, not price levels.** INR has a strong secular
trend: USD/INR never exceeds 68.78 in the training rows but the test period runs
81.65-87.59. A ``MinMaxScaler`` fitted on training levels maps that test period to
roughly 1.44-1.64, outside the ``[0, 1]`` range the network's output ever learned to
produce, so a level-space model cannot extrapolate and its error is dominated by the
trend rather than by any real skill. Differencing removes the trend; predictions are
reconstructed back into rate units for reporting so MAE stays interpretable.

A **naive persistence baseline** (tomorrow equals today) is reported alongside both
arms. It is the number any forecaster must beat to be worth deploying, and on daily
FX it is a genuinely strong competitor.

Usage::

    python train_multivariate.py                 # all currencies
    python train_multivariate.py --currency "US Dollar" --epochs 5
    python train_multivariate.py --refresh       # re-download macro history
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import joblib
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler

from rupeevision.core.config import get_settings
from rupeevision.core.constants import CURRENCIES
from rupeevision.core.logging import configure_logging, get_logger
from rupeevision.ml.data import load_rate_history
from rupeevision.ml.features.macro_features import MacroFeatureBuilder

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd

logger = get_logger(__name__)

SEQ_LENGTH = 30
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
EPOCHS = 20
BATCH_SIZE = 32
SEED = 1337


def _slug(currency: str) -> str:
    """Filesystem-safe form of a currency name, matching the existing artifacts."""
    return currency.lower().replace(" ", "_")


def make_windows(data: np.ndarray, seq_length: int) -> tuple[np.ndarray, np.ndarray]:
    """Slice a scaled matrix into supervised windows.

    Args:
        data: Shape ``(rows, features)``; column 0 must be the target currency.
        seq_length: Number of trailing observations per window.

    Returns:
        ``(X, y)`` with shapes ``(samples, seq_length, features)`` and ``(samples,)``.
        ``y`` is the target column only — the macro features are inputs, not outputs.
        Window ``i`` covers rows ``i .. i+seq_length-1`` and predicts row
        ``i+seq_length``.
    """
    if len(data) <= seq_length:
        msg = f"need more than {seq_length} rows to build windows, got {len(data)}"
        raise ValueError(msg)
    windows = np.lib.stride_tricks.sliding_window_view(data, seq_length, axis=0)
    # sliding_window_view yields (samples, features, seq_length); LSTM wants time-major.
    X = windows[:-1].transpose(0, 2, 1).astype("float32")
    y = data[seq_length:, 0].astype("float32")
    return X, y


def to_returns(levels: np.ndarray) -> np.ndarray:
    """Convert a level matrix to fractional daily changes.

    Returns are stationary, so a scaler fitted on the training rows stays valid for
    later periods even when the underlying level has drifted well outside the
    training range.

    Args:
        levels: Shape ``(rows, features)``, strictly positive.

    Returns:
        Shape ``(rows - 1, features)``; row ``i`` is the change from ``i`` to ``i+1``.
    """
    return np.diff(levels, axis=0) / levels[:-1]


def _split(rows: int) -> tuple[slice, slice, slice]:
    """Chronological train/validation/test slices.

    Time series must not be shuffled: a random split lets the model interpolate
    between neighbouring days and reports an optimistic error.
    """
    train_end = int(rows * TRAIN_RATIO)
    val_end = int(rows * (TRAIN_RATIO + VAL_RATIO))
    return slice(0, train_end), slice(train_end, val_end), slice(val_end, rows)


def _build_model(seq_length: int, features: int) -> Any:
    """Two-layer LSTM regressor, mirroring the shipped architecture's shape."""
    from tensorflow import keras

    model = keras.Sequential(
        [
            keras.layers.Input(shape=(seq_length, features)),
            keras.layers.LSTM(64, return_sequences=True),
            keras.layers.LSTM(32),
            keras.layers.Dense(16, activation="relu"),
            keras.layers.Dense(1),
        ]
    )
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


def train_arm(
    frame: pd.DataFrame,
    currency: str,
    columns: list[str],
    *,
    epochs: int,
    label: str,
) -> dict[str, Any]:
    """Fit one model and score it in original rate units.

    Args:
        frame: Aligned frame containing ``columns``.
        currency: Target currency name; must be ``columns[0]``.
        columns: Model inputs, target first.
        epochs: Training epochs.
        label: ``"univariate"`` or ``"multivariate"``, for logging.

    Returns:
        Metrics dict plus the fitted ``model`` and ``scaler`` objects.
    """
    from tensorflow import keras

    keras.utils.set_random_seed(SEED)

    levels = frame[columns].to_numpy(dtype="float64")
    returns = to_returns(levels)
    # Row r of `returns` is the move from level row r to r+1, so a window ending at
    # returns row i-1 predicts returns row i, which lands on level row i+1.
    train_idx, val_idx, test_idx = _split(len(returns))

    # Fit the scaler on the training rows only. Fitting on the full array leaks the
    # test period's min/max into training and inflates the reported accuracy.
    scaler = MinMaxScaler()
    scaler.fit(returns[train_idx])
    scaled = scaler.transform(returns)

    X_train, y_train = make_windows(scaled[train_idx], SEQ_LENGTH)
    X_val, y_val = make_windows(scaled[val_idx], SEQ_LENGTH)
    X_test, y_test = make_windows(scaled[test_idx], SEQ_LENGTH)

    model = _build_model(SEQ_LENGTH, len(columns))
    model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=BATCH_SIZE,
        verbose=0,
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=4, restore_best_weights=True
            )
        ],
    )

    pred_scaled = model.predict(X_test, verbose=0).reshape(-1)
    actual_returns = _invert_target(scaler, y_test, len(columns))
    predicted_returns = _invert_target(scaler, pred_scaled, len(columns))

    # Reconstruct rate units so the error is interpretable and comparable to the
    # persistence baseline. The anchor for test sample j is the last level the model
    # actually saw, i.e. the level the predicted return is applied to.
    anchor_start = test_idx.start + SEQ_LENGTH
    anchors = levels[anchor_start : anchor_start + len(y_test), 0]
    actual_levels = anchors * (1 + actual_returns)
    predicted_levels = anchors * (1 + predicted_returns)

    mae = float(mean_absolute_error(actual_levels, predicted_levels))
    rmse = float(np.sqrt(mean_squared_error(actual_levels, predicted_levels)))
    mape = float(np.mean(np.abs((actual_levels - predicted_levels) / actual_levels)) * 100)
    # Fraction of days the predicted direction of movement was correct. A level-space
    # model can post a flattering MAE while being a coin flip on direction, which is
    # the part a user of a forecast actually acts on.
    hit_rate = float(np.mean(np.sign(predicted_returns) == np.sign(actual_returns)) * 100)

    logger.info(
        "arm_trained",
        currency=currency,
        arm=label,
        features=len(columns),
        train_samples=len(X_train),
        test_samples=len(X_test),
        test_mae=round(mae, 4),
        test_rmse=round(rmse, 4),
        test_mape=round(mape, 3),
        direction_hit_rate=round(hit_rate, 2),
    )
    return {
        "arm": label,
        "features": len(columns),
        "feature_names": columns,
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "test_mae": mae,
        "test_rmse": rmse,
        "test_mape": mape,
        "direction_hit_rate": hit_rate,
        "model": model,
        "scaler": scaler,
    }


def persistence_baseline(frame: pd.DataFrame, currency: str) -> dict[str, float]:
    """Score "tomorrow equals today" over the same test window as the models.

    Args:
        frame: Aligned frame containing ``currency``.
        currency: Target currency name.

    Returns:
        MAE, RMSE and MAPE in rate units.
    """
    levels = frame[[currency]].to_numpy(dtype="float64")
    returns = to_returns(levels)
    _, _, test_idx = _split(len(returns))

    anchor_start = test_idx.start + SEQ_LENGTH
    count = len(returns[test_idx]) - SEQ_LENGTH
    anchors = levels[anchor_start : anchor_start + count, 0]
    actual = levels[anchor_start + 1 : anchor_start + 1 + count, 0]

    mae = float(mean_absolute_error(actual, anchors))
    return {
        "arm": "persistence",
        "test_mae": mae,
        "test_rmse": float(np.sqrt(mean_squared_error(actual, anchors))),
        "test_mape": float(np.mean(np.abs((actual - anchors) / actual)) * 100),
        "test_samples": int(count),
    }


def _invert_target(scaler: MinMaxScaler, scaled_target: np.ndarray, features: int) -> np.ndarray:
    """Map a scaled target column back to its original units.

    ``MinMaxScaler.inverse_transform`` requires the full feature width, so the target
    is padded with zeros and only column 0 is read back. Each column is scaled
    independently, so the padding cannot affect the recovered target.
    """
    padded = np.zeros((len(scaled_target), features), dtype="float64")
    padded[:, 0] = scaled_target
    return scaler.inverse_transform(padded)[:, 0]


def run_currency(
    frame: pd.DataFrame,
    currency: str,
    feature_columns: list[str],
    *,
    epochs: int,
    artifacts: Path,
) -> dict[str, Any]:
    """Train both arms for one currency, score the baseline, persist the winner."""
    naive = persistence_baseline(frame, currency)
    uni = train_arm(frame, currency, [currency], epochs=epochs, label="univariate")
    multi = train_arm(
        frame, currency, [currency, *feature_columns], epochs=epochs, label="multivariate"
    )

    model_path = artifacts / f"lstm_multivariate_{_slug(currency)}.h5"
    scaler_path = artifacts / f"scaler_multivariate_{_slug(currency)}.pkl"
    multi["model"].save(model_path)
    joblib.dump(multi["scaler"], scaler_path)
    logger.info("artifacts_saved", currency=currency, model=str(model_path))

    improvement = (uni["test_mae"] - multi["test_mae"]) / uni["test_mae"] * 100
    vs_naive = (naive["test_mae"] - multi["test_mae"]) / naive["test_mae"] * 100
    return {
        "currency": currency,
        "rows": int(len(frame)),
        "persistence": naive,
        "univariate": {k: v for k, v in uni.items() if k not in {"model", "scaler"}},
        "multivariate": {k: v for k, v in multi.items() if k not in {"model", "scaler"}},
        "mae_improvement_pct": float(improvement),
        "mae_vs_persistence_pct": float(vs_naive),
        "model_path": str(model_path),
        "scaler_path": str(scaler_path),
    }


def main() -> None:
    """Entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--currency", action="append", choices=list(CURRENCIES))
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--refresh", action="store_true", help="re-download macro history")
    args = parser.parse_args()

    settings = get_settings()
    configure_logging(level=settings.log_level, json_output=False)

    history = load_rate_history(settings.data_path)
    builder = MacroFeatureBuilder(history)
    frame = builder.build(refresh=args.refresh)

    artifacts = settings.model_dir
    artifacts.mkdir(parents=True, exist_ok=True)

    targets = args.currency or list(CURRENCIES)
    results = [
        run_currency(
            frame, currency, builder.feature_columns, epochs=args.epochs, artifacts=artifacts
        )
        for currency in targets
    ]

    report = artifacts / "multivariate_benchmark.json"
    report.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print()
    print("=" * 92)
    print(f"1-DAY-AHEAD TEST MAE (INR)  |  {len(frame)} rows  |  {args.epochs} epochs max")
    print("=" * 92)
    print(
        f"{'currency':<16}{'naive':>9}{'uni':>9}{'multi':>9}"
        f"{'multi vs uni':>14}{'multi vs naive':>16}{'direction':>12}"
    )
    print("-" * 92)
    for r in results:
        print(
            f"{r['currency']:<16}"
            f"{r['persistence']['test_mae']:>9.4f}"
            f"{r['univariate']['test_mae']:>9.4f}"
            f"{r['multivariate']['test_mae']:>9.4f}"
            f"{r['mae_improvement_pct']:>13.1f}%"
            f"{r['mae_vs_persistence_pct']:>15.1f}%"
            f"{r['multivariate']['direction_hit_rate']:>11.1f}%"
        )
    print("-" * 92)
    print(f"features added: {', '.join(builder.feature_columns)}")
    print("naive = tomorrow equals today. Positive % means the model beat that column.")
    print(f"report: {report}")


if __name__ == "__main__":
    main()
