"""Summarize the paired errors from benchmark_sessions.py."""
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent / "reports" / "fx_session_backtest"


def paired_interval(errors, model):
    """Circular moving-block bootstrap of the paired relative MAE improvement."""
    rng = np.random.default_rng(42)
    n = len(errors)
    block = min(30, n)
    starts = rng.integers(n, size=(2000, (n + block - 1) // block))
    indices = ((starts[:, :, None] + np.arange(block)) % n).reshape(len(starts), -1)[:, :n]
    base = errors["Last rate"].to_numpy()[indices].mean(axis=1)
    candidate = errors[model].to_numpy()[indices].mean(axis=1)
    return np.percentile(100 * (1 - candidate / base), [2.5, 97.5])


def main():
    rows = pd.read_csv(OUT / "forecasts.csv.gz", parse_dates=["origin", "target_date"])
    rows["ae"] = (rows.predicted - rows.actual).abs()
    holdout = rows.query("phase == 'holdout'")
    summary = []
    for (code, h), group in holdout.groupby(["currency", "horizon"]):
        errors = group.pivot(index="origin", columns="model", values="ae").sort_index()
        assert errors.notna().all().all() and len(errors.columns) == 4
        for model in errors:
            lo, hi = paired_interval(errors, model)
            summary.append(dict(currency=code, horizon=h, model=model, mae=errors[model].mean(),
                                gain_low_pct=lo, gain_high_pct=hi))
    pd.DataFrame(summary).to_csv(OUT / "paired_uncertainty.csv", index=False)

    score = pd.read_csv(OUT / "scorecard.csv")
    ranks = score.groupby(["phase", "currency", "model"]).gain_pct.mean().unstack()
    ranks.to_csv(OUT / "overall_gain.csv")
    print(ranks.round(2).to_string())
    # Compare selection stability in nonoverlapping six-month validation blocks.
    val = rows.query("phase == 'validation'").copy()
    val["block"] = val.origin.dt.year * 2 + (val.origin.dt.month > 6).astype(int)
    means = val.groupby(["currency", "block", "horizon", "model"]).ae.mean().unstack()
    by_block = (100 * (1 - means.div(means["Last rate"], axis=0))).groupby(["currency","block"]).mean()
    by_block.to_csv(OUT / "validation_blocks.csv")
    print("Validation blocks beating last rate (out of seven):")
    print((by_block > 0).groupby("currency").sum().to_string())


if __name__ == "__main__":
    main()
