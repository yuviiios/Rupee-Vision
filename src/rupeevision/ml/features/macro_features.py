"""Macro feature sourcing and alignment for multivariate forecasting.

The univariate LSTM saw only the INR series itself. Exchange rates are driven by
observable external variables, so this module sources real daily histories for the
assets that move INR and aligns them onto the RBI reference-rate calendar:

* **Gold** and **Crude oil** — India's two largest import categories by value, so
  their price directly drives the current-account deficit and therefore INR.
* **S&P 500** — proxy for global risk appetite and emerging-market capital flows.
* **US Dollar Index (DXY)** — dollar strength against a basket; mechanically the
  dominant term in USD/INR.
* **US 10-year Treasury yield** — the rate differential that drives carry flows.

Data comes from Yahoo Finance via ``yfinance``: real observed prices, free, no key.
Downloads are cached to CSV because training reruns must be reproducible and the
upstream endpoint is rate-limited and occasionally unavailable.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from rupeevision.core.exceptions import DataError
from rupeevision.core.logging import get_logger

if TYPE_CHECKING:  # pragma: no cover
    from rupeevision.ml.data import RateHistory

logger = get_logger(__name__)

#: Yahoo Finance ticker -> feature column name. Continuous futures contracts (``=F``)
#: are used for the commodities because spot series are not freely available.
MACRO_TICKERS: dict[str, str] = {
    "GC=F": "gold",
    "CL=F": "crude_oil",
    "^GSPC": "sp500",
    "DX-Y.NYB": "dxy",
    "^TNX": "us_10y_yield",
}

#: Default on-disk cache for downloaded histories.
DEFAULT_CACHE_PATH = Path("data") / "macro_features.csv"

#: A download yielding fewer rows than this is treated as an upstream failure
#: rather than a legitimately short series.
_MIN_USABLE_ROWS = 500


class MacroFeatureBuilder:
    """Build a multivariate frame of INR rates plus aligned macro features.

    Args:
        history: Validated INR rate history; supplies the target calendar.
        cache_path: CSV cache location for downloaded macro histories.
    """

    def __init__(self, history: RateHistory, cache_path: Path | None = None) -> None:
        self.history = history
        self.cache_path = Path(cache_path) if cache_path else DEFAULT_CACHE_PATH

    @property
    def feature_columns(self) -> list[str]:
        """Names of the macro columns added to the frame."""
        return list(MACRO_TICKERS.values())

    def load_macro_frame(self, *, refresh: bool = False) -> pd.DataFrame:
        """Return raw daily macro closes, indexed by date.

        Args:
            refresh: Ignore the cache and re-download.

        Returns:
            DataFrame indexed by ``DatetimeIndex`` with one column per macro feature.

        Raises:
            DataError: If no usable data is available from cache or upstream.
        """
        if not refresh and self.cache_path.exists():
            frame = pd.read_csv(self.cache_path, index_col=0, parse_dates=True)
            missing = set(self.feature_columns) - set(frame.columns)
            if missing:
                logger.info("macro_cache_incomplete", missing=sorted(missing))
            else:
                logger.info("macro_cache_hit", path=str(self.cache_path), rows=len(frame))
                return frame

        frame = self._download()
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(self.cache_path)
        logger.info("macro_cache_written", path=str(self.cache_path), rows=len(frame))
        return frame

    def _download(self) -> pd.DataFrame:
        """Download every macro series from Yahoo Finance.

        The requested window is the INR calendar's own span, so the result needs no
        extrapolation beyond what the target series covers.

        Raises:
            DataError: If ``yfinance`` is absent or every ticker fails.
        """
        try:
            import yfinance
        except ImportError as exc:  # pragma: no cover - optional at runtime
            msg = "yfinance is required to build macro features; pip install yfinance"
            raise DataError(msg) from exc

        start = self.history.frame.index[0]
        end = self.history.frame.index[-1] + pd.Timedelta(days=1)

        series: dict[str, pd.Series] = {}
        for ticker, column in MACRO_TICKERS.items():
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    raw = yfinance.download(
                        ticker,
                        start=start,
                        end=end,
                        progress=False,
                        auto_adjust=True,
                        threads=False,
                    )
            except Exception as exc:  # noqa: BLE001 - upstream raises many types
                logger.warning("macro_download_failed", ticker=ticker, error=str(exc))
                continue

            close = self._extract_close(raw, ticker)
            if close is None or len(close) < _MIN_USABLE_ROWS:
                logger.warning(
                    "macro_download_unusable",
                    ticker=ticker,
                    rows=0 if close is None else len(close),
                )
                continue

            series[column] = close
            logger.info(
                "macro_downloaded",
                ticker=ticker,
                column=column,
                rows=len(close),
                first=str(close.index[0].date()),
                last=str(close.index[-1].date()),
            )

        if not series:
            msg = "no macro series could be downloaded; check network connectivity"
            raise DataError(msg)

        frame = pd.DataFrame(series).sort_index()
        frame.index.name = "date"
        return frame

    @staticmethod
    def _extract_close(raw: pd.DataFrame | None, ticker: str) -> pd.Series | None:
        """Pull the close column out of a yfinance response.

        yfinance returns a MultiIndex ``(field, ticker)`` column layout for some
        calls and a flat layout for others; both shapes must be handled or a valid
        download is silently discarded.
        """
        if raw is None or raw.empty:
            return None
        if isinstance(raw.columns, pd.MultiIndex):
            if ("Close", ticker) in raw.columns:
                close = raw[("Close", ticker)]
            elif "Close" in raw.columns.get_level_values(0):
                close = raw.xs("Close", axis=1, level=0).iloc[:, 0]
            else:
                return None
        elif "Close" in raw.columns:
            close = raw["Close"]
        else:
            return None
        return pd.to_numeric(close, errors="coerce").dropna()

    def build(self, *, refresh: bool = False) -> pd.DataFrame:
        """Return INR rates joined with aligned macro features.

        Macro markets and the RBI calendar have different holidays, so each macro
        series is forward-filled onto the INR index: on a day a foreign market was
        shut, the last observed price is the information actually available. Rows
        before a feature's first observation cannot be filled that way and are
        dropped, which trims the start of the frame to the latest feature inception.

        Args:
            refresh: Bypass the download cache.

        Returns:
            DataFrame indexed by date with the INR currency columns first, followed
            by :attr:`feature_columns`. Free of ``NaN``.

        Raises:
            DataError: If the join leaves no usable rows.
        """
        frame = self.history.frame.copy()
        macro = self.load_macro_frame(refresh=refresh)

        aligned = macro.reindex(frame.index.union(macro.index)).ffill().reindex(frame.index)
        combined = frame.join(aligned, how="left")

        before = len(combined)
        combined = combined.dropna()
        if combined.empty:
            msg = "macro alignment produced no usable rows"
            raise DataError(msg)

        logger.info(
            "macro_frame_built",
            rows=len(combined),
            dropped_leading_rows=before - len(combined),
            columns=list(combined.columns),
            first=str(combined.index[0].date()),
            last=str(combined.index[-1].date()),
        )
        return combined

    def correlations(self, currency: str, window: int = 90) -> dict[str, float]:
        """Correlate daily returns of ``currency`` against each macro feature.

        Returns are used rather than levels: two series that both trend upward over
        decades correlate near 1.0 in levels regardless of any real relationship.

        Args:
            currency: Dataset currency name, e.g. ``"US Dollar"``.
            window: Number of trailing observations to correlate over.

        Returns:
            Mapping of feature name to Pearson correlation of daily returns.

        Raises:
            DataError: If ``currency`` is absent from the dataset.
        """
        frame = self.build()
        if currency not in frame.columns:
            msg = f"currency {currency!r} is not present in the dataset"
            raise DataError(msg)

        returns = frame[[currency, *self.feature_columns]].pct_change().dropna().tail(window)
        target = returns[currency]
        return {
            column: float(target.corr(returns[column]))
            for column in self.feature_columns
            if column in returns.columns
        }
