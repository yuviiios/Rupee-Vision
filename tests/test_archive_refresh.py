from unittest.mock import Mock

import pandas as pd
import pytest
import requests

from rupeevision.external import rate_archive


@pytest.mark.parametrize("bad_unit", [False, True])
def test_latest_forecast_validates_archive_and_reuses_snapshot(client, monkeypatch, bad_unit):
    initial = client.get("/currencies").json["as_of"]
    dates = pd.date_range(initial, periods=32)
    rows = ["date,currency,rate,unit"]
    for day in dates:
        for code in ("EUR", "USD", "GBP", "JPY"):
            unit = 100 if code == "JPY" and not bad_unit else 1
            rows.append(f"{day.date()},{code},80,{unit}")
    get = Mock(return_value=Mock(content="\n".join(rows).encode()))
    monkeypatch.setattr(rate_archive.requests, "get", get)
    for _ in range(2):
        result = client.get("/predict?days=1&latest=1").json
        assert result["archive_status"] == ("unavailable" if bad_unit else "verified")
        assert result["as_of"] == (initial if bad_unit else dates[-1].date().isoformat())
        assert result["as_of"] == client.get("/currencies").json["as_of"]
    get.assert_called_once()
    if not bad_unit:
        client.application.extensions["rate_archive"].next_check = 0
        get.side_effect = requests.Timeout()
        fallback = client.get("/predict?latest=1&days=1").json
        assert fallback["archive_status"] == "unavailable"
        assert fallback["predictions"] == result["predictions"]
        assert fallback["as_of"] == result["as_of"]
