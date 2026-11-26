"""Exercise the Streamlit form against the actual release API."""
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import requests
from streamlit.testing.v1 import AppTest

from rupeevision.api.app import create_app
from rupeevision.core.config import Settings


def test_dashboard_release_and_refresh_fallback(monkeypatch):
    client = create_app(Settings()).test_client()
    calls = []

    def local_get(url, params=None, timeout=None):
        assert url.endswith("/predict")
        calls.append(params)
        response = client.get("/predict", query_string=params)
        assert response.status_code == 200
        return Mock(json=response.get_json)

    monkeypatch.setattr(requests, "get", local_get)
    def offline():
        raise requests.Timeout()

    monkeypatch.setattr("rupeevision.external.rate_archive.fetch_history", offline)
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "streamlit_app.py")).run(timeout=30)
    assert not app.exception and len(app.metric) == 4
    app.slider[0].set_value(30)
    app.checkbox[0].check()
    app.button[0].click().run()
    assert calls[-1] == {"days": 30, "latest": 1}
    assert not app.exception and len(app.warning) == 1
    rates = app.dataframe[0].value
    assert rates.shape == (30, 4)
    assert np.isfinite(rates.to_numpy()).all() and "INR per 100 JPY" in rates
    app.selectbox[0].select("JPY").run()
    assert any("Last rate" in item.value for item in app.caption)

    def disconnected(*args, **kwargs):
        raise requests.ConnectionError()

    monkeypatch.setattr(requests, "get", disconnected)
    app.button[0].click().run()
    assert not app.exception and len(app.error) == 1 and not app.metric
