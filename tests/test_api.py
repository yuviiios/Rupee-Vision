"""Integration tests for the HTTP layer.

These use the app factory with fake services, so no TensorFlow import and no
artifact loading occurs. They pin the response contract and, in particular, the
status codes for malformed input — the original endpoint returned HTTP 500 for a
non-numeric ``days`` and leaked internal exception text.
"""

from __future__ import annotations

from typing import Any

import pytest

from rupeevision.api.app import Services, create_app
from rupeevision.core.config import Settings
from rupeevision.core.constants import CURRENCIES, MAX_FORECAST_DAYS


class TestHealthAndReadiness:
    def test_health_is_ok(self, client: Any) -> None:
        response = client.get("/health")

        assert response.status_code == 200
        assert response.get_json() == {"status": "ok"}

    def test_ready_reports_model_and_data_checks(self, client: Any) -> None:
        response = client.get("/ready")

        assert response.status_code == 200
        body = response.get_json()
        assert body["status"] == "ready"
        assert body["checks"]["models"]["count"] == len(CURRENCIES)
        assert body["checks"]["data"]["status"] == "ok"

    def test_ready_fails_when_data_is_stale(
        self, settings: Settings, services: Services
    ) -> None:
        """Readiness must turn red on stale data so a balancer stops routing here."""
        strict = settings.model_copy(update={"max_data_staleness_days": 1})
        app = create_app(strict, services=services)

        response = app.test_client().get("/ready")

        assert response.status_code == 503
        body = response.get_json()
        assert body["status"] == "not_ready"
        assert body["checks"]["data"]["status"] == "stale"


class TestCurrencies:
    def test_lists_supported_currencies_with_codes(self, client: Any) -> None:
        response = client.get("/currencies")

        assert response.status_code == 200
        body = response.get_json()
        assert body["max_forecast_days"] == MAX_FORECAST_DAYS
        codes = {entry["code"] for entry in body["currencies"]}
        assert codes == {"USD", "EUR", "JPY", "GBP"}
        assert all(entry["latest_rate"] > 0 for entry in body["currencies"])


class TestPredict:
    def test_get_with_days_returns_that_horizon(self, client: Any) -> None:
        response = client.get("/predict?days=5")

        assert response.status_code == 200
        body = response.get_json()
        assert body["horizon_days"] == 5
        assert len(body["dates"]) == 5
        assert set(body["predictions"]) == set(CURRENCIES)

    def test_post_with_json_body(self, client: Any) -> None:
        response = client.post("/predict", json={"days": 3})

        assert response.status_code == 200
        assert response.get_json()["horizon_days"] == 3

    def test_defaults_to_seven_days(self, client: Any) -> None:
        assert client.post("/predict").get_json()["horizon_days"] == 7
        assert client.get("/predict").get_json()["horizon_days"] == 7

    def test_response_includes_as_of_so_staleness_is_visible(self, client: Any) -> None:
        """Clients must be able to see which data a forecast was based on."""
        body = client.get("/predict?days=1").get_json()

        assert "as_of" in body
        assert body["dates"][0] > body["as_of"]

    @pytest.mark.parametrize("days", [0, -5, 31, 1000])
    def test_out_of_range_days_is_400_not_500(self, client: Any, days: int) -> None:
        response = client.get(f"/predict?days={days}")

        assert response.status_code == 400
        error = response.get_json()["error"]
        assert error["code"] == "validation_error"
        assert error["details"][0]["field"] == "days"

    @pytest.mark.parametrize("days", ["abc", "", "1.5", "7; DROP TABLE rates", "<script>"])
    def test_non_integer_days_is_400_not_500(self, client: Any, days: str) -> None:
        """Regression test: ``int()`` inside a bare except produced HTTP 500."""
        response = client.get("/predict", query_string={"days": days})

        assert response.status_code == 400
        assert response.get_json()["error"]["code"] == "validation_error"

    def test_unknown_body_fields_are_rejected(self, client: Any) -> None:
        response = client.post("/predict", json={"days": 3, "unexpected": True})

        assert response.status_code == 400
        fields = {entry["field"] for entry in response.get_json()["error"]["details"]}
        assert "unexpected" in fields

    def test_non_json_body_falls_back_to_the_default(self, client: Any) -> None:
        """A missing content type must not dereference ``None`` and 500."""
        response = client.post("/predict", data="not json", content_type="text/plain")

        assert response.status_code == 200
        assert response.get_json()["horizon_days"] == 7

    def test_malformed_json_body_falls_back_to_the_default(self, client: Any) -> None:
        response = client.post("/predict", data="{broken", content_type="application/json")

        assert response.status_code == 200

    def test_query_parameter_takes_precedence_over_body(self, client: Any) -> None:
        response = client.post("/predict?days=2", json={"days": 20})

        assert response.get_json()["horizon_days"] == 2

    def test_oversized_body_is_rejected(self, client: Any) -> None:
        response = client.post(
            "/predict",
            data="x" * 100_000,
            content_type="application/json",
        )

        assert response.status_code == 413
        assert response.get_json()["error"]["code"] == "payload_too_large"


class TestErrorContract:
    def test_unknown_route_returns_structured_error(self, client: Any) -> None:
        response = client.get("/does-not-exist")

        assert response.status_code == 404
        assert "error" in response.get_json()

    def test_wrong_method_returns_405(self, client: Any) -> None:
        assert client.delete("/predict").status_code == 405

    def test_every_error_carries_a_request_id(self, client: Any) -> None:
        """A correlation id is what makes a generic 500 message actionable."""
        response = client.get("/predict?days=0")

        assert response.get_json()["error"]["request_id"]

    def test_internal_errors_do_not_leak_exception_text(
        self, settings: Settings, services: Services
    ) -> None:
        """Regression test for ``jsonify({"error": str(e)})``."""
        secret = "C:/private/path/artifacts/secret.pkl"

        class ExplodingForecaster:
            def forecast(self, days: int) -> None:
                raise RuntimeError(secret)

        broken = Services(
            settings=services.settings,
            history=services.history,
            registry=services.registry,
            forecaster=ExplodingForecaster(),  # type: ignore[arg-type]
        )
        app = create_app(settings, services=broken)
        app.config.update(TESTING=False, PROPAGATE_EXCEPTIONS=False)

        response = app.test_client().get("/predict?days=3")

        assert response.status_code == 500
        payload = response.get_data(as_text=True)
        assert secret not in payload
        assert "RuntimeError" not in payload
        assert response.get_json()["error"]["code"] == "internal_error"


class TestSecurityHeadersAndCors:
    def test_hardening_headers_are_present(self, client: Any) -> None:
        headers = client.get("/health").headers

        assert headers["X-Content-Type-Options"] == "nosniff"
        assert headers["X-Frame-Options"] == "DENY"
        assert headers["Referrer-Policy"] == "no-referrer"

    def test_request_id_is_echoed_to_the_client(self, client: Any) -> None:
        response = client.get("/health", headers={"X-Request-ID": "abc123"})

        assert response.headers["X-Request-ID"] == "abc123"

    def test_allowed_origin_is_permitted(self, client: Any) -> None:
        response = client.get("/health", headers={"Origin": "http://localhost:3000"})

        assert response.headers["Access-Control-Allow-Origin"] == "http://localhost:3000"

    def test_disallowed_origin_is_not_granted_cors_access(self, client: Any) -> None:
        """Replaces the previous wildcard ``CORS(app)`` policy."""
        response = client.get("/health", headers={"Origin": "https://evil.example"})

        assert "Access-Control-Allow-Origin" not in response.headers


class TestRateLimiting:
    def test_prediction_requests_are_throttled(
        self, settings: Settings, services: Services
    ) -> None:
        """One request costs up to 120 forward passes, so a cap is required."""
        limited = settings.model_copy(update={"rate_limit": "3 per minute"})
        client = create_app(limited, services=services).test_client()

        codes = [client.get("/predict?days=1").status_code for _ in range(5)]

        assert codes[:3] == [200, 200, 200]
        assert codes[3:] == [429, 429]

    def test_rate_limit_error_is_structured(
        self, settings: Settings, services: Services
    ) -> None:
        limited = settings.model_copy(update={"rate_limit": "1 per minute"})
        client = create_app(limited, services=services).test_client()

        client.get("/predict?days=1")
        response = client.get("/predict?days=1")

        assert response.status_code == 429
        assert response.get_json()["error"]["code"] == "rate_limit_exceeded"

    def test_probes_are_never_throttled(
        self, settings: Settings, services: Services
    ) -> None:
        """Orchestrators poll these constantly; a 429 would look like an outage."""
        limited = settings.model_copy(update={"rate_limit": "1 per minute"})
        client = create_app(limited, services=services).test_client()

        codes = {client.get("/health").status_code for _ in range(6)}

        assert codes == {200}
