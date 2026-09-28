import pytest

pytest.importorskip("flask")

from app import create_app
from src.config import FEATURES


class FakeModel:
    def predict(self, frame):
        assert frame.shape == (1, len(FEATURES))
        return [25.0]


def valid_payload() -> dict[str, float]:
    return {
        "CRIM": 0.00632,
        "ZN": 18.0,
        "INDUS": 2.31,
        "CHAS": 0,
        "NOX": 0.538,
        "RM": 6.575,
        "AGE": 65.2,
        "DIS": 4.09,
        "RAD": 1,
        "TAX": 296,
        "PTRATIO": 15.3,
        "LSTAT": 4.98,
    }


def test_health():
    client = create_app(FakeModel(), FEATURES).test_client()

    response = client.get("/health")

    assert response.status_code == 200
    assert response.get_json()["model_ready"] is True


def test_health_reports_unavailable_model(monkeypatch):
    monkeypatch.setattr("app.load_model_bundle", lambda: (None, []))
    client = create_app().test_client()

    response = client.get("/health")

    assert response.status_code == 503
    assert response.get_json() == {
        "model_ready": False,
        "status": "model_not_trained",
    }


def test_prediction_api():
    client = create_app(FakeModel(), FEATURES).test_client()

    response = client.post(
        "/predict_api",
        json={"data": valid_payload()},
    )

    assert response.status_code == 200
    assert response.get_json()["predicted_median_value_thousands_usd"] == 25.0


def test_missing_feature():
    client = create_app(FakeModel(), FEATURES).test_client()
    payload = valid_payload()
    payload.pop("RM")

    response = client.post(
        "/predict_api",
        json={"data": payload},
    )

    assert response.status_code == 400
    assert "Missing features" in response.get_json()["error"]


def test_unexpected_feature():
    client = create_app(FakeModel(), FEATURES).test_client()
    payload = valid_payload()
    payload["EXTRA"] = 1

    response = client.post(
        "/predict_api",
        json={"data": payload},
    )

    assert response.status_code == 400
    assert "Unexpected features" in response.get_json()["error"]


def test_non_numeric_value():
    client = create_app(FakeModel(), FEATURES).test_client()
    payload = valid_payload()
    payload["RM"] = "six"

    response = client.post(
        "/predict_api",
        json={"data": payload},
    )

    assert response.status_code == 400
    assert "RM must be numeric" in response.get_json()["error"]


def test_non_finite_value():
    client = create_app(FakeModel(), FEATURES).test_client()
    payload = valid_payload()
    payload["RM"] = "nan"

    response = client.post(
        "/predict_api",
        json={"data": payload},
    )

    assert response.status_code == 400
    assert "RM must be finite" in response.get_json()["error"]


def test_invalid_chas():
    client = create_app(FakeModel(), FEATURES).test_client()
    payload = valid_payload()
    payload["CHAS"] = 2

    response = client.post(
        "/predict_api",
        json={"data": payload},
    )

    assert response.status_code == 400
    assert "CHAS must be 0 or 1" in response.get_json()["error"]
