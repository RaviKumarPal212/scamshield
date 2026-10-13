from __future__ import annotations

from fastapi.testclient import TestClient

from scamshield.api.app import create_app


class StubPredictor:
    model_id = "test-model"
    threshold = 0.609175443649292

    def __init__(self, probability: float):
        self.probability = probability

    def predict_spam_probability(self, message: str) -> float:
        return self.probability


def test_local_frontend_and_assets_are_served_by_api() -> None:
    with TestClient(create_app(StubPredictor(0.2))) as client:
        page = client.get("/")
        styles = client.get("/styles.css")
        script = client.get("/app.js")

    assert page.status_code == styles.status_code == script.status_code == 200
    assert 'id="analyze-form"' in page.text
    assert "text/css" in styles.headers["content-type"]
    assert "application/javascript" in script.headers["content-type"]


def test_health_reports_ready_with_injected_predictor() -> None:
    with TestClient(create_app(StubPredictor(0.8))) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ready",
        "model_id": "test-model",
        "device": "test-double",
    }


def test_analyze_returns_model_decision_and_rule_explanations() -> None:
    with TestClient(create_app(StubPredictor(0.8))) as client:
        response = client.post(
            "/v1/analyze/sms",
            json={"message": "Share your OTP immediately: http://bit.ly/demo"},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["classification"] == "spam_like"
    assert payload["spam_probability"] == 0.8
    assert payload["threshold"] == StubPredictor.threshold
    assert {signal["code"] for signal in payload["signals"]} == {
        "credential_request",
        "urgency_or_threat",
        "suspicious_link_characteristics",
    }
    assert "not proof of fraud" in payload["disclaimer"]


def test_not_flagged_is_not_called_safe() -> None:
    with TestClient(create_app(StubPredictor(0.2))) as client:
        response = client.post("/v1/analyze/sms", json={"message": "See you tomorrow"})

    assert response.status_code == 200
    assert response.json()["classification"] == "not_flagged"
    assert "not guaranteed safe" in response.json()["disclaimer"]


def test_rejects_blank_oversized_and_extra_fields() -> None:
    with TestClient(create_app(StubPredictor(0.2))) as client:
        blank = client.post("/v1/analyze/sms", json={"message": "   "})
        oversized = client.post("/v1/analyze/sms", json={"message": "x" * 5001})
        extra = client.post("/v1/analyze/sms", json={"message": "hello", "user_id": "private"})

    assert blank.status_code == 422
    assert oversized.status_code == 422
    assert extra.status_code == 422
