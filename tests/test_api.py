"""API tests — skipped without FastAPI/httpx or the trained bundle."""
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient

    import api.main as main

    if not (main.settings.models_dir / "medpredict_bundle.joblib").exists():
        pytest.skip("Run `python -m medpredict.cli pipeline` first")
    return TestClient(main.app)


def test_health_has_disclaimer(client):
    assert "Not a diagnostic device" in client.get("/health").json()["disclaimer"]


def test_risk_response(client):
    body = client.post("/risk", json={}).json()
    assert 0 <= body["probability"] <= 1
    lo, hi = body["uncertainty_band"]
    assert lo <= hi
    assert "diagnosis" not in body and body["disclaimer"]


def test_risk_validation(client):
    assert client.post("/risk", json={"age": 5}).status_code == 422
