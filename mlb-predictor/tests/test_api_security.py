import pytest

pytest.importorskip("httpx")
from fastapi.testclient import TestClient

from src.api import app


def test_api_health_ok_from_test_client():
    client = TestClient(app)
    response = client.get("/api/health")
    assert response.status_code == 200


def test_static_index_not_subject_to_api_localhost_gate():
    client = TestClient(app)
    response = client.get("/")
    assert response.status_code == 200
