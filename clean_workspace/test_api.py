"""
Basic smoke tests for the Hip Torque API server.
"""
import pytest
from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

def test_health_endpoint():
    """Test that the health endpoint returns 200."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_root_endpoint():
    """Test that the root endpoint serves the HTML page."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")

def test_static_favicon():
    """Test that favicon is served or returns appropriate response."""
    response = client.get("/favicon.ico")
    # Should be either 200 (if favicon exists) or 404 (if not found)
    assert response.status_code in [200, 404]

def test_analyze_endpoint_requires_params():
    """Test that the analyze endpoint requires proper parameters."""
    response = client.post("/api/analyze/")
    assert response.status_code == 422  # Validation error for missing required fields

def test_analyze_endpoint_with_sample_data():
    """Test the analyze endpoint with minimal valid parameters."""
    response = client.post(
        "/api/analyze/",
        data={
            "height_m": "1.70",
            "mass_kg": "70.0"
        }
    )
    # Should process successfully with sample data
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, dict)
