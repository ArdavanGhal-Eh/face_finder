"""
Integration Tests for FastAPI Endpoints
Tests:
- /api/dashboard
- /api/students CRUD
- /api/settings GET and PUT
- /api/benchmark/metrics
"""

import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.database import init_db

client = TestClient(app)


@pytest.fixture(autouse=True)
def setup_database():
    init_db()


def test_get_dashboard():
    response = client.get("/api/dashboard")
    assert response.status_code == 200
    data = response.json()
    assert "total_students" in data
    assert "active_students" in data
    assert "total_sessions" in data


def test_student_api_flow():
    # 1. Create student
    payload = {
        "name": "Integration Test Student",
        "student_number": "INT_987654",
        "has_consent": True,
        "status": "active"
    }
    res = client.post("/api/students", json=payload)
    assert res.status_code in [201, 409]
    
    if res.status_code == 201:
        sid = res.json()["id"]
        
        # 2. Get student
        res_get = client.get(f"/api/students/{sid}")
        assert res_get.status_code == 200
        assert res_get.json()["name"] == "Integration Test Student"
        
        # 3. Update student
        res_put = client.put(f"/api/students/{sid}", json={"name": "Integration Student Updated"})
        assert res_put.status_code == 200
        
        # 4. Delete student
        res_del = client.delete(f"/api/students/{sid}")
        assert res_del.status_code == 200


def test_settings_api():
    # Read settings
    res = client.get("/api/settings")
    assert res.status_code == 200
    data = res.json()
    assert "settings" in data
    assert "hardware" in data
    
    # Update settings
    res_update = client.put("/api/settings", json={"match_threshold": 0.55, "min_face_size": 30})
    assert res_update.status_code == 200
    assert res_update.json()["updated"]["match_threshold"] == 0.55


def test_benchmark_api_insufficient_data():
    res = client.get("/api/benchmark/metrics")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] in ["insufficient_data", "success"]
