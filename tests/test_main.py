"""tests/test_main.py — Tests for main application endpoints (like health)."""
from unittest.mock import patch

def test_health_check_db_ping(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    
def test_health_check_db_down_returns_503(client):
    # Mock Session.execute to throw an exception to simulate DB down
    with patch("sqlalchemy.orm.Session.execute") as mock_execute:
        mock_execute.side_effect = Exception("DB Connection Refused")
        response = client.get("/health")
        assert response.status_code == 503
        assert response.json()["status"] == "degraded"
