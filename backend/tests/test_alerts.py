"""
tests/test_alerts.py — Alert management endpoint tests.

Covers: create, list, get, update (status/message), delete, auth guards, ownership.
"""
import pytest
from unittest.mock import patch


# ══════════════════════════════════════════════════════════════
# LOCAL FIXTURE: sample_anomaly
# ══════════════════════════════════════════════════════════════
@pytest.fixture()
def sample_anomaly(client, auth_headers, sample_market_data_with_history):
    """
    Create and return a detected anomaly using threshold=0.0 so it is
    always flagged regardless of market data values.
    """
    response = client.post("/api/v1/anomalies", json={
        "market_data_id": sample_market_data_with_history["id"],
        "threshold": 0.0,
    }, headers=auth_headers)
    assert response.status_code == 201, f"Anomaly creation failed: {response.text}"
    return response.json()


def _create_alert(client, auth_headers, anomaly_id, message=None):
    """Helper: create an alert and return its JSON body."""
    payload = {"anomaly_id": anomaly_id}
    if message is not None:
        payload["message"] = message
    r = client.post("/api/v1/alerts", json=payload, headers=auth_headers)
    assert r.status_code == 201, f"Alert creation failed: {r.text}"
    return r.json()


# ══════════════════════════════════════════════════════════════
# CREATE
# ══════════════════════════════════════════════════════════════
class TestCreateAlert:
    def test_create_returns_201(self, client, auth_headers, sample_anomaly):
        response = client.post("/api/v1/alerts", json={
            "anomaly_id": sample_anomaly["id"],
        }, headers=auth_headers)
        assert response.status_code == 201

    def test_response_has_required_fields(self, client, auth_headers, sample_anomaly):
        response = client.post("/api/v1/alerts", json={
            "anomaly_id": sample_anomaly["id"],
            "message": "Suspicious price spike",
        }, headers=auth_headers)
        body = response.json()
        for field in ("id", "anomaly_id", "user_id", "status", "message", "created_at", "updated_at"):
            assert field in body, f"Missing field: {field}"

    def test_default_status_is_pending(self, client, auth_headers, sample_anomaly):
        response = client.post("/api/v1/alerts", json={
            "anomaly_id": sample_anomaly["id"],
        }, headers=auth_headers)
        assert response.json()["status"] == "PENDING"

    def test_create_with_message_stores_message(self, client, auth_headers, sample_anomaly):
        msg = "Volume spike detected on AAPL"
        response = client.post("/api/v1/alerts", json={
            "anomaly_id": sample_anomaly["id"],
            "message": msg,
        }, headers=auth_headers)
        assert response.json()["message"] == msg

    def test_create_without_message_is_null(self, client, auth_headers, sample_anomaly):
        response = client.post("/api/v1/alerts", json={
            "anomaly_id": sample_anomaly["id"],
        }, headers=auth_headers)
        assert response.json()["message"] is None

    def test_anomaly_id_linked_correctly(self, client, auth_headers, sample_anomaly):
        response = client.post("/api/v1/alerts", json={
            "anomaly_id": sample_anomaly["id"],
        }, headers=auth_headers)
        assert response.json()["anomaly_id"] == sample_anomaly["id"]

    def test_nonexistent_anomaly_returns_404(self, client, auth_headers):
        response = client.post("/api/v1/alerts", json={
            "anomaly_id": 999999,
        }, headers=auth_headers)
        assert response.status_code == 404

    def test_zero_anomaly_id_returns_422(self, client, auth_headers):
        response = client.post("/api/v1/alerts", json={"anomaly_id": 0}, headers=auth_headers)
        assert response.status_code == 422

    def test_message_too_long_returns_422(self, client, auth_headers, sample_anomaly):
        response = client.post("/api/v1/alerts", json={
            "anomaly_id": sample_anomaly["id"],
            "message": "x" * 1001,          # max is 1000 chars
        }, headers=auth_headers)
        assert response.status_code == 422

    def test_create_requires_auth(self, client, sample_anomaly):
        response = client.post("/api/v1/alerts", json={"anomaly_id": sample_anomaly["id"]})
        assert response.status_code in (401, 403)


# ══════════════════════════════════════════════════════════════
# LIST
# ══════════════════════════════════════════════════════════════
class TestListAlerts:
    def test_list_returns_200(self, client, auth_headers):
        response = client.get("/api/v1/alerts", headers=auth_headers)
        assert response.status_code == 200

    def test_list_empty_for_new_user(self, client, auth_headers):
        response = client.get("/api/v1/alerts", headers=auth_headers)
        assert response.json() == []

    def test_list_returns_created_alert(self, client, auth_headers, sample_anomaly):
        _create_alert(client, auth_headers, sample_anomaly["id"])
        response = client.get("/api/v1/alerts", headers=auth_headers)
        assert len(response.json()) == 1

    def test_list_returns_multiple_alerts(self, client, auth_headers, sample_anomaly):
        _create_alert(client, auth_headers, sample_anomaly["id"], "First alert")
        _create_alert(client, auth_headers, sample_anomaly["id"], "Second alert")
        response = client.get("/api/v1/alerts", headers=auth_headers)
        assert len(response.json()) == 2

    def test_list_requires_auth(self, client):
        response = client.get("/api/v1/alerts")
        assert response.status_code in (401, 403)


# ══════════════════════════════════════════════════════════════
# GET SINGLE
# ══════════════════════════════════════════════════════════════
class TestGetAlert:
    def test_get_returns_200(self, client, auth_headers, sample_anomaly):
        alert = _create_alert(client, auth_headers, sample_anomaly["id"])
        response = client.get(f"/api/v1/alerts/{alert['id']}", headers=auth_headers)
        assert response.status_code == 200

    def test_get_returns_correct_alert(self, client, auth_headers, sample_anomaly):
        alert = _create_alert(client, auth_headers, sample_anomaly["id"], "Check me")
        response = client.get(f"/api/v1/alerts/{alert['id']}", headers=auth_headers)
        body = response.json()
        assert body["id"] == alert["id"]
        assert body["message"] == "Check me"

    def test_get_nonexistent_returns_404(self, client, auth_headers):
        response = client.get("/api/v1/alerts/999999", headers=auth_headers)
        assert response.status_code == 404

    def test_get_requires_auth(self, client):
        response = client.get("/api/v1/alerts/1")
        assert response.status_code in (401, 403)


# ══════════════════════════════════════════════════════════════
# UPDATE (PATCH)
# ══════════════════════════════════════════════════════════════
class TestUpdateAlert:
    def test_update_status_returns_200(self, client, auth_headers, sample_anomaly):
        alert = _create_alert(client, auth_headers, sample_anomaly["id"])
        response = client.patch(
            f"/api/v1/alerts/{alert['id']}",
            json={"status": "ACTIVE"},
            headers=auth_headers,
        )
        assert response.status_code == 200

    def test_update_status_persisted(self, client, auth_headers, sample_anomaly):
        alert = _create_alert(client, auth_headers, sample_anomaly["id"])
        client.patch(f"/api/v1/alerts/{alert['id']}", json={"status": "RESOLVED"}, headers=auth_headers)
        response = client.get(f"/api/v1/alerts/{alert['id']}", headers=auth_headers)
        assert response.json()["status"] == "RESOLVED"

    def test_update_message(self, client, auth_headers, sample_anomaly):
        alert = _create_alert(client, auth_headers, sample_anomaly["id"])
        response = client.patch(
            f"/api/v1/alerts/{alert['id']}",
            json={"message": "Updated investigation note"},
            headers=auth_headers,
        )
        assert response.json()["message"] == "Updated investigation note"

    def test_all_valid_status_transitions(self, client, auth_headers, sample_anomaly):
        """All four status values should be accepted."""
        alert = _create_alert(client, auth_headers, sample_anomaly["id"])
        alert_id = alert["id"]
        for status in ("ACTIVE", "RESOLVED", "DISMISSED", "PENDING"):
            r = client.patch(f"/api/v1/alerts/{alert_id}", json={"status": status}, headers=auth_headers)
            assert r.status_code == 200, f"Status '{status}' was rejected"
            assert r.json()["status"] == status

    def test_invalid_status_returns_422(self, client, auth_headers, sample_anomaly):
        alert = _create_alert(client, auth_headers, sample_anomaly["id"])
        response = client.patch(
            f"/api/v1/alerts/{alert['id']}",
            json={"status": "BOGUS_STATUS"},
            headers=auth_headers,
        )
        assert response.status_code == 422

    def test_update_nonexistent_returns_404(self, client, auth_headers):
        response = client.patch(
            "/api/v1/alerts/999999",
            json={"status": "RESOLVED"},
            headers=auth_headers,
        )
        assert response.status_code == 404

    def test_update_requires_auth(self, client, sample_anomaly):
        response = client.patch("/api/v1/alerts/1", json={"status": "RESOLVED"})
        assert response.status_code in (401, 403)


# ══════════════════════════════════════════════════════════════
# DELETE
# ══════════════════════════════════════════════════════════════
class TestDeleteAlert:
    def test_delete_returns_204(self, client, auth_headers, sample_anomaly):
        alert = _create_alert(client, auth_headers, sample_anomaly["id"])
        response = client.delete(f"/api/v1/alerts/{alert['id']}", headers=auth_headers)
        assert response.status_code == 204

    def test_deleted_alert_not_found(self, client, auth_headers, sample_anomaly):
        alert = _create_alert(client, auth_headers, sample_anomaly["id"])
        alert_id = alert["id"]
        client.delete(f"/api/v1/alerts/{alert_id}", headers=auth_headers)
        response = client.get(f"/api/v1/alerts/{alert_id}", headers=auth_headers)
        assert response.status_code == 404

    def test_delete_nonexistent_returns_404(self, client, auth_headers):
        response = client.delete("/api/v1/alerts/999999", headers=auth_headers)
        assert response.status_code == 404

    def test_delete_requires_auth(self, client):
        response = client.delete("/api/v1/alerts/1")
        assert response.status_code in (401, 403)


# ══════════════════════════════════════════════════════════════
# OWNERSHIP ISOLATION
# ══════════════════════════════════════════════════════════════
class TestAlertOwnership:
    def _other_user_headers(self, client, email, username):
        client.post("/api/v1/auth/register", json={
            "email": email, "username": username, "password": "SecurePass1",
        })
        login = client.post("/api/v1/auth/login", json={"email": email, "password": "SecurePass1"})
        return {"Authorization": f"Bearer {login.json()['access_token']}"}

    def test_user_cannot_list_other_users_alerts(self, client, auth_headers, sample_anomaly):
        _create_alert(client, auth_headers, sample_anomaly["id"])
        other_headers = self._other_user_headers(client, "otherA@example.com", "otheruserA")
        response = client.get("/api/v1/alerts", headers=other_headers)
        assert response.json() == []

    def test_user_cannot_get_other_users_alert(self, client, auth_headers, sample_anomaly):
        alert = _create_alert(client, auth_headers, sample_anomaly["id"])
        other_headers = self._other_user_headers(client, "otherB@example.com", "otheruserB")
        response = client.get(f"/api/v1/alerts/{alert['id']}", headers=other_headers)
        assert response.status_code == 404

    def test_user_cannot_update_other_users_alert(self, client, auth_headers, sample_anomaly):
        alert = _create_alert(client, auth_headers, sample_anomaly["id"])
        other_headers = self._other_user_headers(client, "otherC@example.com", "otheruserC")
        response = client.patch(
            f"/api/v1/alerts/{alert['id']}",
            json={"status": "DISMISSED"},
            headers=other_headers,
        )
        assert response.status_code == 404

    def test_user_cannot_delete_other_users_alert(self, client, auth_headers, sample_anomaly):
        alert = _create_alert(client, auth_headers, sample_anomaly["id"])
        other_headers = self._other_user_headers(client, "otherD@example.com", "otheruserD")
        response = client.delete(f"/api/v1/alerts/{alert['id']}", headers=other_headers)
        assert response.status_code == 404

# --------------------------------------------------------------
# SSE STREAMING ALERTS (Phase 8)
# --------------------------------------------------------------
class TestSSELiveAlerts:
    @patch("app.services.redis_service.get_async_redis")
    def test_stream_live_alerts_requires_valid_sse_token(self, mock_get_redis, client, auth_headers):
        # Mock Redis to just return an empty list immediately so the loop runs fast
        import asyncio
        class MockRedis:
            async def xread(self, *args, **kwargs):
                await asyncio.sleep(0.1)
                # Raise CancelledError to break the while True loop, 
                # since TestClient doesn't reliably send disconnect events
                raise asyncio.CancelledError()
        mock_get_redis.return_value = MockRedis()
        
        # 1. No token -> 422 (FastAPI validation error for missing query param)
        response = client.get("/api/v1/alerts/stream/live")
        assert response.status_code == 422
        
        # 2. Normal access token (wrong type) -> 401
        access_token = auth_headers["Authorization"].split(" ")[1]
        response = client.get(f"/api/v1/alerts/stream/live?token={access_token}")
        assert response.status_code == 401
        
        # 3. Valid SSE token -> 200
        token_resp = client.post("/api/v1/auth/sse-token", headers=auth_headers)
        assert token_resp.status_code == 200
        sse_token = token_resp.json()["sse_token"]
        
        # Must use client.stream otherwise httpx blocks infinitely trying to read the whole stream
        with client.stream("GET", f"/api/v1/alerts/stream/live?token={sse_token}") as response:
            assert response.status_code == 200

    @patch("app.services.redis_service.get_async_redis")
    def test_stream_live_alerts_filters_by_watchlist(self, mock_get_redis, client, auth_headers):
        import asyncio
        import json

        # Mock Redis to yield a batch of alerts, then cancel the stream
        class MockRedis:
            def __init__(self):
                self.called = False

            async def xread(self, *args, **kwargs):
                if not self.called:
                    self.called = True
                    # Return 3 anomalies, only one is AAPL
                    payloads = [
                        ("1-0", {"data": json.dumps({"symbol": "TSLA", "anomaly_score": 0.9})}),
                        ("2-0", {"data": json.dumps({"symbol": "AAPL", "anomaly_score": 0.95})}),
                        ("3-0", {"data": json.dumps({"symbol": "MSFT", "anomaly_score": 0.8})}),
                    ]
                    return [("stream_alerts", payloads)]
                else:
                    await asyncio.sleep(0.1)
                    raise asyncio.CancelledError()

        mock_get_redis.return_value = MockRedis()

        # 1. Create a watchlist and add "AAPL"
        wl_resp = client.post("/api/v1/watchlists", json={"name": "Tech"}, headers=auth_headers)
        assert wl_resp.status_code == 201
        wl_id = wl_resp.json()["id"]
        client.post(f"/api/v1/watchlists/{wl_id}/symbols", json={"symbol": "AAPL"}, headers=auth_headers)

        # 2. Get valid SSE token
        token_resp = client.post("/api/v1/auth/sse-token", headers=auth_headers)
        sse_token = token_resp.json()["sse_token"]

        # 3. Read stream and verify only AAPL is emitted
        received_data = []
        with client.stream("GET", f"/api/v1/alerts/stream/live?token={sse_token}") as response:
            assert response.status_code == 200
            for line in response.iter_lines():
                if line.startswith("data: "):
                    payload_str = line[len("data: "):]
                    received_data.append(json.loads(payload_str))

        # 4. Assert filtering logic (B2)
        assert len(received_data) == 1
        assert received_data[0]["symbol"] == "AAPL"
