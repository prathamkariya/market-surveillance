"""
tests/test_market_data.py — OHLCV market data endpoint tests.

Covers: ingest, list, get, delete, validation, user isolation.
"""
import pytest


VALID_PAYLOAD = {
    "symbol": "aapl",           # lowercase — should be auto-uppercased
    "timestamp": "2024-01-15T10:00:00Z",
    "open": 185.50,
    "high": 186.20,
    "low": 185.10,
    "close": 185.90,
    "volume": 1250000.0,
}


# ══════════════════════════════════════════════════════════════
# CREATE
# ══════════════════════════════════════════════════════════════
class TestCreateMarketData:
    def test_create_returns_201(self, client, auth_headers):
        response = client.post("/api/v1/market-data", json=VALID_PAYLOAD, headers=auth_headers)
        assert response.status_code == 201

    def test_response_has_required_fields(self, client, auth_headers):
        response = client.post("/api/v1/market-data", json=VALID_PAYLOAD, headers=auth_headers)
        body = response.json()
        for field in ("id", "user_id", "symbol", "timestamp", "open", "high", "low", "close", "volume", "created_at"):
            assert field in body, f"Missing field: {field}"

    def test_symbol_auto_uppercased(self, client, auth_headers):
        response = client.post("/api/v1/market-data", json=VALID_PAYLOAD, headers=auth_headers)
        assert response.json()["symbol"] == "AAPL"

    def test_symbol_leading_trailing_whitespace_stripped(self, client, auth_headers):
        payload = {**VALID_PAYLOAD, "symbol": "  tsla  "}
        response = client.post("/api/v1/market-data", json=payload, headers=auth_headers)
        assert response.json()["symbol"] == "TSLA"

    def test_user_id_set_from_token(self, client, auth_headers, registered_user):
        response = client.post("/api/v1/market-data", json=VALID_PAYLOAD, headers=auth_headers)
        assert response.json()["user_id"] == registered_user["id"]

    def test_ohlcv_values_preserved(self, client, auth_headers):
        response = client.post("/api/v1/market-data", json=VALID_PAYLOAD, headers=auth_headers)
        body = response.json()
        assert float(body["open"])  == VALID_PAYLOAD["open"]
        assert float(body["high"])  == VALID_PAYLOAD["high"]
        assert float(body["low"])   == VALID_PAYLOAD["low"]
        assert float(body["close"]) == VALID_PAYLOAD["close"]
        assert float(body["volume"]) == VALID_PAYLOAD["volume"]

    def test_requires_auth(self, client):
        response = client.post("/api/v1/market-data", json=VALID_PAYLOAD)
        assert response.status_code in (401, 403)

    # ── Validation ──────────────────────────────────────────────
    def test_high_less_than_low_returns_422(self, client, auth_headers):
        bad = {**VALID_PAYLOAD, "high": 183.00, "low": 186.00}   # high < low
        response = client.post("/api/v1/market-data", json=bad, headers=auth_headers)
        assert response.status_code == 422

    def test_zero_open_returns_422(self, client, auth_headers):
        bad = {**VALID_PAYLOAD, "open": 0.0}                     # must be > 0
        response = client.post("/api/v1/market-data", json=bad, headers=auth_headers)
        assert response.status_code == 422

    def test_negative_price_returns_422(self, client, auth_headers):
        bad = {**VALID_PAYLOAD, "close": -5.0}
        response = client.post("/api/v1/market-data", json=bad, headers=auth_headers)
        assert response.status_code == 422

    def test_negative_volume_returns_422(self, client, auth_headers):
        bad = {**VALID_PAYLOAD, "volume": -100.0}
        response = client.post("/api/v1/market-data", json=bad, headers=auth_headers)
        assert response.status_code == 422

    def test_empty_symbol_returns_422(self, client, auth_headers):
        bad = {**VALID_PAYLOAD, "symbol": ""}
        response = client.post("/api/v1/market-data", json=bad, headers=auth_headers)
        assert response.status_code == 422

    def test_missing_timestamp_returns_422(self, client, auth_headers):
        bad = {k: v for k, v in VALID_PAYLOAD.items() if k != "timestamp"}
        response = client.post("/api/v1/market-data", json=bad, headers=auth_headers)
        assert response.status_code == 422


# ══════════════════════════════════════════════════════════════
# LIST
# ══════════════════════════════════════════════════════════════
class TestListMarketData:
    def test_list_returns_200(self, client, auth_headers):
        response = client.get("/api/v1/market-data", headers=auth_headers)
        assert response.status_code == 200

    def test_list_empty_for_new_user(self, client, auth_headers):
        response = client.get("/api/v1/market-data", headers=auth_headers)
        assert response.json() == []

    def test_list_returns_created_record(self, client, auth_headers):
        client.post("/api/v1/market-data", json=VALID_PAYLOAD, headers=auth_headers)
        response = client.get("/api/v1/market-data", headers=auth_headers)
        records = response.json()
        assert len(records) == 1
        assert records[0]["symbol"] == "AAPL"

    def test_list_multiple_records(self, client, auth_headers):
        for ts in ["2024-01-15T10:00:00Z", "2024-01-15T11:00:00Z", "2024-01-15T12:00:00Z"]:
            client.post("/api/v1/market-data", json={**VALID_PAYLOAD, "timestamp": ts}, headers=auth_headers)
        response = client.get("/api/v1/market-data", headers=auth_headers)
        assert len(response.json()) == 3

    def test_list_symbol_filter_aapl(self, client, auth_headers):
        client.post("/api/v1/market-data", json=VALID_PAYLOAD, headers=auth_headers)
        tsla_payload = {**VALID_PAYLOAD, "symbol": "TSLA", "timestamp": "2024-01-15T11:00:00Z"}
        client.post("/api/v1/market-data", json=tsla_payload, headers=auth_headers)

        response = client.get("/api/v1/market-data?symbol=AAPL", headers=auth_headers)
        results = response.json()
        assert len(results) == 1
        assert all(r["symbol"] == "AAPL" for r in results)

    def test_list_symbol_filter_case_insensitive(self, client, auth_headers):
        client.post("/api/v1/market-data", json=VALID_PAYLOAD, headers=auth_headers)
        response = client.get("/api/v1/market-data?symbol=aapl", headers=auth_headers)
        # Filter uses .upper() so lowercase query should still match
        assert len(response.json()) >= 1

    def test_list_requires_auth(self, client):
        response = client.get("/api/v1/market-data")
        assert response.status_code in (401, 403)


# ══════════════════════════════════════════════════════════════
# GET SINGLE
# ══════════════════════════════════════════════════════════════
class TestGetMarketData:
    def test_get_returns_200(self, client, auth_headers, sample_market_data):
        record_id = sample_market_data["id"]
        response = client.get(f"/api/v1/market-data/{record_id}", headers=auth_headers)
        assert response.status_code == 200

    def test_get_returns_correct_record(self, client, auth_headers, sample_market_data):
        record_id = sample_market_data["id"]
        response = client.get(f"/api/v1/market-data/{record_id}", headers=auth_headers)
        assert response.json()["id"] == record_id
        assert response.json()["symbol"] == "AAPL"

    def test_get_nonexistent_returns_404(self, client, auth_headers):
        response = client.get("/api/v1/market-data/999999", headers=auth_headers)
        assert response.status_code == 404

    def test_get_requires_auth(self, client, sample_market_data):
        response = client.get(f"/api/v1/market-data/{sample_market_data['id']}")
        assert response.status_code in (401, 403)


# ══════════════════════════════════════════════════════════════
# DELETE
# ══════════════════════════════════════════════════════════════
class TestDeleteMarketData:
    def test_delete_returns_204(self, client, auth_headers, sample_market_data):
        record_id = sample_market_data["id"]
        response = client.delete(f"/api/v1/market-data/{record_id}", headers=auth_headers)
        assert response.status_code == 204

    def test_deleted_record_not_found(self, client, auth_headers, sample_market_data):
        record_id = sample_market_data["id"]
        client.delete(f"/api/v1/market-data/{record_id}", headers=auth_headers)
        response = client.get(f"/api/v1/market-data/{record_id}", headers=auth_headers)
        assert response.status_code == 404

    def test_delete_nonexistent_returns_404(self, client, auth_headers):
        response = client.delete("/api/v1/market-data/999999", headers=auth_headers)
        assert response.status_code == 404

    def test_delete_requires_auth(self, client, sample_market_data):
        response = client.delete(f"/api/v1/market-data/{sample_market_data['id']}")
        assert response.status_code in (401, 403)


# ══════════════════════════════════════════════════════════════
# USER ISOLATION
# ══════════════════════════════════════════════════════════════
class TestMarketDataIsolation:
    def _register_and_login(self, client, email, username):
        """Helper: register + login a second user, return auth headers."""
        client.post("/api/v1/auth/register", json={
            "email": email, "username": username, "password": "SecurePass1",
        })
        login = client.post("/api/v1/auth/login", json={"email": email, "password": "SecurePass1"})
        return {"Authorization": f"Bearer {login.json()['access_token']}"}

    def test_user_cannot_list_other_users_data(self, client, sample_market_data, auth_headers):
        other_headers = self._register_and_login(client, "other@example.com", "otheruser99")
        response = client.get("/api/v1/market-data", headers=other_headers)
        assert response.json() == []

    def test_user_cannot_get_other_users_record(self, client, sample_market_data, auth_headers):
        other_headers = self._register_and_login(client, "other2@example.com", "otheruser98")
        record_id = sample_market_data["id"]
        response = client.get(f"/api/v1/market-data/{record_id}", headers=other_headers)
        assert response.status_code == 404

    def test_user_cannot_delete_other_users_record(self, client, sample_market_data, auth_headers):
        other_headers = self._register_and_login(client, "other3@example.com", "otheruser97")
        record_id = sample_market_data["id"]
        response = client.delete(f"/api/v1/market-data/{record_id}", headers=other_headers)
        assert response.status_code == 404
