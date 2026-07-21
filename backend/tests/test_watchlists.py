"""
tests/test_watchlists.py

Full test suite for the watchlist feature.
Tests all CRUD operations, symbol management, ownership isolation,
and validation errors.
"""
import pytest


# ══════════════════════════════════════════════════════════════
# WATCHLIST CRUD
# ══════════════════════════════════════════════════════════════
class TestCreateWatchlist:
    def test_create_watchlist_returns_201(self, client, auth_headers):
        response = client.post(
            "/api/v1/watchlists",
            json={"name": "Tech Stocks"},
            headers=auth_headers,
        )
        assert response.status_code == 201

    def test_create_watchlist_response_fields(self, client, auth_headers):
        response = client.post(
            "/api/v1/watchlists",
            json={"name": "My Watchlist", "description": "Stocks I'm monitoring"},
            headers=auth_headers,
        )
        body = response.json()
        assert body["name"] == "My Watchlist"
        assert body["description"] == "Stocks I'm monitoring"
        assert "id" in body
        assert "user_id" in body
        assert "created_at" in body
        assert body["symbols"] == []   # New watchlist has no symbols

    def test_create_watchlist_without_description(self, client, auth_headers):
        response = client.post(
            "/api/v1/watchlists",
            json={"name": "No Description"},
            headers=auth_headers,
        )
        assert response.status_code == 201
        assert response.json()["description"] is None

    def test_create_duplicate_watchlist_name_returns_409(self, client, auth_headers):
        client.post("/api/v1/watchlists", json={"name": "Duplicate"}, headers=auth_headers)
        response = client.post(
            "/api/v1/watchlists",
            json={"name": "Duplicate"},
            headers=auth_headers,
        )
        assert response.status_code == 409

    def test_create_watchlist_empty_name_returns_422(self, client, auth_headers):
        response = client.post(
            "/api/v1/watchlists",
            json={"name": ""},
            headers=auth_headers,
        )
        assert response.status_code == 422

    def test_create_watchlist_requires_auth(self, client):
        response = client.post("/api/v1/watchlists", json={"name": "No Auth"})
        assert response.status_code == 403


class TestGetWatchlist:
    def test_get_watchlist_by_id(self, client, auth_headers):
        create_resp = client.post(
            "/api/v1/watchlists", json={"name": "Get Test"}, headers=auth_headers
        )
        wl_id = create_resp.json()["id"]

        get_resp = client.get(f"/api/v1/watchlists/{wl_id}", headers=auth_headers)
        assert get_resp.status_code == 200
        assert get_resp.json()["name"] == "Get Test"

    def test_get_nonexistent_watchlist_returns_404(self, client, auth_headers):
        response = client.get("/api/v1/watchlists/999999", headers=auth_headers)
        assert response.status_code == 404

    def test_get_requires_auth(self, client, auth_headers):
        create_resp = client.post(
            "/api/v1/watchlists", json={"name": "Auth Test"}, headers=auth_headers
        )
        wl_id = create_resp.json()["id"]
        response = client.get(f"/api/v1/watchlists/{wl_id}")   # no headers
        assert response.status_code == 403


class TestListWatchlists:
    def test_list_watchlists_empty_for_new_user(self, client, auth_headers):
        response = client.get("/api/v1/watchlists", headers=auth_headers)
        assert response.status_code == 200
        assert response.json() == []

    def test_list_watchlists_returns_all_user_watchlists(self, client, auth_headers):
        client.post("/api/v1/watchlists", json={"name": "List A"}, headers=auth_headers)
        client.post("/api/v1/watchlists", json={"name": "List B"}, headers=auth_headers)
        client.post("/api/v1/watchlists", json={"name": "List C"}, headers=auth_headers)

        response = client.get("/api/v1/watchlists", headers=auth_headers)
        assert response.status_code == 200
        assert len(response.json()) == 3

    def test_list_watchlists_includes_symbol_count(self, client, auth_headers):
        create_resp = client.post(
            "/api/v1/watchlists", json={"name": "With Symbols"}, headers=auth_headers
        )
        wl_id = create_resp.json()["id"]
        client.post(
            f"/api/v1/watchlists/{wl_id}/symbols",
            json={"symbol": "AAPL"},
            headers=auth_headers,
        )
        client.post(
            f"/api/v1/watchlists/{wl_id}/symbols",
            json={"symbol": "GOOG"},
            headers=auth_headers,
        )

        list_resp = client.get("/api/v1/watchlists", headers=auth_headers)
        watchlist = next(w for w in list_resp.json() if w["name"] == "With Symbols")
        assert watchlist["symbol_count"] == 2


class TestUpdateWatchlist:
    def test_update_watchlist_name(self, client, auth_headers):
        create_resp = client.post(
            "/api/v1/watchlists", json={"name": "Old Name"}, headers=auth_headers
        )
        wl_id = create_resp.json()["id"]

        update_resp = client.put(
            f"/api/v1/watchlists/{wl_id}",
            json={"name": "New Name"},
            headers=auth_headers,
        )
        assert update_resp.status_code == 200
        assert update_resp.json()["name"] == "New Name"

    def test_update_watchlist_description(self, client, auth_headers):
        create_resp = client.post(
            "/api/v1/watchlists", json={"name": "Desc Test"}, headers=auth_headers
        )
        wl_id = create_resp.json()["id"]

        update_resp = client.put(
            f"/api/v1/watchlists/{wl_id}",
            json={"description": "Updated description"},
            headers=auth_headers,
        )
        assert update_resp.json()["description"] == "Updated description"

    def test_update_to_duplicate_name_returns_409(self, client, auth_headers):
        client.post("/api/v1/watchlists", json={"name": "First"}, headers=auth_headers)
        create_resp = client.post(
            "/api/v1/watchlists", json={"name": "Second"}, headers=auth_headers
        )
        wl_id = create_resp.json()["id"]

        response = client.put(
            f"/api/v1/watchlists/{wl_id}",
            json={"name": "First"},   # conflicts with existing
            headers=auth_headers,
        )
        assert response.status_code == 409


class TestDeleteWatchlist:
    def test_delete_watchlist_returns_204(self, client, auth_headers):
        create_resp = client.post(
            "/api/v1/watchlists", json={"name": "To Delete"}, headers=auth_headers
        )
        wl_id = create_resp.json()["id"]

        delete_resp = client.delete(f"/api/v1/watchlists/{wl_id}", headers=auth_headers)
        assert delete_resp.status_code == 204

    def test_deleted_watchlist_not_found(self, client, auth_headers):
        create_resp = client.post(
            "/api/v1/watchlists", json={"name": "Gone"}, headers=auth_headers
        )
        wl_id = create_resp.json()["id"]
        client.delete(f"/api/v1/watchlists/{wl_id}", headers=auth_headers)

        get_resp = client.get(f"/api/v1/watchlists/{wl_id}", headers=auth_headers)
        assert get_resp.status_code == 404


# ══════════════════════════════════════════════════════════════
# SYMBOL MANAGEMENT
# ══════════════════════════════════════════════════════════════
class TestWatchlistSymbols:
    @pytest.fixture(autouse=True)
    def create_wl(self, client, auth_headers):
        """Create a watchlist for use in all symbol tests."""
        resp = client.post(
            "/api/v1/watchlists", json={"name": "Symbol Tests"}, headers=auth_headers
        )
        self.wl_id = resp.json()["id"]

    def test_add_symbol_returns_201(self, client, auth_headers):
        response = client.post(
            f"/api/v1/watchlists/{self.wl_id}/symbols",
            json={"symbol": "AAPL"},
            headers=auth_headers,
        )
        assert response.status_code == 201

    def test_add_symbol_auto_uppercased(self, client, auth_headers):
        response = client.post(
            f"/api/v1/watchlists/{self.wl_id}/symbols",
            json={"symbol": "msft"},   # lowercase input
            headers=auth_headers,
        )
        assert response.status_code == 201
        assert response.json()["symbol"] == "MSFT"

    def test_add_symbol_with_notes(self, client, auth_headers):
        response = client.post(
            f"/api/v1/watchlists/{self.wl_id}/symbols",
            json={"symbol": "TSLA", "notes": "Watching for Q4 earnings"},
            headers=auth_headers,
        )
        assert response.json()["notes"] == "Watching for Q4 earnings"

    def test_add_duplicate_symbol_returns_409(self, client, auth_headers):
        client.post(
            f"/api/v1/watchlists/{self.wl_id}/symbols",
            json={"symbol": "GOOG"},
            headers=auth_headers,
        )
        response = client.post(
            f"/api/v1/watchlists/{self.wl_id}/symbols",
            json={"symbol": "GOOG"},
            headers=auth_headers,
        )
        assert response.status_code == 409

    def test_symbol_appears_in_watchlist_detail(self, client, auth_headers):
        client.post(
            f"/api/v1/watchlists/{self.wl_id}/symbols",
            json={"symbol": "NVDA"},
            headers=auth_headers,
        )
        get_resp = client.get(f"/api/v1/watchlists/{self.wl_id}", headers=auth_headers)
        symbols = [s["symbol"] for s in get_resp.json()["symbols"]]
        assert "NVDA" in symbols

    def test_remove_symbol_returns_204(self, client, auth_headers):
        client.post(
            f"/api/v1/watchlists/{self.wl_id}/symbols",
            json={"symbol": "AMD"},
            headers=auth_headers,
        )
        response = client.delete(
            f"/api/v1/watchlists/{self.wl_id}/symbols/AMD",
            headers=auth_headers,
        )
        assert response.status_code == 204

    def test_removed_symbol_not_in_watchlist(self, client, auth_headers):
        client.post(
            f"/api/v1/watchlists/{self.wl_id}/symbols",
            json={"symbol": "INTC"},
            headers=auth_headers,
        )
        client.delete(
            f"/api/v1/watchlists/{self.wl_id}/symbols/INTC",
            headers=auth_headers,
        )
        get_resp = client.get(f"/api/v1/watchlists/{self.wl_id}", headers=auth_headers)
        symbols = [s["symbol"] for s in get_resp.json()["symbols"]]
        assert "INTC" not in symbols

    def test_remove_nonexistent_symbol_returns_404(self, client, auth_headers):
        response = client.delete(
            f"/api/v1/watchlists/{self.wl_id}/symbols/NONEXISTENT",
            headers=auth_headers,
        )
        assert response.status_code == 404


# ══════════════════════════════════════════════════════════════
# OWNERSHIP ISOLATION
# ══════════════════════════════════════════════════════════════
class TestWatchlistOwnership:
    """
    Critical: one user must not be able to access another user's watchlists.
    Uses two separate users to verify isolation.
    """

    @pytest.fixture
    def second_user_headers(self, client) -> dict:
        """Register and login a second user."""
        client.post(
            "/api/v1/auth/register",
            json={
                "email": "second@example.com",
                "username": "seconduser",
                "password": "SecurePass1",
            },
        )
        login = client.post(
            "/api/v1/auth/login",
            json={"email": "second@example.com", "password": "SecurePass1"},
        )
        return {"Authorization": f"Bearer {login.json()['access_token']}"}

    def test_user_cannot_see_other_user_watchlist(
        self, client, auth_headers, second_user_headers
    ):
        # User 1 creates a watchlist
        create_resp = client.post(
            "/api/v1/watchlists",
            json={"name": "User1 Private"},
            headers=auth_headers,
        )
        wl_id = create_resp.json()["id"]

        # User 2 tries to access it — must get 404, not 403
        # (never reveal that the resource exists)
        response = client.get(
            f"/api/v1/watchlists/{wl_id}",
            headers=second_user_headers,
        )
        assert response.status_code == 404

    def test_user_only_sees_own_watchlists_in_list(
        self, client, auth_headers, second_user_headers
    ):
        # User 1 creates watchlists
        client.post("/api/v1/watchlists", json={"name": "U1 List"}, headers=auth_headers)

        # User 2 creates a watchlist
        client.post("/api/v1/watchlists", json={"name": "U2 List"}, headers=second_user_headers)

        # User 1 lists — should only see their own
        u1_list = client.get("/api/v1/watchlists", headers=auth_headers).json()
        u1_names = [w["name"] for w in u1_list]
        assert "U1 List" in u1_names
        assert "U2 List" not in u1_names

    def test_user_cannot_delete_other_user_watchlist(
        self, client, auth_headers, second_user_headers
    ):
        create_resp = client.post(
            "/api/v1/watchlists", json={"name": "Protected"}, headers=auth_headers
        )
        wl_id = create_resp.json()["id"]

        # User 2 tries to delete user 1's watchlist
        response = client.delete(
            f"/api/v1/watchlists/{wl_id}",
            headers=second_user_headers,
        )
        assert response.status_code == 404   # Not revealed, not 403

        # Watchlist still exists for user 1
        get_resp = client.get(f"/api/v1/watchlists/{wl_id}", headers=auth_headers)
        assert get_resp.status_code == 200
