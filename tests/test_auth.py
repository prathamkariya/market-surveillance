"""
tests/test_auth.py — Basic authentication endpoint tests.

Covers: registration, login, token validation, and protected route guards.
"""
import pytest


# ══════════════════════════════════════════════════════════════
# REGISTRATION
# ══════════════════════════════════════════════════════════════
class TestRegister:
    def test_register_returns_201(self, client):
        response = client.post("/api/v1/auth/register", json={
            "email": "newuser@example.com",
            "username": "newuser",
            "password": "SecurePass1",
        })
        assert response.status_code == 201

    def test_register_response_has_required_fields(self, client):
        response = client.post("/api/v1/auth/register", json={
            "email": "fields@example.com",
            "username": "fielduser",
            "password": "SecurePass1",
        })
        body = response.json()
        assert "id" in body
        assert body["email"] == "fields@example.com"
        assert body["username"] == "fielduser"
        assert body["is_active"] is True
        assert "created_at" in body

    def test_register_does_not_expose_password(self, client):
        response = client.post("/api/v1/auth/register", json={
            "email": "secure@example.com",
            "username": "secureuser",
            "password": "SecurePass1",
        })
        body = response.json()
        assert "password" not in body
        assert "hashed_password" not in body

    def test_register_duplicate_email_returns_409(self, client, registered_user):
        response = client.post("/api/v1/auth/register", json={
            "email": "test@example.com",        # same as registered_user fixture
            "username": "differentusername",
            "password": "SecurePass1",
        })
        assert response.status_code == 409

    def test_register_duplicate_username_returns_409(self, client, registered_user):
        response = client.post("/api/v1/auth/register", json={
            "email": "different@example.com",
            "username": "testuser",             # same as registered_user fixture
            "password": "SecurePass1",
        })
        assert response.status_code == 409

    def test_register_password_no_digit_returns_422(self, client):
        response = client.post("/api/v1/auth/register", json={
            "email": "nodigit@example.com",
            "username": "nodigituser",
            "password": "NoDigitsHere",         # missing digit
        })
        assert response.status_code == 422

    def test_register_password_no_uppercase_returns_422(self, client):
        response = client.post("/api/v1/auth/register", json={
            "email": "nocase@example.com",
            "username": "nocaseuser",
            "password": "nouppercase1",         # missing uppercase
        })
        assert response.status_code == 422

    def test_register_password_too_short_returns_422(self, client):
        response = client.post("/api/v1/auth/register", json={
            "email": "short@example.com",
            "username": "shortpwuser",
            "password": "Sh1",                  # < 8 chars
        })
        assert response.status_code == 422

    def test_register_invalid_username_chars_returns_422(self, client):
        response = client.post("/api/v1/auth/register", json={
            "email": "badname@example.com",
            "username": "bad user!",            # spaces and ! not allowed
            "password": "SecurePass1",
        })
        assert response.status_code == 422

    def test_register_username_too_short_returns_422(self, client):
        response = client.post("/api/v1/auth/register", json={
            "email": "ab@example.com",
            "username": "ab",                   # < 3 chars
            "password": "SecurePass1",
        })
        assert response.status_code == 422

    def test_register_invalid_email_format_returns_422(self, client):
        response = client.post("/api/v1/auth/register", json={
            "email": "not-an-email",
            "username": "bademail",
            "password": "SecurePass1",
        })
        assert response.status_code == 422

    def test_register_missing_email_returns_422(self, client):
        response = client.post("/api/v1/auth/register", json={
            "username": "nomail",
            "password": "SecurePass1",
        })
        assert response.status_code == 422

    def test_register_missing_password_returns_422(self, client):
        response = client.post("/api/v1/auth/register", json={
            "email": "nopw@example.com",
            "username": "nopwuser",
        })
        assert response.status_code == 422


# ══════════════════════════════════════════════════════════════
# LOGIN
# ══════════════════════════════════════════════════════════════
class TestLogin:
    def test_login_returns_200(self, client, registered_user):
        response = client.post("/api/v1/auth/login", json={
            "email": "test@example.com",
            "password": "SecurePass1",
        })
        assert response.status_code == 200

    def test_login_response_has_access_token(self, client, registered_user):
        response = client.post("/api/v1/auth/login", json={
            "email": "test@example.com",
            "password": "SecurePass1",
        })
        body = response.json()
        assert "access_token" in body
        assert body["token_type"] == "bearer"
        assert isinstance(body["access_token"], str)
        assert len(body["access_token"]) > 20

    def test_login_response_has_refresh_token(self, client, registered_user):
        response = client.post("/api/v1/auth/login", json={
            "email": "test@example.com",
            "password": "SecurePass1",
        })
        body = response.json()
        assert "refresh_token" in body
        assert isinstance(body["refresh_token"], str)

    def test_login_tokens_are_different(self, client, registered_user):
        response = client.post("/api/v1/auth/login", json={
            "email": "test@example.com",
            "password": "SecurePass1",
        })
        body = response.json()
        assert body["access_token"] != body["refresh_token"]

    def test_login_wrong_password_returns_401(self, client, registered_user):
        response = client.post("/api/v1/auth/login", json={
            "email": "test@example.com",
            "password": "WrongPassword99",
        })
        assert response.status_code == 401

    def test_login_unknown_email_returns_401(self, client):
        response = client.post("/api/v1/auth/login", json={
            "email": "nobody@example.com",
            "password": "SecurePass1",
        })
        assert response.status_code == 401

    def test_login_missing_password_returns_422(self, client, registered_user):
        response = client.post("/api/v1/auth/login", json={
            "email": "test@example.com",
        })
        assert response.status_code == 422

    def test_login_empty_password_returns_422(self, client, registered_user):
        response = client.post("/api/v1/auth/login", json={
            "email": "test@example.com",
            "password": "",
        })
        assert response.status_code == 422


# ══════════════════════════════════════════════════════════════
# PROTECTED ROUTE GUARDS
# ══════════════════════════════════════════════════════════════
class TestProtectedRoutes:
    def test_no_auth_header_returns_401(self, client):
        response = client.get("/api/v1/market-data")
        assert response.status_code in (401, 403)

    def test_invalid_token_returns_401(self, client):
        response = client.get(
            "/api/v1/market-data",
            headers={"Authorization": "Bearer this.is.garbage.jwt"},
        )
        assert response.status_code == 401

    def test_malformed_auth_header_returns_401(self, client):
        response = client.get(
            "/api/v1/market-data",
            headers={"Authorization": "NotBearer sometoken"},
        )
        assert response.status_code in (401, 403)

    def test_valid_token_grants_access(self, client, auth_headers):
        # With a real token, a protected endpoint should respond (200, not 401)
        response = client.get("/api/v1/market-data", headers=auth_headers)
        assert response.status_code == 200

    def test_watchlists_also_protected(self, client):
        response = client.get("/api/v1/watchlists")
        assert response.status_code in (401, 403)

    def test_anomalies_also_protected(self, client):
        response = client.post("/api/v1/anomalies", json={"market_data_id": 1})
        assert response.status_code in (401, 403)

    def test_alerts_also_protected(self, client):
        response = client.get("/api/v1/alerts")
        assert response.status_code in (401, 403)
