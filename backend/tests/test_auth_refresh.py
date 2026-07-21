"""
tests/test_auth_refresh.py

Tests for Phase 2 JWT refresh token functionality:
- Login returns both access + refresh tokens
- /auth/refresh issues new access token, rotates refresh token
- /auth/logout revokes a specific refresh token
- /auth/logout-all revokes all tokens for a user
- Invalid/revoked/expired tokens are rejected
"""
import pytest


class TestLoginTokenPair:
    def test_login_returns_both_tokens(self, client, registered_user):
        """Login must return access_token, refresh_token, and token_type."""
        response = client.post(
            "/api/v1/auth/login",
            json={"email": "test@example.com", "password": "SecurePass1"},
        )
        assert response.status_code == 200
        body = response.json()
        assert "access_token" in body
        assert "refresh_token" in body
        assert body["token_type"] == "bearer"
        assert isinstance(body["expires_in"], int)
        assert body["expires_in"] > 0

    def test_login_tokens_are_different(self, client, registered_user):
        """Access token and refresh token must be different strings."""
        response = client.post(
            "/api/v1/auth/login",
            json={"email": "test@example.com", "password": "SecurePass1"},
        )
        body = response.json()
        assert body["access_token"] != body["refresh_token"]

    def test_login_invalid_password_no_tokens(self, client, registered_user):
        """Bad credentials must not return any tokens."""
        response = client.post(
            "/api/v1/auth/login",
            json={"email": "test@example.com", "password": "WrongPassword1"},
        )
        assert response.status_code == 401
        assert "access_token" not in response.json()


class TestRefreshToken:
    def test_refresh_returns_new_access_token(self, client, auth_tokens):
        """Valid refresh token returns new access token."""
        response = client.post(
            "/api/v1/auth/refresh",
            json={"refresh_token": auth_tokens["refresh_token"]},
        )
        assert response.status_code == 200
        body = response.json()
        assert "access_token" in body
        assert body["token_type"] == "bearer"
        assert isinstance(body["expires_in"], int)

    def test_refresh_new_access_token_is_valid(self, client, auth_tokens):
        """New access token from refresh must work on protected endpoints."""
        # Get new access token
        refresh_resp = client.post(
            "/api/v1/auth/refresh",
            json={"refresh_token": auth_tokens["refresh_token"]},
        )
        new_access = refresh_resp.json()["access_token"]

        # Use it on a protected endpoint
        me_resp = client.get(
            "/api/v1/auth/me",
            headers={"Authorization": f"Bearer {new_access}"},
        )
        assert me_resp.status_code == 200

    def test_refresh_token_rotation_revokes_old(self, client, auth_tokens):
        """A refresh token is single-use — using it again must fail."""
        old_refresh = auth_tokens["refresh_token"]

        # First use — should succeed
        first_resp = client.post(
            "/api/v1/auth/refresh",
            json={"refresh_token": old_refresh},
        )
        assert first_resp.status_code == 200

        # Second use of the same token — must fail (token was rotated/revoked)
        second_resp = client.post(
            "/api/v1/auth/refresh",
            json={"refresh_token": old_refresh},
        )
        assert second_resp.status_code == 401

    def test_refresh_invalid_token_rejected(self, client, registered_user):
        """Garbage refresh token must be rejected."""
        response = client.post(
            "/api/v1/auth/refresh",
            json={"refresh_token": "this-is-not-a-valid-token"},
        )
        assert response.status_code == 401

    def test_refresh_empty_token_rejected(self, client, registered_user):
        """Empty refresh token must be rejected by Pydantic validation."""
        response = client.post(
            "/api/v1/auth/refresh",
            json={"refresh_token": ""},
        )
        assert response.status_code == 422


class TestLogout:
    def test_logout_requires_auth(self, client, auth_tokens):
        """POST /auth/logout requires a valid access token."""
        response = client.post(
            "/api/v1/auth/logout",
            json={"refresh_token": auth_tokens["refresh_token"]},
            # No Authorization header
        )
        assert response.status_code == 403  # HTTPBearer returns 403 on missing

    def test_logout_revokes_refresh_token(self, client, auth_tokens, auth_headers):
        """After logout, the refresh token must no longer work."""
        refresh_token = auth_tokens["refresh_token"]

        # Logout
        logout_resp = client.post(
            "/api/v1/auth/logout",
            json={"refresh_token": refresh_token},
            headers=auth_headers,
        )
        assert logout_resp.status_code == 204

        # Try to use the refresh token — must fail
        refresh_resp = client.post(
            "/api/v1/auth/refresh",
            json={"refresh_token": refresh_token},
        )
        assert refresh_resp.status_code == 401

    def test_logout_returns_204_for_unknown_token(self, client, auth_headers):
        """Logout with a non-existent token still returns 204 (no info leakage)."""
        response = client.post(
            "/api/v1/auth/logout",
            json={"refresh_token": "nonexistent-token"},
            headers=auth_headers,
        )
        assert response.status_code == 204

    def test_logout_all_revokes_all_tokens(self, client, registered_user, auth_headers):
        """After logout-all, all refresh tokens for the user must be invalid."""
        # Log in again to get a second refresh token (simulate second device)
        second_login = client.post(
            "/api/v1/auth/login",
            json={"email": "test@example.com", "password": "SecurePass1"},
        )
        second_refresh = second_login.json()["refresh_token"]

        # Get first refresh token
        first_login = client.post(
            "/api/v1/auth/login",
            json={"email": "test@example.com", "password": "SecurePass1"},
        )
        first_refresh = first_login.json()["refresh_token"]

        # Logout all
        logout_all_resp = client.post("/api/v1/auth/logout-all", headers=auth_headers)
        assert logout_all_resp.status_code == 204

        # Both tokens must now be invalid
        assert client.post(
            "/api/v1/auth/refresh", json={"refresh_token": first_refresh}
        ).status_code == 401

        assert client.post(
            "/api/v1/auth/refresh", json={"refresh_token": second_refresh}
        ).status_code == 401
