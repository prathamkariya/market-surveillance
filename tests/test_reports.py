"""tests/test_reports.py — Tests for the MAR report generator endpoints."""
import pytest
from unittest.mock import patch, MagicMock

# ══════════════════════════════════════════════════════════════
# MAR REPORT GENERATION (Phase 9 - Tests for IDOR & Edge cases)
# ══════════════════════════════════════════════════════════════
class TestMarReports:
    @patch("app.services.mar_generator.genai.GenerativeModel")
    def test_mar_report_missing_market_data_returns_404(self, mock_model, client, auth_headers, db_session):
        from app.models import Alert, Anomaly, MarketData, User
        db = db_session
        user = db.query(User).filter(User.email == "test@example.com").first()
        
        md = MarketData(
            user_id=user.id, symbol="DELME", timestamp="2022-01-01T12:00:00Z",
            open=10.0, high=10.0, low=10.0, close=10.0, volume=10.0, market="CRYPTO"
        )
        db.add(md)
        db.commit()
        db.refresh(md)
        
        anom = Anomaly(market_data_id=md.id, anomaly_score=0.99)
        db.add(anom)
        db.commit()
        db.refresh(anom)
        
        alert = Alert(anomaly_id=anom.id, user_id=user.id, message="Test")
        db.add(alert)
        db.commit()
        db.refresh(alert)
        alert_id = alert.id
        
        # Now delete the market data!
        db.delete(md)
        db.commit()
        
        # 2. Try to generate report
        response = client.get(f"/api/v1/reports/mar/{anom.id}", headers=auth_headers)
        assert response.status_code == 404
        detail = response.json()["detail"].lower()
        assert "anomaly not found" in detail or "market data record" in detail

    @patch("app.services.mar_generator.genai.GenerativeModel")
    def test_mar_report_idor_blocked(self, mock_model, client, auth_headers, db_session):
        from app.models import Alert, Anomaly, MarketData, User
        db = db_session
        user_a = db.query(User).filter(User.email == "test@example.com").first()
        
        # Create user B to test IDOR
        import uuid
        hacker_id = uuid.uuid4().hex[:6]
        hacker_email = f"hacker_{hacker_id}@example.com"
        hacker_username = f"hacker_{hacker_id}"
        client.post("/api/v1/auth/register", json={"email": hacker_email, "username": hacker_username, "password": "SecurePass1"})
        resp = client.post("/api/v1/auth/login", json={"email": hacker_email, "password": "SecurePass1"})
        other_headers = {"Authorization": f"Bearer {resp.json()['access_token']}"}
        
        md = MarketData(
            user_id=user_a.id, symbol="IDORTEST", timestamp="2022-01-01T12:00:00Z",
            open=10.0, high=10.0, low=10.0, close=10.0, volume=10.0, market="CRYPTO"
        )
        db.add(md)
        db.commit()
        
        anom = Anomaly(market_data_id=md.id, anomaly_score=0.99)
        db.add(anom)
        db.commit()
        
        alert = Alert(anomaly_id=anom.id, user_id=user_a.id, message="Test IDOR")
        db.add(alert)
        db.commit()
        
        response = client.get(f"/api/v1/reports/mar/{anom.id}", headers=other_headers)
        assert response.status_code == 403
        assert "permission to access this report" in response.json()["detail"].lower()

    @patch("app.services.mar_generator.genai.GenerativeModel")
    def test_mar_report_works_without_alert(self, mock_model, client, auth_headers, db_session):
        """
        An Anomaly can exist without an Alert -- POST /alerts is a separate,
        optional user action, not something that happens automatically when
        an anomaly is detected. The MAR endpoint's route param is `alert_id`
        but must resolve against Anomaly.id (matching its pre-B4 behavior),
        not Alert.id -- otherwise every un-alerted anomaly is unreportable.
        """
        from app.models import Anomaly, MarketData, User

        mock_instance = MagicMock()
        mock_instance.generate_content.return_value = MagicMock(text="# Mock MAR Report")
        mock_model.return_value = mock_instance

        db = db_session
        user = db.query(User).filter(User.email == "test@example.com").first()

        md = MarketData(
            user_id=user.id, symbol="NOALERT", timestamp="2022-01-01T12:00:00Z",
            open=10.0, high=10.0, low=10.0, close=10.0, volume=10.0, market="CRYPTO"
        )
        db.add(md)
        db.commit()
        db.refresh(md)

        anom = Anomaly(market_data_id=md.id, anomaly_score=0.95)
        db.add(anom)
        db.commit()
        db.refresh(anom)
        # Deliberately no Alert row created here.

        response = client.get(f"/api/v1/reports/mar/{anom.id}", headers=auth_headers)
        assert response.status_code == 200, (
            f"expected 200, got {response.status_code}: {response.text} -- "
            "if this 404s with 'Alert not found', the endpoint is querying "
            "Alert.id instead of Anomaly.id"
        )
