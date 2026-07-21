import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session
from datetime import datetime, timezone
import json

from app.main import app
from app.database import Base, engine, get_db
from app.models import User, MarketData, Anomaly

# Create test tables if they don't exist
Base.metadata.create_all(bind=engine)

def get_test_db():
    from app.database import SessionLocal
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

app.dependency_overrides[get_db] = get_test_db
client = TestClient(app)

def setup_test_data():
    db = next(get_test_db())
    # clear old
    db.query(Anomaly).delete()
    db.query(MarketData).delete()
    db.query(User).filter(User.email.in_(["sys@test.com", "user@test.com", "other@test.com", "system@marketsurveillance.local"])).delete()
    db.commit()

    sys_user = User(email="system@marketsurveillance.local", username="sys_test", hashed_password="pw")
    user1 = User(email="user@test.com", username="u1", hashed_password="pw")
    user2 = User(email="other@test.com", username="u2", hashed_password="pw")
    db.add_all([sys_user, user1, user2])
    db.commit()

    # User 1 data + anomaly
    md1 = MarketData(user_id=user1.id, symbol="BTC", timestamp=datetime.now(timezone.utc), open=1, high=2, low=1, close=2, volume=100)
    db.add(md1)
    db.commit()
    an1 = Anomaly(market_data_id=md1.id, anomaly_score=0.9, is_anomaly=True, detected_at=datetime.now(timezone.utc))
    db.add(an1)

    # User 2 data + anomaly (should be hidden from User 1)
    md2 = MarketData(user_id=user2.id, symbol="ETH", timestamp=datetime.now(timezone.utc), open=1, high=2, low=1, close=2, volume=100)
    db.add(md2)
    db.commit()
    an2 = Anomaly(market_data_id=md2.id, anomaly_score=0.8, is_anomaly=True, detected_at=datetime.now(timezone.utc))
    db.add(an2)

    # System data + anomaly (should be visible to everyone)
    md3 = MarketData(user_id=sys_user.id, symbol="SOL", timestamp=datetime.now(timezone.utc), open=1, high=2, low=1, close=2, volume=100)
    db.add(md3)
    db.commit()
    an3 = Anomaly(market_data_id=md3.id, anomaly_score=0.95, is_anomaly=True, detected_at=datetime.now(timezone.utc))
    db.add(an3)

    db.commit()
    return user1

if __name__ == "__main__":
    user1 = setup_test_data()
    # Mock authentication by passing a valid token. Or just override the get_current_user dependency for testing
    from app.dependencies import get_current_user
    app.dependency_overrides[get_current_user] = lambda: user1

    resp = client.get("/anomalies?limit=10")
    assert resp.status_code == 200, resp.text
    data = resp.json()
    symbols = {item["symbol"] for item in data["items"]}
    assert symbols == {"BTC", "SOL"}, symbols
