"""app/routers/anomaly.py — Anomaly detection endpoints."""
from fastapi import APIRouter, Depends, status
from sqlalchemy.orm import Session

from app.database import get_db
from app.dependencies import get_current_user
from app.models import User
from app.schemas import AnomalyDetectRequest, AnomalyResponse
from app.services.anomaly_service import detect_anomaly

router = APIRouter(prefix="/anomalies", tags=["anomalies"])


@router.post("", response_model=AnomalyResponse, status_code=status.HTTP_201_CREATED)
def run_anomaly_detection(
    payload: AnomalyDetectRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Run anomaly detection on an existing market data record.
    Returns the anomaly result including the composite score.
    """
    return detect_anomaly(
        db=db,
        market_data_id=payload.market_data_id,
        user_id=current_user.id,
        threshold=payload.threshold,
    )
