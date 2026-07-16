"""app/routers/alerts.py — Alert management endpoints."""
from typing import List

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.database import get_db
from app.dependencies import get_current_user
from app.models import Alert, Anomaly, User
from app.schemas import AlertCreate, AlertResponse, AlertUpdate

router = APIRouter(prefix="/alerts", tags=["alerts"])


@router.post("", response_model=AlertResponse, status_code=status.HTTP_201_CREATED)
def create_alert(
    payload: AlertCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Create an alert for an anomaly owned by the current user."""
    anomaly = db.query(Anomaly).filter(Anomaly.id == payload.anomaly_id).first()
    if anomaly is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Anomaly not found")

    alert = Alert(
        anomaly_id=payload.anomaly_id,
        user_id=current_user.id,
        message=payload.message,
    )
    db.add(alert)
    db.commit()
    db.refresh(alert)
    return alert


@router.get("", response_model=List[AlertResponse])
def list_alerts(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """List all alerts for the current user."""
    return db.query(Alert).filter(Alert.user_id == current_user.id).all()


@router.get("/{alert_id}", response_model=AlertResponse)
def get_alert(
    alert_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Fetch a single alert by ID."""
    alert = db.query(Alert).filter(
        Alert.id == alert_id,
        Alert.user_id == current_user.id,
    ).first()
    if alert is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Alert not found")
    return alert


@router.patch("/{alert_id}", response_model=AlertResponse)
def update_alert(
    alert_id: int,
    payload: AlertUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Update alert status or message."""
    alert = db.query(Alert).filter(
        Alert.id == alert_id,
        Alert.user_id == current_user.id,
    ).first()
    if alert is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Alert not found")

    if payload.status is not None:
        alert.status = payload.status
    if payload.message is not None:
        alert.message = payload.message

    db.commit()
    db.refresh(alert)
    return alert


@router.delete("/{alert_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_alert(
    alert_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Delete an alert."""
    alert = db.query(Alert).filter(
        Alert.id == alert_id,
        Alert.user_id == current_user.id,
    ).first()
    if alert is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Alert not found")
    db.delete(alert)
    db.commit()
    return None


# ──────────────────────────────────────────────
# Streaming Endpoint (Phase 8)
# ──────────────────────────────────────────────
from fastapi.responses import StreamingResponse
import asyncio
import json

@router.get("/stream/live")
async def stream_live_alerts():
    """
    Server-Sent Events (SSE) endpoint for live anomalies.
    The frontend UI connects here to receive real-time push notifications 
    when run_engine.py detects a threat.
    """
    from app.services.redis_service import get_async_redis, STREAM_ALERTS

    async def event_generator():
        client = get_async_redis()
        last_id = "$"
        while True:
            try:
                results = await client.xread({STREAM_ALERTS: last_id}, count=10, block=2000)
                if results:
                    for _stream_name, entries in results:
                        for entry_id, fields in entries:
                            last_id = entry_id
                            yield f"data: {fields['data']}\n\n"
            except asyncio.CancelledError:
                break
            except Exception as e:
                # Silently wait on Redis failure, don't break the SSE connection
                await asyncio.sleep(2)

    return StreamingResponse(event_generator(), media_type="text/event-stream")
