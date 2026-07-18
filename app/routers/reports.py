"""app/routers/reports.py"""
import asyncio

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from fastapi.responses import PlainTextResponse

from app.database import get_db
from app.models import User, Anomaly, MarketData
from app.dependencies import get_current_user
from app.services.mar_generator import generate_mar

router = APIRouter(prefix="/reports", tags=["Reports"])

# Limit concurrent Gemini generation requests to avoid exhausting workers/rate limits
mar_generation_semaphore = asyncio.BoundedSemaphore(5)

@router.get("/mar/{alert_id}")
async def get_mar_report(
    alert_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Generate an AI-driven Market Abuse Report (MAR) for a specific alert.
    Returns markdown content. Only accessible by the user who owns the alert (B4).
    The Gemini call is offloaded to a thread with a 30-second timeout (B6).
    """
    
    # 1. Fetch anomaly
    anomaly = db.query(Anomaly).filter(Anomaly.id == alert_id).first()
    if not anomaly:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Anomaly not found")

    # B4: Ownership check — any logged-in user could previously read any user's report
    md = db.query(MarketData).filter(MarketData.id == anomaly.market_data_id).first()
    if md is None:
        # B5: Parent MarketData was deleted — don't crash into md.symbol AttributeError
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Market data record associated with this alert no longer exists.",
        )
        
    system_user = db.query(User).filter(User.email == "system@marketsurveillance.local").first()
    system_user_id = system_user.id if system_user else None
    
    if md.user_id != current_user.id and md.user_id != system_user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have permission to access this report.",
        )
        
    # Extract necessary variables before closing the session
    context_data = {
        "md_symbol": md.symbol,
        "md_timestamp": md.timestamp,
        "md_close": md.close,
        "md_volume": md.volume,
        "anomaly_id": anomaly.id,
        "anomaly_score": anomaly.anomaly_score,
        "anomaly_if": anomaly.isolation_forest_score,
        "anomaly_rf": anomaly.multi_pattern_max_score,
        "anomaly_features": anomaly.features,
    }

    try:
        # Wait up to 10s to acquire the permit, rejecting if the queue is too long
        await asyncio.wait_for(mar_generation_semaphore.acquire(), timeout=10.0)
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Server is busy generating reports. Please try again later.",
        )

    try:
        # The thread enforces its own 30s timeout via the Gemini SDK,
        # ensuring the permit is held for the exact duration of the worker thread.
        report_md = await asyncio.to_thread(generate_mar, context_data)

        return PlainTextResponse(
            content=report_md,
            media_type="text/markdown",
            headers={"Content-Disposition": f"attachment; filename=MAR_Alert_{alert_id}.md"},
        )
    except Exception as e:
        if isinstance(e, HTTPException):
            raise
        import logging
        logging.error("Error generating MAR report: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An internal error occurred while generating the report."
        )
    finally:
        mar_generation_semaphore.release()
