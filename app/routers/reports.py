"""app/routers/reports.py"""
import asyncio

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from fastapi.responses import PlainTextResponse

from app.database import get_db
from app.models import User
from app.dependencies import get_current_user
from app.services.mar_generator import generate_mar

router = APIRouter(prefix="/reports", tags=["Reports"])

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
    try:
        # B6: offload blocking Gemini call to thread; cap at 30s to avoid exhausting workers
        report_md = await asyncio.wait_for(
            asyncio.to_thread(generate_mar, db, alert_id, current_user.id),
            timeout=30.0,
        )
        return PlainTextResponse(
            content=report_md,
            media_type="text/markdown",
            headers={"Content-Disposition": f"attachment; filename=MAR_Alert_{alert_id}.md"},
        )
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="Report generation timed out (Gemini API took >30s). Try again.",
        )
    except Exception as e:
        if isinstance(e, HTTPException):
            raise
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
