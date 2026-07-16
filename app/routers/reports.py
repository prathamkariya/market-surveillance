"""app/routers/reports.py"""
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from fastapi.responses import PlainTextResponse

from app.database import get_db
from app.models import User
from app.dependencies import get_current_user
from app.services.mar_generator import generate_mar

router = APIRouter(prefix="/reports", tags=["Reports"])

@router.get("/mar/{alert_id}")
def get_mar_report(
    alert_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Generate an AI-driven Market Abuse Report (MAR) for a specific alert.
    Returns markdown content.
    """
    try:
        report_md = generate_mar(db, alert_id)
        # Return as markdown file download
        return PlainTextResponse(
            content=report_md,
            media_type="text/markdown",
            headers={"Content-Disposition": f"attachment; filename=MAR_Alert_{alert_id}.md"}
        )
    except Exception as e:
        if isinstance(e, HTTPException):
            raise
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
