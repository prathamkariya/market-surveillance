"""app/services/mar_generator.py

Phase 9: AI Auto-MAR Report Generation
Fetches the anomaly context from the database and uses Gemini to generate a 
Market Abuse Report (MAR) explaining the threat, the features, and the severity.
"""
import json
import logging
import os

from sqlalchemy.orm import Session
from fastapi import HTTPException, status
import google.generativeai as genai

from app.models import Anomaly, MarketData

logger = logging.getLogger(__name__)


def generate_mar(db: Session, alert_id: int) -> str:
    """
    Fetch the alert context, and ask Gemini 1.5 Flash to generate a 
    suspicious activity report.
    Returns Markdown text.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="GEMINI_API_KEY not configured in .env",
        )
        
    # 1. Fetch data
    anomaly = db.query(Anomaly).filter(Anomaly.id == alert_id).first()
    if not anomaly:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Alert not found")
        
    md = db.query(MarketData).filter(MarketData.id == anomaly.market_data_id).first()
    
    # 2. Prepare Context Prompt
    context = f"""
    You are an expert financial compliance officer. Please generate a Market Abuse Report (MAR)
    for the following detected anomaly on {md.symbol} at {md.timestamp}.
    
    Alert ID: {anomaly.id}
    Anomaly Score: {anomaly.anomaly_score} (Threshold: 0.7)
    Isolation Forest Unsupervised Score: {anomaly.isolation_forest_score}
    Random Forest Supervised Score: {anomaly.multi_pattern_max_score}
    
    Price: {md.close}
    Volume: {md.volume}
    
    Features computed at the time of anomaly:
    {json.dumps(json.loads(anomaly.features), indent=2) if anomaly.features else 'N/A'}
    
    Please structure the report with:
    1. Executive Summary
    2. Event Details
    3. Technical ML Breakdown (explain the Isolation Forest vs Random Forest scores)
    4. Compliance Action Recommended
    
    Use clear Markdown format. Make it look professional.
    """
    
    # 3. Call Gemini
    try:
        genai.configure(api_key=api_key)
        # Using gemini-1.5-flash for speed
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(context)
        
        return response.text
    except Exception as e:
        logger.error(f"Gemini API error: {e}")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Failed to generate report: {e}",
        )
