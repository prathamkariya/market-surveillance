"""app/services/mar_generator.py

Phase 9: AI Auto-MAR Report Generation
Fetches the anomaly context from the database and uses Gemini to generate a 
Market Abuse Report (MAR) explaining the threat, the features, and the severity.
"""
import json
import logging
import os

from fastapi import HTTPException, status
from google import genai
from google.genai import types

logger = logging.getLogger(__name__)


def generate_mar(context_data: dict) -> str:
    """
    Fetch the alert context, and ask Gemini 1.5 Flash to generate a
    suspicious activity report.
    Returns Markdown text.

    Args:
        context_data: A dictionary containing the necessary DB properties.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="GEMINI_API_KEY not configured in .env",
        )

    # 2. Prepare Context Prompt
    context = f"""
    You are an expert financial compliance officer. Please generate a Market Abuse Report (MAR)
    for the following detected anomaly on {context_data['md_symbol']} at {context_data['md_timestamp']}.
    
    Alert ID: {context_data['anomaly_id']}
    Anomaly Score: {context_data['anomaly_score']} (Threshold: 0.7)
    Isolation Forest Unsupervised Score: {context_data['anomaly_if']}
    Random Forest Supervised Score: {context_data['anomaly_rf']}
    
    Price: {context_data['md_close']}
    Volume: {context_data['md_volume']}
    
    Features computed at the time of anomaly:
    {json.dumps(json.loads(context_data['anomaly_features']), indent=2) if context_data.get('anomaly_features') else 'N/A'}
    
    Please structure the report with:
    1. Executive Summary
    2. Event Details
    3. Technical ML Breakdown (explain the Isolation Forest vs Random Forest scores)
    4. Compliance Action Recommended
    
    Use clear Markdown format. Make it look professional.
    """
    
    # 3. Call Gemini
    try:
        client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(timeout=30000)
        )
        try:
            model_name = os.getenv("GEMINI_MODEL", "").strip() or "gemini-3.5-flash"
            response = client.models.generate_content(
                model=model_name,
                contents=context,
            )
            return response.text
        finally:
            client.close()
    except Exception as e:
        logger.error(f"Gemini API error: {e}")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Failed to generate report: {e}",
        )
