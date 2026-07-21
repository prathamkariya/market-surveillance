"""app/routers/market_data.py — OHLCV market data endpoints."""
from typing import List

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.database import get_db
from app.dependencies import get_current_user
from app.models import MarketData, User
from app.schemas import MarketDataCreate, MarketDataResponse

router = APIRouter(prefix="/market-data", tags=["market-data"])


@router.post("", response_model=MarketDataResponse, status_code=status.HTTP_201_CREATED)
def create_market_data(
    payload: MarketDataCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Ingest one OHLCV candle for the authenticated user.

    NOTE: MarketData's uniqueness is enforced per-user via the
    uq_market_data_user_symbol_timestamp constraint. Collisions
    return a 409 Conflict.
    """
    def infer_market(symbol: str) -> str:
        # Crypto: typically ends in USDT, BTC, ETH, or contains a hyphen/slash
        if any(suffix in symbol.upper() for suffix in ["USDT", "BTC", "ETH", "-", "/"]):
            return "CRYPTO"
        # Fallback/Equities: shorter alpha symbols
        return "US_EQUITY"

    record = MarketData(
        user_id=current_user.id,
        symbol=payload.symbol,
        timestamp=payload.timestamp,
        open=payload.open,
        high=payload.high,
        low=payload.low,
        close=payload.close,
        volume=payload.volume,
        market=infer_market(payload.symbol),
    )
    db.add(record)
    try:
        db.commit()
    except IntegrityError as e:
        db.rollback()
        if "uq_market_data_user_symbol_timestamp" in str(e.orig):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Market data for {payload.symbol} at {payload.timestamp} already exists.",
            )
        raise
    db.refresh(record)
    return record


@router.get("", response_model=List[MarketDataResponse])
def list_market_data(
    symbol: str | None = None,
    limit: int = 100,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """List OHLCV records for the authenticated user, optionally filtered by symbol."""
    query = db.query(MarketData).filter(MarketData.user_id == current_user.id)
    if symbol:
        query = query.filter(MarketData.symbol == symbol.upper())
    return query.order_by(MarketData.timestamp.desc()).limit(limit).all()


@router.get("/{record_id}", response_model=MarketDataResponse)
def get_market_data(
    record_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Fetch a single OHLCV record by ID."""
    record = db.query(MarketData).filter(
        MarketData.id == record_id,
        MarketData.user_id == current_user.id,
    ).first()
    if record is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Record not found")
    return record


@router.delete("/{record_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_market_data(
    record_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Delete a single OHLCV record."""
    record = db.query(MarketData).filter(
        MarketData.id == record_id,
        MarketData.user_id == current_user.id,
    ).first()
    if record is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Record not found")
    db.delete(record)
    db.commit()
    return None
