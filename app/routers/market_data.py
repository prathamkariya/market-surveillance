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

    NOTE: MarketData's (symbol, timestamp) uniqueness is currently enforced
    globally, not per-user (see uq_market_data_symbol_timestamp in
    001_initial_schema.py) -- so two different users ingesting the same
    symbol+timestamp collide here even though every read path in this
    service treats MarketData as owned per-user. Catching the conflict and
    returning 409 stops it from surfacing as an unhandled 500; it does not
    by itself resolve which of the two users' OHLCV values should be
    considered authoritative. If MarketData is meant to be genuinely
    per-user, the real fix is scoping the constraint to
    (user_id, symbol, timestamp) instead.
    """
    record = MarketData(
        user_id=current_user.id,
        symbol=payload.symbol,
        timestamp=payload.timestamp,
        open=payload.open,
        high=payload.high,
        low=payload.low,
        close=payload.close,
        volume=payload.volume,
    )
    db.add(record)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Market data for {payload.symbol} at {payload.timestamp} already exists.",
        )
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
