from typing import List

from fastapi import APIRouter, Depends, status
from sqlalchemy.orm import Session

from app.database import get_db
from app.dependencies import get_current_user
from app.models import User
from app.schemas import (
    WatchlistCreate,
    WatchlistListResponse,
    WatchlistResponse,
    WatchlistSymbolAdd,
    WatchlistSymbolResponse,
    WatchlistUpdate,
)
from app.services import watchlist_service

router = APIRouter(prefix="/watchlists", tags=["watchlists"])


@router.post("", response_model=WatchlistResponse, status_code=status.HTTP_201_CREATED)
def create_watchlist(
    payload: WatchlistCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Create a new named watchlist for the authenticated user."""
    return watchlist_service.create_watchlist(db, current_user.id, payload)


@router.get("", response_model=List[WatchlistListResponse])
def list_watchlists(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """List all watchlists belonging to the authenticated user."""
    watchlists = watchlist_service.list_watchlists(db, current_user.id)
    # Build lightweight list response with symbol count
    result = []
    for wl in watchlists:
        resp = WatchlistListResponse(
            id=wl.id,
            user_id=wl.user_id,
            name=wl.name,
            description=wl.description,
            symbol_count=len(wl.symbols),
            created_at=wl.created_at,
        )
        result.append(resp)
    return result


@router.get("/{watchlist_id}", response_model=WatchlistResponse)
def get_watchlist(
    watchlist_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Get a single watchlist with all symbols. 404 if not found or wrong user."""
    return watchlist_service.get_watchlist(db, watchlist_id, current_user.id)


@router.put("/{watchlist_id}", response_model=WatchlistResponse)
def update_watchlist(
    watchlist_id: int,
    payload: WatchlistUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Update watchlist name or description."""
    return watchlist_service.update_watchlist(db, watchlist_id, current_user.id, payload)


@router.delete("/{watchlist_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_watchlist(
    watchlist_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Delete a watchlist and all its symbols."""
    watchlist_service.delete_watchlist(db, watchlist_id, current_user.id)


# ──────────────────────────────────────────────
# Symbol sub-resource
# ──────────────────────────────────────────────
@router.post(
    "/{watchlist_id}/symbols",
    response_model=WatchlistSymbolResponse,
    status_code=status.HTTP_201_CREATED,
)
def add_symbol(
    watchlist_id: int,
    payload: WatchlistSymbolAdd,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Add a symbol to a watchlist. 409 if symbol already present."""
    return watchlist_service.add_symbol(db, watchlist_id, current_user.id, payload)


@router.delete(
    "/{watchlist_id}/symbols/{symbol}",
    status_code=status.HTTP_204_NO_CONTENT,
)
def remove_symbol(
    watchlist_id: int,
    symbol: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Remove a symbol from a watchlist. 404 if symbol not in watchlist."""
    watchlist_service.remove_symbol(db, watchlist_id, symbol, current_user.id)
