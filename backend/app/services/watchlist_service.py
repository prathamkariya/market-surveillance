from typing import List, Optional

from fastapi import HTTPException, status
from sqlalchemy.orm import Session, selectinload

from app.models import Watchlist, WatchlistSymbol
from app.schemas import WatchlistCreate, WatchlistSymbolAdd, WatchlistUpdate


# ──────────────────────────────────────────────
# Private helpers
# ──────────────────────────────────────────────
def _get_watchlist_or_404(db: Session, watchlist_id: int, user_id: int) -> Watchlist:
    """
    Fetch a watchlist by ID and verify ownership.
    Raises 404 if not found — deliberately same error for not-found and wrong-user
    (don't reveal that the resource exists but belongs to someone else).
    """
    wl = (
        db.query(Watchlist)
        .options(selectinload(Watchlist.symbols))
        .filter(Watchlist.id == watchlist_id, Watchlist.user_id == user_id)
        .first()
    )
    if wl is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Watchlist not found")
    return wl


# ──────────────────────────────────────────────
# Watchlist CRUD
# ──────────────────────────────────────────────
def create_watchlist(db: Session, user_id: int, payload: WatchlistCreate) -> Watchlist:
    """
    Create a new watchlist for a user.
    Returns 409 if the user already has a watchlist with this name.
    """
    existing = db.query(Watchlist).filter(
        Watchlist.user_id == user_id,
        Watchlist.name == payload.name,
    ).first()
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Watchlist named '{payload.name}' already exists",
        )

    wl = Watchlist(
        user_id=user_id,
        name=payload.name,
        description=payload.description,
    )
    db.add(wl)
    db.commit()
    db.refresh(wl)
    return wl


def get_watchlist(db: Session, watchlist_id: int, user_id: int) -> Watchlist:
    """Get a single watchlist with all its symbols."""
    return _get_watchlist_or_404(db, watchlist_id, user_id)


def list_watchlists(db: Session, user_id: int) -> List[Watchlist]:
    """
    List all watchlists for a user.
    Uses selectinload to avoid N+1 when rendering symbol counts.
    """
    return (
        db.query(Watchlist)
        .options(selectinload(Watchlist.symbols))
        .filter(Watchlist.user_id == user_id)
        .order_by(Watchlist.created_at.desc())
        .all()
    )


def update_watchlist(
    db: Session,
    watchlist_id: int,
    user_id: int,
    payload: WatchlistUpdate,
) -> Watchlist:
    """
    Update watchlist name and/or description.
    Returns 409 if the new name conflicts with another watchlist.
    """
    wl = _get_watchlist_or_404(db, watchlist_id, user_id)

    if payload.name is not None and payload.name != wl.name:
        # Check name uniqueness for this user
        conflict = db.query(Watchlist).filter(
            Watchlist.user_id == user_id,
            Watchlist.name == payload.name,
            Watchlist.id != watchlist_id,
        ).first()
        if conflict:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Watchlist named '{payload.name}' already exists",
            )
        wl.name = payload.name

    if payload.description is not None:
        wl.description = payload.description

    db.commit()
    db.refresh(wl)
    return wl


def delete_watchlist(db: Session, watchlist_id: int, user_id: int) -> None:
    """Delete a watchlist and all its symbols (cascade)."""
    wl = _get_watchlist_or_404(db, watchlist_id, user_id)
    db.delete(wl)
    db.commit()


# ──────────────────────────────────────────────
# Symbol management inside a watchlist
# ──────────────────────────────────────────────
def add_symbol(
    db: Session,
    watchlist_id: int,
    user_id: int,
    payload: WatchlistSymbolAdd,
) -> WatchlistSymbol:
    """
    Add a symbol to a watchlist.
    Returns 409 if symbol already in this watchlist.
    """
    wl = _get_watchlist_or_404(db, watchlist_id, user_id)

    existing = db.query(WatchlistSymbol).filter(
        WatchlistSymbol.watchlist_id == wl.id,
        WatchlistSymbol.symbol == payload.symbol,
    ).first()
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Symbol '{payload.symbol}' already in watchlist",
        )

    ws = WatchlistSymbol(
        watchlist_id=wl.id,
        symbol=payload.symbol,
        notes=payload.notes,
    )
    db.add(ws)
    db.commit()
    db.refresh(ws)
    return ws


def remove_symbol(
    db: Session,
    watchlist_id: int,
    symbol: str,
    user_id: int,
) -> None:
    """Remove a symbol from a watchlist."""
    wl = _get_watchlist_or_404(db, watchlist_id, user_id)

    ws = db.query(WatchlistSymbol).filter(
        WatchlistSymbol.watchlist_id == wl.id,
        WatchlistSymbol.symbol == symbol.upper(),
    ).first()
    if ws is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Symbol '{symbol}' not in watchlist",
        )

    db.delete(ws)
    db.commit()
