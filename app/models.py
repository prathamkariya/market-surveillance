from datetime import datetime, timezone
from sqlalchemy import (
    Boolean, Column, DateTime, Float, ForeignKey,
    Index, Integer, Numeric, String, Text, UniqueConstraint,
)
from sqlalchemy.orm import relationship

from app.database import Base


# ══════════════════════════════════════════════════════════════
# USER
# ══════════════════════════════════════════════════════════════
class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, autoincrement=True)
    email = Column(String(255), nullable=False, unique=True, index=True)
    username = Column(String(50), nullable=False, unique=True, index=True)
    hashed_password = Column(String(255), nullable=False)
    is_active = Column(Boolean, nullable=False, default=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )

    # Relationships
    market_data = relationship("MarketData", back_populates="user", cascade="all, delete-orphan")
    alerts = relationship("Alert", back_populates="user", cascade="all, delete-orphan")
    watchlists = relationship("Watchlist", back_populates="user", cascade="all, delete-orphan")
    refresh_tokens = relationship("RefreshToken", back_populates="user", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<User(id={self.id}, email='{self.email}')>"


# ══════════════════════════════════════════════════════════════
# MARKET DATA  (OHLCV candles)
# ══════════════════════════════════════════════════════════════
class MarketData(Base):
    __tablename__ = "market_data"
    __table_args__ = (
        UniqueConstraint("symbol", "timestamp", name="uq_market_data_symbol_timestamp"),
        Index("ix_market_data_symbol_timestamp", "symbol", "timestamp"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(
        Integer, ForeignKey("users.id", ondelete="CASCADE", name="fk_market_data_user_id"),
        nullable=False, index=True,
    )
    symbol = Column(String(20), nullable=False, index=True)
    timestamp = Column(DateTime(timezone=True), nullable=False)

    # OHLCV — Numeric(15,6) for exact decimal arithmetic (never FLOAT for prices)
    open = Column(Numeric(15, 6), nullable=False)
    high = Column(Numeric(15, 6), nullable=False)
    low = Column(Numeric(15, 6), nullable=False)
    close = Column(Numeric(15, 6), nullable=False)
    volume = Column(Numeric(20, 2), nullable=False)
    # Market classification stamped at ingestion time (e.g. "CRYPTO", "US_EQUITY").
    # Nullable for backward compat with records created before this column was added.
    # detect_anomaly() raises 400 if this is None — callers must re-submit old records
    # via POST /market-data with market= set rather than silently routing to a wrong model.
    market = Column(String(20), nullable=True, index=True)

    created_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))

    # Relationships
    user = relationship("User", back_populates="market_data")
    anomalies = relationship("Anomaly", back_populates="market_data", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<MarketData(symbol='{self.symbol}', timestamp='{self.timestamp}')>"


# ══════════════════════════════════════════════════════════════
# ANOMALY
# ══════════════════════════════════════════════════════════════
class Anomaly(Base):
    __tablename__ = "anomalies"

    id = Column(Integer, primary_key=True, autoincrement=True)
    market_data_id = Column(
        Integer,
        ForeignKey("market_data.id", ondelete="CASCADE", name="fk_anomalies_market_data_id"),
        nullable=False,
        index=True,
    )
    anomaly_score = Column(Float, nullable=False)
    is_anomaly = Column(Boolean, nullable=False, default=False)
    isolation_forest_score = Column(Float, nullable=True)
    multi_pattern_max_score = Column(Float, nullable=True)  # max per-pattern probability; see pattern_scores for the full breakdown
    pattern_scores = Column(Text, nullable=True)   # JSON string: {"pump_and_dump": 0.02, "wash_trading": 0.87, ...}
    model_version = Column(String(255), nullable=True)  # trained_at_utc of the model(s) that produced this score, for provenance
    features = Column(Text, nullable=True)   # JSON string of feature values
    detected_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))

    # Relationships
    market_data = relationship("MarketData", back_populates="anomalies")
    alerts = relationship("Alert", back_populates="anomaly", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<Anomaly(id={self.id}, score={self.anomaly_score:.3f})>"


# ══════════════════════════════════════════════════════════════
# ALERT
# ══════════════════════════════════════════════════════════════
class AlertStatus:
    PENDING = "PENDING"
    ACTIVE = "ACTIVE"
    RESOLVED = "RESOLVED"
    DISMISSED = "DISMISSED"


class Alert(Base):
    __tablename__ = "alerts"

    id = Column(Integer, primary_key=True, autoincrement=True)
    anomaly_id = Column(
        Integer,
        ForeignKey("anomalies.id", ondelete="CASCADE", name="fk_alerts_anomaly_id"),
        nullable=False,
        index=True,
    )
    user_id = Column(
        Integer,
        ForeignKey("users.id", ondelete="CASCADE", name="fk_alerts_user_id"),
        nullable=False,
        index=True,
    )
    status = Column(String(20), nullable=False, default=AlertStatus.PENDING)
    message = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )

    # Relationships
    anomaly = relationship("Anomaly", back_populates="alerts")
    user = relationship("User", back_populates="alerts")

    def __repr__(self) -> str:
        return f"<Alert(id={self.id}, status='{self.status}')>"


# ══════════════════════════════════════════════════════════════
# WATCHLIST  (Phase 2 — NEW)
# ══════════════════════════════════════════════════════════════
class Watchlist(Base):
    """
    A named collection of symbols that a user wants to monitor.
    One user can have many watchlists (e.g., "Tech Stocks", "Crypto").
    """
    __tablename__ = "watchlists"
    __table_args__ = (
        UniqueConstraint("user_id", "name", name="uq_watchlists_user_name"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(
        Integer,
        ForeignKey("users.id", ondelete="CASCADE", name="fk_watchlists_user_id"),
        nullable=False,
        index=True,
    )
    name = Column(String(100), nullable=False)
    description = Column(String(500), nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )

    # Relationships
    user = relationship("User", back_populates="watchlists")
    symbols = relationship(
        "WatchlistSymbol",
        back_populates="watchlist",
        cascade="all, delete-orphan",
        lazy="selectin",    # Always eager-load symbols with the watchlist
    )

    def __repr__(self) -> str:
        return f"<Watchlist(id={self.id}, name='{self.name}', user_id={self.user_id})>"


class WatchlistSymbol(Base):
    """
    A single symbol entry inside a watchlist.
    Composite unique: same symbol cannot appear twice in the same watchlist.
    """
    __tablename__ = "watchlist_symbols"
    __table_args__ = (
        UniqueConstraint(
            "watchlist_id", "symbol",
            name="uq_watchlist_symbols_watchlist_symbol",
        ),
        Index("ix_watchlist_symbols_watchlist_id", "watchlist_id"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    watchlist_id = Column(
        Integer,
        ForeignKey("watchlists.id", ondelete="CASCADE", name="fk_watchlist_symbols_watchlist_id"),
        nullable=False,
    )
    symbol = Column(String(20), nullable=False)
    notes = Column(String(500), nullable=True)
    added_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))

    # Relationships
    watchlist = relationship("Watchlist", back_populates="symbols")

    def __repr__(self) -> str:
        return f"<WatchlistSymbol(symbol='{self.symbol}', watchlist_id={self.watchlist_id})>"


# ══════════════════════════════════════════════════════════════
# REFRESH TOKEN  (Phase 2 — NEW)
# ══════════════════════════════════════════════════════════════
class RefreshToken(Base):
    """
    Stores hashed refresh tokens for rotating JWT auth.
    The raw token is never stored — only bcrypt hash.
    One user can have multiple refresh tokens (different devices/sessions).
    """
    __tablename__ = "refresh_tokens"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(
        Integer,
        ForeignKey("users.id", ondelete="CASCADE", name="fk_refresh_tokens_user_id"),
        nullable=False,
        index=True,
    )
    # Store a hash of the token, not the raw token
    token_hash = Column(String(255), nullable=False, unique=True, index=True)
    expires_at = Column(DateTime(timezone=True), nullable=False)
    revoked = Column(Boolean, nullable=False, default=False)
    created_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))

    # Relationships
    user = relationship("User", back_populates="refresh_tokens")

    @property
    def is_expired(self) -> bool:
        from datetime import timezone
        return datetime.now(timezone.utc) > self.expires_at

    @property
    def is_valid(self) -> bool:
        return not self.revoked and not self.is_expired

    def __repr__(self) -> str:
        return f"<RefreshToken(id={self.id}, user_id={self.user_id}, revoked={self.revoked})>"
