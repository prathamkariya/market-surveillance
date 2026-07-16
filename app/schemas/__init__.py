from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, EmailStr, Field, field_validator, model_validator, ConfigDict


# ══════════════════════════════════════════════════════════════
# SHARED UTILITIES
# ══════════════════════════════════════════════════════════════
class OrmBase(BaseModel):
    """Base for all response schemas. Enables ORM mode (from_orm)."""
    model_config = ConfigDict(from_attributes=True)


# ══════════════════════════════════════════════════════════════
# AUTH SCHEMAS
# ══════════════════════════════════════════════════════════════
class UserRegister(BaseModel):
    email: EmailStr
    username: str = Field(..., min_length=3, max_length=50, pattern=r"^[a-zA-Z0-9_]+$")
    password: str = Field(..., min_length=8, max_length=128)

    @field_validator("password")
    @classmethod
    def password_strength(cls, v: str) -> str:
        if not any(c.isdigit() for c in v):
            raise ValueError("Password must contain at least one digit")
        if not any(c.isupper() for c in v):
            raise ValueError("Password must contain at least one uppercase letter")
        return v


class UserLogin(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=1)


class UserResponse(OrmBase):
    id: int
    email: str
    username: str
    is_active: bool
    created_at: datetime


class TokenResponse(BaseModel):
    """Response from /auth/login — includes both token types."""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int   # Access token TTL in seconds


class AccessTokenResponse(BaseModel):
    """Response from /auth/refresh — access token only."""
    access_token: str
    token_type: str = "bearer"
    expires_in: int


class RefreshRequest(BaseModel):
    """Body for POST /auth/refresh."""
    refresh_token: str = Field(..., min_length=1)


class LogoutRequest(BaseModel):
    """Body for POST /auth/logout — revokes a specific refresh token."""
    refresh_token: str = Field(..., min_length=1)


# ══════════════════════════════════════════════════════════════
# MARKET DATA SCHEMAS
# ══════════════════════════════════════════════════════════════
class MarketDataCreate(BaseModel):
    symbol: str = Field(..., min_length=1, max_length=20)
    timestamp: datetime
    open: float = Field(..., gt=0)
    high: float = Field(..., gt=0)
    low: float = Field(..., gt=0)
    close: float = Field(..., gt=0)
    volume: float = Field(..., ge=0)

    @field_validator("symbol")
    @classmethod
    def symbol_uppercase(cls, v: str) -> str:
        return v.upper().strip()

    @model_validator(mode="after")
    def high_gte_low(self) -> "MarketDataCreate":
        if self.high < self.low:
            raise ValueError("high must be >= low")
        return self


class MarketDataResponse(OrmBase):
    id: int
    user_id: int
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    created_at: datetime


# ══════════════════════════════════════════════════════════════
# ANOMALY SCHEMAS
# ══════════════════════════════════════════════════════════════
class AnomalyDetectRequest(BaseModel):
    market_data_id: int = Field(..., gt=0)
    threshold: float = Field(default=0.7, ge=0.0, le=1.0)


class AnomalyResponse(OrmBase):
    id: int
    market_data_id: int
    anomaly_score: float
    is_anomaly: bool
    isolation_forest_score: Optional[float]
    multi_pattern_max_score: Optional[float]
    pattern_scores: Optional[str]
    model_version: Optional[str]
    features: Optional[str]
    detected_at: datetime


# ══════════════════════════════════════════════════════════════
# ALERT SCHEMAS
# ══════════════════════════════════════════════════════════════
class AlertCreate(BaseModel):
    anomaly_id: int = Field(..., gt=0)
    message: Optional[str] = Field(None, max_length=1000)


class AlertUpdate(BaseModel):
    status: Optional[str] = Field(None, pattern=r"^(PENDING|ACTIVE|RESOLVED|DISMISSED)$")
    message: Optional[str] = Field(None, max_length=1000)


class AlertResponse(OrmBase):
    id: int
    anomaly_id: int
    user_id: int
    status: str
    message: Optional[str]
    created_at: datetime
    updated_at: datetime


# ══════════════════════════════════════════════════════════════
# WATCHLIST SCHEMAS  (Phase 2 — NEW)
# ══════════════════════════════════════════════════════════════
class WatchlistCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    description: Optional[str] = Field(None, max_length=500)


class WatchlistUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=100)
    description: Optional[str] = Field(None, max_length=500)


class WatchlistSymbolAdd(BaseModel):
    symbol: str = Field(..., min_length=1, max_length=20)
    notes: Optional[str] = Field(None, max_length=500)

    @field_validator("symbol")
    @classmethod
    def symbol_uppercase(cls, v: str) -> str:
        return v.upper().strip()


class WatchlistSymbolResponse(OrmBase):
    id: int
    watchlist_id: int
    symbol: str
    notes: Optional[str]
    added_at: datetime


class WatchlistResponse(OrmBase):
    id: int
    user_id: int
    name: str
    description: Optional[str]
    symbols: List[WatchlistSymbolResponse] = []
    created_at: datetime
    updated_at: datetime


class WatchlistListResponse(OrmBase):
    """Lightweight list view — no symbol details."""
    id: int
    user_id: int
    name: str
    description: Optional[str]
    symbol_count: int = 0
    created_at: datetime
