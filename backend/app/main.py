from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from app.database import get_db
from slowapi.errors import RateLimitExceeded
from slowapi import _rate_limit_exceeded_handler
from app.limiter import limiter

from app.config import settings
from app.routers import alerts, anomaly, auth, market_data, watchlists, reports

app = FastAPI(
    title="Market Surveillance & Anomaly Detection",
    description="Production-grade API for detecting market manipulation patterns.",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# ──────────────────────────────────────────────
# Middleware
# ──────────────────────────────────────────────
# B8: allow_origins=["*"] + allow_credentials=True is invalid per the CORS spec
# (browsers reject it). Use the explicit origin list from settings instead.
_allowed_origins = [o.strip() for o in settings.ALLOWED_ORIGINS.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ──────────────────────────────────────────────
# Routers
# ──────────────────────────────────────────────
API_PREFIX = "/api/v1"

app.include_router(auth.router, prefix=API_PREFIX)
app.include_router(market_data.router, prefix=API_PREFIX)
app.include_router(anomaly.router, prefix=API_PREFIX)
app.include_router(alerts.router, prefix=API_PREFIX)
app.include_router(watchlists.router, prefix=API_PREFIX)   # Phase 2
app.include_router(reports.router, prefix=API_PREFIX)

# ──────────────────────────────────────────────
# Health check
# ──────────────────────────────────────────────
@app.get("/health", tags=["health"])
def health_check(db: Session = Depends(get_db)):
    """Live health check — pings the database so it reflects actual system state.
    Returns 200 ok when the DB is reachable, 503 degraded when it isn't.
    A static dict here would report 'ok' through a full DB outage.
    """
    from sqlalchemy import text
    try:
        db.execute(text("SELECT 1"))
        return {"status": "ok", "version": "2.0.0"}
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={"status": "degraded", "version": "2.0.0", "detail": str(e)},
        )
