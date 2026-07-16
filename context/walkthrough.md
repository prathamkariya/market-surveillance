
## Priority 2 — Fix the Known Architectural Issue

### 2.1 Scope the MarketData unique constraint to per-user
Currently documented in 3 places (`market_data.py` docstring, `test_integration_config.py`, `docs/audit_phase7.md`) as a known issue:

```python
UniqueConstraint("symbol", "timestamp")  # global — wrong
# should be:
UniqueConstraint("user_id", "symbol", "timestamp")  # per-user
```

This requires:
- A new Alembic migration `005_fix_market_data_unique_constraint.py`
- Updating `test_integration_config.py`'s `TestCrossUserMarketDataCollision` to expect `201` (not `409`) for the second user

### 2.2 Return new refresh token on `/auth/refresh`
There's an explicit `NOTE` in `auth.py` line 104:
```python
# NOTE: We return only the new access token here.
# The new refresh token from rotation is stored in DB but not returned
# in this simplified implementation. In production, return both.
```
The rotated refresh token is generated and saved but silently discarded. This means clients **can't stay logged in past the first token rotation**. Fix: return both tokens from `/auth/refresh`, update `AccessTokenResponse` → `TokenResponse`.

---

## Priority 3 — Observability & Production Readiness

### 3.1 Health check is shallow
`GET /health` returns `{"status": "ok"}` always, even when:
- The database is unreachable
- No models are loaded (anomaly detection would return 503)

A real health check should test both. Needed before any load balancer or uptime monitor.

### 3.2 No rate limiting is actually enforced
`config.py` has `RATE_LIMIT_PER_MINUTE: int = 60` but nothing in the codebase reads or uses it. It's a config value with no implementation. The auth endpoints (register, login) are wide open to brute force.

### 3.3 CORS is `allow_origins=["*"]` in debug mode
In `main.py`: `allow_origins=["*"] if settings.DEBUG else [...]`. Since `DEBUG=True` in `.env.example` and `config.py`, this is effectively always `"*"`. This needs a concrete allowed-origins list before any deployment.

### 3.4 Structured logging
Currently `logger.warning(...)` calls are plain strings. No request IDs, no correlation between a user action and the model load warning it triggered. For a surveillance system, this matters.

---

## Priority 4 — Missing Features (Gaps in the API)

### 4.1 No `GET /api/v1/anomalies` list endpoint
The anomaly router has only `POST` (detect). There's no way to retrieve past anomaly records for a user — you'd have to query them through the alerts. The `AnomalyResponse` schema is there, the model has the data, the endpoint just doesn't exist.

### 4.2 No pagination on list endpoints
`GET /market-data?limit=100` has a hardcoded max of 100 with no cursor/offset. At scale this breaks.

### 4.3 Watchlist endpoint is missing `GET /watchlists/{id}`
The watchlist router has list and create, but no single-item fetch by ID (or symbol-level operations beyond add).

### 4.4 No model reload endpoint
`anomaly_service.py` explicitly notes:
> *"A hot-reload endpoint... is a natural follow-up if models get retrained while this service is running"*

Right now, retraining models requires restarting the API process.

---

## Summary Table

| # | Task | Effort | Impact |
|---|---|---|---|
| 1.1 | Train models + deployment strategy | Medium | **Unblocks core feature** |
| 1.2 | Fix docker-compose port | Trivial | Developer UX |
| 2.1 | Fix MarketData unique constraint (migration 005) | Small | Correctness |
| 2.2 | Return new refresh token on `/auth/refresh` | Small | **Auth correctness** |
| 3.1 | Deep health check (DB + models) | Small | Production readiness |
| 3.2 | Implement rate limiting | Medium | Security |
| 3.3 | Concrete CORS origins | Trivial | Security |
| 3.4 | Structured logging | Medium | Observability |
| 4.1 | `GET /anomalies` list endpoint | Small | Feature completeness |
| 4.2 | Pagination | Medium | Scalability |
| 4.3 | `GET /watchlists/{id}` | Small | Feature completeness |
| 4.4 | Model reload endpoint | Small | Operational |

---

## Suggested Order

```
Week 1: 1.1 → 2.1 → 2.2 → 1.2   (unblock + fix known bugs)
Week 2: 3.1 → 3.3 → 4.1 → 4.3   (production readiness + missing endpoints)
Week 3: 3.2 → 3.4 → 4.2 → 4.4   (security + scalability)
```
