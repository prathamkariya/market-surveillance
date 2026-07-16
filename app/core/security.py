"""Password hashing and JWT helpers."""

from datetime import datetime, timedelta, timezone
import logging

import bcrypt
import jwt

from app.config import get_settings
from app.models.schemas import TokenPayload

settings = get_settings()
logger = logging.getLogger(__name__)

BCRYPT_ROUNDS = 12


def hash_password(plain: str) -> str:
    salt = bcrypt.gensalt(rounds=BCRYPT_ROUNDS)
    return bcrypt.hashpw(plain.encode("utf-8"), salt).decode("utf-8")


def verify_password(plain: str, hashed: str) -> bool:
    if not hashed:
        return False
    try:
        return bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("utf-8"))
    except (TypeError, ValueError):
        return False


def create_access_token(user_id: int, email: str, role: str) -> tuple[str, int]:
    now = datetime.now(timezone.utc)
    expire = now + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    expires_in = settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60

    payload = {
        "sub": str(user_id),
        "email": email,
        "role": role,
        "iat": int(now.timestamp()),
        "exp": int(expire.timestamp()),
        "type": "access",
    }
    token = jwt.encode(payload, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
    return token, expires_in


def verify_access_token(token: str) -> TokenPayload:
    payload = jwt.decode(
        token,
        settings.SECRET_KEY,
        algorithms=[settings.ALGORITHM],
        options={"verify_exp": True},
    )
    if payload.get("type") != "access":
        raise ValueError("Token type is not 'access'")
    return TokenPayload(
        user_id=int(payload["sub"]),
        email=payload["email"],
        role=payload["role"],
        exp=payload["exp"],
    )
