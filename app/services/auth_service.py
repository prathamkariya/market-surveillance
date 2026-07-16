import hashlib
import secrets
from datetime import datetime, timedelta
from typing import Optional

from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy.orm import Session

from app.config import settings
from app.models import RefreshToken, User

import bcrypt

# ──────────────────────────────────────────────
# Password hashing
# ──────────────────────────────────────────────

def hash_password(plain: str) -> str:
    # Truncate to 72 bytes per bcrypt limitation just in case
    encoded = plain.encode('utf-8')[:72]
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(encoded, salt).decode('utf-8')


def verify_password(plain: str, hashed: str) -> bool:
    try:
        encoded = plain.encode('utf-8')[:72]
        return bcrypt.checkpw(encoded, hashed.encode('utf-8'))
    except Exception:
        return False


# ──────────────────────────────────────────────
# JWT access tokens
# ──────────────────────────────────────────────
def create_access_token(user_id: int, email: str) -> str:
    """
    Create a short-lived JWT access token.
    Payload contains user_id (sub) and email.
    Expires per ACCESS_TOKEN_EXPIRE_MINUTES setting.
    """
    expire = datetime.utcnow() + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    payload = {
        "sub": str(user_id),
        "email": email,
        "exp": expire,
        "type": "access",
    }
    return jwt.encode(payload, settings.SECRET_KEY, algorithm=settings.ALGORITHM)


def decode_access_token(token: str) -> Optional[dict]:
    """
    Decode and verify a JWT access token.
    Returns the payload dict or None if invalid/expired.
    """
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        # Reject refresh tokens used as access tokens
        if payload.get("type") != "access":
            return None
        return payload
    except JWTError:
        return None


# ──────────────────────────────────────────────
# Refresh tokens  (Phase 2)
# ──────────────────────────────────────────────
def _hash_refresh_token(raw_token: str) -> str:
    """
    SHA-256 hash of the raw refresh token for storage.
    We use SHA-256 (not bcrypt) here because:
      - Refresh tokens are already 256-bit cryptographically random (no dictionary attack risk)
      - SHA-256 is fast for lookup without sacrificing security
    """
    return hashlib.sha256(raw_token.encode()).hexdigest()


def create_refresh_token(db: Session, user_id: int) -> str:
    """
    Generate a cryptographically random refresh token, store its hash,
    and return the raw token to the client (only time it's ever visible).
    """
    raw_token = secrets.token_urlsafe(64)   # 512-bit entropy
    token_hash = _hash_refresh_token(raw_token)
    expires_at = datetime.utcnow() + timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS)

    db_token = RefreshToken(
        user_id=user_id,
        token_hash=token_hash,
        expires_at=expires_at,
    )
    db.add(db_token)
    db.commit()
    db.refresh(db_token)
    return raw_token


def rotate_refresh_token(db: Session, raw_token: str) -> Optional[tuple[str, str]]:
    """
    Validate a refresh token, revoke it, issue a new pair.

    Implements token rotation:
      1. Compute hash of the presented raw token
      2. Look it up in the DB
      3. Verify it's valid (not revoked, not expired)
      4. Revoke the old token
      5. Issue new access token + new refresh token

    Returns (new_access_token, new_refresh_token) or None if invalid.

    Security: if a refresh token is used after it's already been revoked,
    that's a sign of token theft — log and alert (not implemented here yet).
    """
    token_hash = _hash_refresh_token(raw_token)
    db_token = db.query(RefreshToken).filter(
        RefreshToken.token_hash == token_hash
    ).first()

    if db_token is None:
        return None  # Token doesn't exist

    if not db_token.is_valid:
        return None  # Revoked or expired

    # Revoke the used token (rotation — each token is single-use)
    db_token.revoked = True
    db.commit()

    # Issue new token pair
    user = db_token.user
    new_access = create_access_token(user.id, user.email)
    new_refresh = create_refresh_token(db, user.id)
    return new_access, new_refresh


def revoke_refresh_token(db: Session, raw_token: str, user_id: int) -> bool:
    """
    Revoke a specific refresh token (logout from one device).
    user_id check prevents one user revoking another user's tokens.
    Returns True if found and revoked, False otherwise.
    """
    token_hash = _hash_refresh_token(raw_token)
    db_token = db.query(RefreshToken).filter(
        RefreshToken.token_hash == token_hash,
        RefreshToken.user_id == user_id,
    ).first()

    if db_token is None or db_token.revoked:
        return False

    db_token.revoked = True
    db.commit()
    return True


def revoke_all_user_tokens(db: Session, user_id: int) -> int:
    """
    Revoke ALL refresh tokens for a user (logout from all devices).
    Returns the count of tokens revoked.
    """
    tokens = db.query(RefreshToken).filter(
        RefreshToken.user_id == user_id,
        RefreshToken.revoked == False,  # noqa: E712
    ).all()

    count = 0
    for token in tokens:
        token.revoked = True
        count += 1

    db.commit()
    return count


# ──────────────────────────────────────────────
# User lookup helpers
# ──────────────────────────────────────────────
def get_user_by_email(db: Session, email: str) -> Optional[User]:
    return db.query(User).filter(User.email == email).first()


def get_user_by_id(db: Session, user_id: int) -> Optional[User]:
    return db.query(User).filter(User.id == user_id).first()


def authenticate_user(db: Session, email: str, password: str) -> Optional[User]:
    """Verify credentials. Returns User if valid, None otherwise."""
    user = get_user_by_email(db, email)
    if user is None:
        # Constant-time dummy verify to prevent user-enumeration timing attacks.
        # The hash is intentionally invalid.
        verify_password("dummy", "$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/xxxxxxxxxxxxxxxxxxxxxx")
        return None
    if not verify_password(password, user.hashed_password):
        return None
    if not user.is_active:
        return None
    return user
