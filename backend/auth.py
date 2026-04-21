"""
backend/auth.py
───────────────
JWT-based authentication and role-based access control.

Roles: admin, doctor, patient
Passwords are bcrypt-hashed — never stored in plaintext.

In production, replace the in-memory user store with a proper DB table.
The interface (get_current_user, require_role) stays the same regardless.
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

import bcrypt
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from pydantic import BaseModel

from backend.config import (
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES,
    JWT_ALGORITHM,
    JWT_SECRET_KEY,
)

logger = logging.getLogger(__name__)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")

# ─── Models ──────────────────────────────────────────────────────────────────

class UserInDB(BaseModel):
    username:      str
    hashed_pw:     str
    role:          str   # "admin" | "doctor" | "patient"
    disabled:      bool = False


class TokenData(BaseModel):
    username: Optional[str] = None
    role:     Optional[str] = None


# ─── In-memory store (swap for DB in prod) ────────────────────────────────────
# Values: {"hashed_pw": ..., "role": ...}
_USER_DB: dict[str, dict] = {}


# ─── Password helpers ─────────────────────────────────────────────────────────

def hash_password(plain: str) -> str:
    return bcrypt.hashpw(plain.encode(), bcrypt.gensalt()).decode()


def verify_password(plain: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(plain.encode(), hashed.encode())
    except Exception:
        return False


# ─── User CRUD ────────────────────────────────────────────────────────────────

def create_user(username: str, password: str, role: str = "patient") -> None:
    if role not in ("admin", "doctor", "patient"):
        raise ValueError(f"Invalid role '{role}'.")
    if username in _USER_DB:
        raise ValueError(f"User '{username}' already exists.")
    _USER_DB[username] = {
        "hashed_pw": hash_password(password),
        "role":      role,
        "disabled":  False,
    }
    logger.info("Created user '%s' with role '%s'.", username, role)


def get_user(username: str) -> Optional[UserInDB]:
    data = _USER_DB.get(username)
    if data:
        return UserInDB(username=username, **data)
    return None


def authenticate_user(username: str, password: str) -> Optional[UserInDB]:
    user = get_user(username)
    if not user or user.disabled:
        return None
    if not verify_password(password, user.hashed_pw):
        return None
    return user


# ─── Token helpers ────────────────────────────────────────────────────────────

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    payload = data.copy()
    expire  = datetime.now(timezone.utc) + (
        expires_delta or timedelta(minutes=JWT_ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    payload["exp"] = expire
    return jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)


# ─── FastAPI dependencies ─────────────────────────────────────────────────────

async def get_current_user(token: str = Depends(oauth2_scheme)) -> UserInDB:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or expired token.",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload   = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        username: str = payload.get("sub")
        role:     str = payload.get("role")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username, role=role)
    except JWTError:
        raise credentials_exception

    user = get_user(token_data.username)
    if user is None or user.disabled:
        raise credentials_exception
    return user


def require_role(*allowed_roles: str):
    """
    FastAPI dependency factory.

    Usage:
        @router.get("/admin-only")
        def admin_endpoint(user=Depends(require_role("admin"))):
            ...
    """
    async def _check(current_user: UserInDB = Depends(get_current_user)):
        if current_user.role not in allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Role '{current_user.role}' is not permitted for this endpoint.",
            )
        return current_user
    return _check