import secrets
from datetime import datetime, timedelta
from typing import Optional
from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import HTTPException, status

from app.config import settings

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def verify_password(plain_password: str) -> bool:
    stored_password = settings.password
    if stored_password.startswith("$2b$") or stored_password.startswith("$2a$"):
        return pwd_context.verify(plain_password, stored_password)
    return plain_password == stored_password


def create_session_token(username: str = "user") -> str:
    expire = datetime.utcnow() + timedelta(hours=settings.session_expire_hours)
    to_encode = {"sub": username, "exp": expire}
    encoded_jwt = jwt.encode(to_encode, settings.session_secret, algorithm="HS256")
    return encoded_jwt


def verify_session_token(token: str) -> Optional[str]:
    try:
        payload = jwt.decode(token, settings.session_secret, algorithms=["HS256"])
        username: str = payload.get("sub")
        if username is None:
            return None
        return username
    except JWTError:
        return None


def hash_password(password: str) -> str:
    return pwd_context.hash(password)