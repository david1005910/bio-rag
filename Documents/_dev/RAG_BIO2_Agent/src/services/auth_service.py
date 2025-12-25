import os
import hashlib
import secrets
from datetime import datetime, timedelta
from typing import Optional
from jose import JWTError, jwt
from sqlalchemy.orm import Session
from src.models.user import User

SECRET_KEY = os.environ.get("SESSION_SECRET", "bio-rag-secret-key-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24  # 24 hours

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify password using SHA256 with salt."""
    if not plain_password or not hashed_password:
        return False
    try:
        salt, stored_hash = hashed_password.split("$", 1)
        computed_hash = hashlib.sha256((salt + plain_password).encode()).hexdigest()
        return computed_hash == stored_hash
    except ValueError:
        return False

def get_password_hash(password: str) -> str:
    """Hash password using SHA256 with random salt."""
    salt = secrets.token_hex(16)
    hash_value = hashlib.sha256((salt + password).encode()).hexdigest()
    return f"{salt}${hash_value}"

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def decode_token(token: str) -> Optional[dict]:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError:
        return None

def get_user_by_email(db: Session, email: str) -> Optional[User]:
    return db.query(User).filter(User.email == email).first()

def create_user(db: Session, email: str, password: str, name: str = None) -> User:
    hashed_password = get_password_hash(password)
    user = User(
        email=email,
        password_hash=hashed_password,
        name=name or email.split('@')[0]
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user

def authenticate_user(db: Session, email: str, password: str) -> Optional[User]:
    user = get_user_by_email(db, email)
    if not user:
        return None
    if not verify_password(password, user.password_hash):
        return None
    user.last_login = datetime.utcnow()
    db.commit()
    return user
