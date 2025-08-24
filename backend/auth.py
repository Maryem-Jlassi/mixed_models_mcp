from datetime import datetime, timedelta
from typing import Optional

from fastapi import APIRouter, HTTPException, Depends, status, Request
from fastapi.security import OAuth2PasswordRequestForm
from passlib.context import CryptContext
import jwt

from config import settings
from db import get_db
from bson import ObjectId

router = APIRouter(prefix="/auth", tags=["auth"])

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(password: str, password_hash: str) -> bool:
    return pwd_context.verify(password, password_hash)


def create_access_token(subject: dict, expires_minutes: int = settings.JWT_EXPIRES_MIN) -> str:
    to_encode = subject.copy()
    expire = datetime.utcnow() + timedelta(minutes=expires_minutes)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, settings.JWT_SECRET, algorithm=settings.JWT_ALGO)


@router.post("/register")
async def register(payload: dict, db=Depends(get_db)):
    # Expected: { name, email, password }
    name = (payload.get("name") or "").strip()
    email = (payload.get("email") or "").strip().lower()
    password = payload.get("password") or ""
    if not name or not email or not password:
        raise HTTPException(status_code=400, detail="name, email, password are required")

    existing = await db.users.find_one({"email": email})
    if existing:
        raise HTTPException(status_code=409, detail="Email already registered")

    doc = {
        "name": name,
        "email": email,
        "password_hash": hash_password(password),
        "created_at": datetime.utcnow().isoformat(),
    }
    res = await db.users.insert_one(doc)
    user_id = str(res.inserted_id)

    token = create_access_token({"sub": user_id, "email": email})
    return {"access_token": token, "token_type": "bearer", "user": {"_id": user_id, "name": name, "email": email}}


@router.post("/login")
async def login(payload: dict, db=Depends(get_db)):
    # Accept JSON body: { email, password }
    email = (payload.get("email") or "").strip().lower()
    password = payload.get("password") or ""
    if not email or not password:
        raise HTTPException(status_code=400, detail="email and password are required")

    user = await db.users.find_one({"email": email})
    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials")

    if not verify_password(password, user.get("password_hash") or ""):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    user_id = str(user.get("_id"))
    token = create_access_token({"sub": user_id, "email": email})
    return {"access_token": token, "token_type": "bearer", "user": {"_id": user_id, "name": user.get("name"), "email": email}}


# Auth helpers
async def get_current_user_optional(request: Request, db=Depends(get_db)) -> Optional[dict]:
    """Return the current user from Bearer token if provided; otherwise None.
    The returned user dict contains stringified "_id".
    """
    auth_header = request.headers.get("authorization") or request.headers.get("Authorization")
    if not auth_header or not auth_header.lower().startswith("bearer "):
        return None
    token = auth_header.split(" ", 1)[1].strip()
    try:
        payload = jwt.decode(token, settings.JWT_SECRET, algorithms=[settings.JWT_ALGO])
        user_id = payload.get("sub")
        if not user_id:
            return None
        user = await db.users.find_one({"_id": ObjectId(user_id)})
        if not user:
            return None
        return {"_id": str(user.get("_id")), "name": user.get("name"), "email": user.get("email")}
    except Exception:
        # Any issue decoding or fetching user -> treat as unauthenticated
        return None


async def get_current_user(request: Request, db=Depends(get_db)) -> dict:
    """Require a valid Bearer token and return the current user, else 401."""
    user = await get_current_user_optional(request, db)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")
    return user
