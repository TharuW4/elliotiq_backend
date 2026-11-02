# app/auth_service.py

from __future__ import annotations

import os
import re
import secrets
import hashlib
import hmac
import base64
import json
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional, Tuple
from google.oauth2 import id_token
from google.auth.transport import requests as google_requests

try:
    from app.main import db
except Exception:
    db = None

from app.subscription_service import (
    create_inactive_subscription,
    get_subscription,
    check_subscription_access,
    datetime_to_iso,
    utc_now,
)


# ================================
# Config
# ================================

TOKEN_EXPIRY_DAYS = int(os.getenv("AUTH_TOKEN_EXPIRY_DAYS", "30"))
JWT_SECRET = os.getenv("JWT_SECRET", "change-this-long-random-secret")
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID", "").strip()


# ================================
# Helpers
# ================================

def normalize_email(email: str) -> str:
    return (email or "").strip().lower()


def is_valid_email(email: str) -> bool:
    email = normalize_email(email)
    return bool(re.match(r"^[^\s@]+@[^\s@]+\.[^\s@]+$", email))


def utc_timestamp() -> int:
    return int(datetime.now(timezone.utc).timestamp())


def make_uid() -> str:
    return secrets.token_urlsafe(18)


def hash_password(password: str) -> str:
    """
    Simple PBKDF2 password hash.
    Stored format:
    pbkdf2_sha256$iterations$salt$hash
    """
    if not password:
        raise ValueError("Password is required.")

    iterations = 260000
    salt = secrets.token_hex(16)

    password_hash = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt.encode("utf-8"),
        iterations,
    ).hex()

    return f"pbkdf2_sha256${iterations}${salt}${password_hash}"


def verify_password(password: str, stored_hash: str) -> bool:
    if not password or not stored_hash:
        return False

    try:
        algo, iterations_str, salt, password_hash = stored_hash.split("$", 3)

        if algo != "pbkdf2_sha256":
            return False

        iterations = int(iterations_str)

        test_hash = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            salt.encode("utf-8"),
            iterations,
        ).hex()

        return hmac.compare_digest(test_hash, password_hash)

    except Exception:
        return False


def _b64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("utf-8")


def _b64url_decode(data: str) -> bytes:
    padding = "=" * (-len(data) % 4)
    return base64.urlsafe_b64decode(data + padding)


def create_token(payload: Dict[str, Any]) -> str:
    """
    Lightweight JWT-like token using HMAC SHA256.
    No external dependency needed.

    Later, if you want, we can replace this with PyJWT.
    """
    header = {
        "alg": "HS256",
        "typ": "JWT",
    }

    now_ts = utc_timestamp()

    token_payload = {
        **payload,
        "iat": now_ts,
        "exp": now_ts + TOKEN_EXPIRY_DAYS * 24 * 60 * 60,
    }

    header_b64 = _b64url_encode(json.dumps(header, separators=(",", ":")).encode("utf-8"))
    payload_b64 = _b64url_encode(json.dumps(token_payload, separators=(",", ":")).encode("utf-8"))

    signing_input = f"{header_b64}.{payload_b64}".encode("utf-8")

    signature = hmac.new(
        JWT_SECRET.encode("utf-8"),
        signing_input,
        hashlib.sha256,
    ).digest()

    signature_b64 = _b64url_encode(signature)

    return f"{header_b64}.{payload_b64}.{signature_b64}"


def verify_token(token: str) -> Optional[Dict[str, Any]]:
    if not token:
        return None

    try:
        parts = token.split(".")
        if len(parts) != 3:
            return None

        header_b64, payload_b64, signature_b64 = parts
        signing_input = f"{header_b64}.{payload_b64}".encode("utf-8")

        expected_signature = hmac.new(
            JWT_SECRET.encode("utf-8"),
            signing_input,
            hashlib.sha256,
        ).digest()

        expected_signature_b64 = _b64url_encode(expected_signature)

        if not hmac.compare_digest(expected_signature_b64, signature_b64):
            return None

        payload = json.loads(_b64url_decode(payload_b64).decode("utf-8"))

        exp = int(payload.get("exp", 0))
        if exp < utc_timestamp():
            return None

        return payload

    except Exception:
        return None


def get_bearer_token(auth_header: Optional[str]) -> Optional[str]:
    if not auth_header:
        return None

    parts = auth_header.split(" ", 1)

    if len(parts) != 2:
        return None

    scheme, token = parts

    if scheme.lower() != "bearer":
        return None

    return token.strip()


# ================================
# Firestore user helpers
# ================================

def get_user_by_email(email: str) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    if db is None:
        raise RuntimeError("Firestore db is not initialized.")

    email = normalize_email(email)

    if not email:
        return None, None

    docs = (
        db.collection("users")
        .where("email", "==", email)
        .limit(1)
        .stream()
    )

    for doc in docs:
        data = doc.to_dict()
        data["uid"] = doc.id
        return doc.id, data

    return None, None


def get_user_by_uid(uid: str) -> Optional[Dict[str, Any]]:
    if db is None:
        raise RuntimeError("Firestore db is not initialized.")

    if not uid:
        return None

    doc = db.collection("users").document(uid).get()

    if not doc.exists:
        return None

    data = doc.to_dict()
    data["uid"] = doc.id
    return data


def update_last_login(uid: str) -> None:
    if db is None:
        raise RuntimeError("Firestore db is not initialized.")

    db.collection("users").document(uid).set(
        {
            "last_login_at": datetime_to_iso(utc_now()),
            "updated_at": datetime_to_iso(utc_now()),
        },
        merge=True,
    )


# ================================
# Auth actions
# ================================

def register_user(
    *,
    email: str,
    password: str,
    name: str = "",
    auth_provider: str = "email_password",
) -> Dict[str, Any]:
    """
    New user signup.
    User collection එකට save කරනවා.
    Subscription default inactive කරනවා.
    """
    if db is None:
        raise RuntimeError("Firestore db is not initialized.")

    email = normalize_email(email)
    name = (name or "").strip()

    if not is_valid_email(email):
        raise ValueError("Invalid email address.")

    if not password or len(password) < 6:
        raise ValueError("Password must be at least 6 characters.")

    existing_uid, _ = get_user_by_email(email)
    if existing_uid:
        raise ValueError("An account already exists with this email.")

    uid = make_uid()
    now = utc_now()

    user_data = {
        "uid": uid,
        "email": email,
        "name": name,
        "auth_provider": auth_provider,
        "password_hash": hash_password(password),

        "role": "user",
        "account_status": "active",

        "subscription_status": "inactive",
        "plan": None,
        "plan_name": None,
        "active_until": None,

        "created_at": datetime_to_iso(now),
        "last_login_at": datetime_to_iso(now),
        "updated_at": datetime_to_iso(now),
    }

    db.collection("users").document(uid).set(user_data)

    create_inactive_subscription(uid, email)

    token = create_token(
        {
            "uid": uid,
            "email": email,
            "role": "user",
        }
    )

    safe_user = sanitize_user(user_data)

    return {
        "token": token,
        "user": safe_user,
        "subscription": {
            "status": "inactive",
            "plan": None,
            "active_until": None,
        },
    }


def login_user(*, email: str, password: str) -> Dict[str, Any]:
    if db is None:
        raise RuntimeError("Firestore db is not initialized.")

    email = normalize_email(email)

    if not is_valid_email(email):
        raise ValueError("Invalid email address.")

    uid, user = get_user_by_email(email)

    if not uid or not user:
        raise ValueError("Invalid email or password.")

    if (user.get("account_status") or "active") != "active":
        raise ValueError("This account is disabled.")

    stored_hash = user.get("password_hash")

    if not verify_password(password, stored_hash):
        raise ValueError("Invalid email or password.")

    update_last_login(uid)

    token = create_token(
        {
            "uid": uid,
            "email": email,
            "role": user.get("role", "user"),
        }
    )

    subscription = get_subscription(uid)

    allowed, sub_payload = check_subscription_access(uid)

    return {
        "token": token,
        "user": sanitize_user(user),
        "subscription": {
            "allowed": allowed,
            **sub_payload,
        },
    }


def get_current_user_from_header(auth_header: Optional[str]) -> Optional[Dict[str, Any]]:
    token = get_bearer_token(auth_header)
    payload = verify_token(token) if token else None

    if not payload:
        return None

    uid = payload.get("uid")
    user = get_user_by_uid(uid)

    if not user:
        return None

    if (user.get("account_status") or "active") != "active":
        return None

    return user


def get_me(auth_header: Optional[str]) -> Dict[str, Any]:
    user = get_current_user_from_header(auth_header)

    if not user:
        raise ValueError("Unauthorized.")

    uid = user.get("uid")
    allowed, sub_payload = check_subscription_access(uid)

    return {
        "user": sanitize_user(user),
        "subscription": {
            "allowed": allowed,
            **sub_payload,
        },
    }


def sanitize_user(user: Dict[str, Any]) -> Dict[str, Any]:
    """
    password_hash frontend එකට යවන්නේ නැහැ.
    """
    if not user:
        return {}

    clean = dict(user)
    clean.pop("password_hash", None)
    clean.pop("google_sub", None)

    return clean

def verify_google_id_token(google_id_token: str) -> Dict[str, Any]:
    """
    Google ID token verify කරලා profile data return කරනවා.
    """
    if not GOOGLE_CLIENT_ID:
        raise ValueError("GOOGLE_CLIENT_ID is not configured.")

    if not google_id_token:
        raise ValueError("Google ID token is required.")

    try:
        info = id_token.verify_oauth2_token(
            google_id_token,
            google_requests.Request(),
            GOOGLE_CLIENT_ID,
        )
    except Exception:
        raise ValueError("Invalid Google login token.")

    email = normalize_email(info.get("email", ""))

    if not email:
        raise ValueError("Google account email not found.")

    if not info.get("email_verified", False):
        raise ValueError("Google email is not verified.")

    return {
        "google_sub": info.get("sub"),
        "email": email,
        "name": info.get("name", ""),
        "picture": info.get("picture", ""),
    }


def google_login_user(*, google_id_token: str) -> Dict[str, Any]:
    """
    Google login.
    User exists නම් login කරනවා.
    User නැත්නම් inactive user create කරනවා.
    """
    if db is None:
        raise RuntimeError("Firestore db is not initialized.")

    google_profile = verify_google_id_token(google_id_token)

    email = google_profile["email"]
    name = google_profile.get("name", "")
    google_sub = google_profile.get("google_sub")
    picture = google_profile.get("picture", "")

    uid, user = get_user_by_email(email)

    now = utc_now()

    if not uid or not user:
        uid = make_uid()

        user_data = {
            "uid": uid,
            "email": email,
            "name": name,
            "auth_provider": "google",
            "google_sub": google_sub,
            "picture": picture,

            "role": "user",
            "account_status": "active",

            "subscription_status": "inactive",
            "plan": None,
            "plan_name": None,
            "active_until": None,

            "created_at": datetime_to_iso(now),
            "last_login_at": datetime_to_iso(now),
            "updated_at": datetime_to_iso(now),
        }

        db.collection("users").document(uid).set(user_data)
        create_inactive_subscription(uid, email)

        user = user_data

    else:
        if (user.get("account_status") or "active") != "active":
            raise ValueError("This account is disabled.")

        db.collection("users").document(uid).set(
            {
                "auth_provider": user.get("auth_provider") or "google",
                "google_sub": google_sub,
                "picture": picture,
                "name": user.get("name") or name,
                "last_login_at": datetime_to_iso(now),
                "updated_at": datetime_to_iso(now),
            },
            merge=True,
        )

        user = get_user_by_uid(uid) or user

    token = create_token(
        {
            "uid": uid,
            "email": email,
            "role": user.get("role", "user"),
        }
    )

    allowed, sub_payload = check_subscription_access(uid)

    return {
        "token": token,
        "user": sanitize_user(user),
        "subscription": {
            "allowed": allowed,
            **sub_payload,
        },
    }

# ================================
# Desktop Google Login Bridge
# ================================

DESKTOP_LOGIN_EXPIRY_MINUTES = 10


def create_desktop_login_request() -> Dict[str, Any]:
    """
    EXE එකෙන් Google login start කරනකොට temporary request එකක් create කරනවා.
    """
    if db is None:
        raise RuntimeError("Firestore db is not initialized.")

    rid = secrets.token_urlsafe(32)
    now = utc_now()
    expires_at = now + timedelta(minutes=DESKTOP_LOGIN_EXPIRY_MINUTES)

    payload = {
        "rid": rid,
        "status": "pending",
        "created_at": datetime_to_iso(now),
        "expires_at": datetime_to_iso(expires_at),
        "approved_at": None,
        "uid": None,
        "email": None,
        "token": None,
    }

    db.collection("desktop_login_requests").document(rid).set(payload)

    return {
        "rid": rid,
        "expires_at": datetime_to_iso(expires_at),
    }


def approve_desktop_login_request(*, rid: str, auth_header: Optional[str]) -> Dict[str, Any]:
    """
    Browser/main site එකේ Google login success උනාට පස්සේ,
    ඒ logged-in user token එක desktop request එකට attach කරනවා.
    """
    if db is None:
        raise RuntimeError("Firestore db is not initialized.")

    rid = (rid or "").strip()

    if not rid:
        raise ValueError("Desktop login request ID is required.")

    user = get_current_user_from_header(auth_header)

    if not user:
        raise ValueError("Login required.")

    ref = db.collection("desktop_login_requests").document(rid)
    doc = ref.get()

    if not doc.exists:
        raise ValueError("Desktop login request not found.")

    req = doc.to_dict() or {}

    if req.get("status") != "pending":
        raise ValueError("Desktop login request is no longer pending.")

    expires_at_raw = req.get("expires_at")
    if expires_at_raw:
        try:
            expires_at = datetime.fromisoformat(str(expires_at_raw).replace("Z", "+00:00"))
            if expires_at < utc_now():
                ref.set({"status": "expired"}, merge=True)
                raise ValueError("Desktop login request expired.")
        except ValueError:
            raise
        except Exception:
            pass

    uid = user.get("uid")
    email = normalize_email(user.get("email", ""))

    token = create_token(
        {
            "uid": uid,
            "email": email,
            "role": user.get("role", "user"),
        }
    )

    allowed, sub_payload = check_subscription_access(uid)

    now = utc_now()

    ref.set(
        {
            "status": "approved",
            "approved_at": datetime_to_iso(now),
            "uid": uid,
            "email": email,
            "token": token,
            "subscription_allowed": allowed,
            "subscription": sub_payload,
        },
        merge=True,
    )

    return {
        "rid": rid,
        "status": "approved",
        "subscription": {
            "allowed": allowed,
            **sub_payload,
        },
    }


def get_desktop_login_status(*, rid: str) -> Dict[str, Any]:
    """
    EXE එක poll කරලා login approvedද බලන endpoint එකට support කරන function එක.
    Approved නම් token + user + subscription return කරනවා.
    """
    if db is None:
        raise RuntimeError("Firestore db is not initialized.")

    rid = (rid or "").strip()

    if not rid:
        raise ValueError("Desktop login request ID is required.")

    ref = db.collection("desktop_login_requests").document(rid)
    doc = ref.get()

    if not doc.exists:
        raise ValueError("Desktop login request not found.")

    req = doc.to_dict() or {}

    expires_at_raw = req.get("expires_at")
    if expires_at_raw and req.get("status") == "pending":
        try:
            expires_at = datetime.fromisoformat(str(expires_at_raw).replace("Z", "+00:00"))
            if expires_at < utc_now():
                ref.set({"status": "expired"}, merge=True)
                return {
                    "status": "expired",
                    "message": "Desktop login request expired.",
                }
        except Exception:
            pass

    status = req.get("status", "pending")

    if status != "approved":
        return {
            "status": status,
            "message": "Waiting for browser login approval.",
        }

    uid = req.get("uid")
    token = req.get("token")

    user = get_user_by_uid(uid)

    if not user or not token:
        return {
            "status": "failed",
            "message": "Approved login data is missing.",
        }

    allowed, sub_payload = check_subscription_access(uid)

    return {
        "status": "approved",
        "token": token,
        "user": sanitize_user(user),
        "subscription": {
            "allowed": allowed,
            **sub_payload,
        },
    }