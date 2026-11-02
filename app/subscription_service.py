# app/subscription_service.py

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional, Tuple

try:
    from app.main import db
except Exception:
    db = None


PLAN_DURATIONS_DAYS = {
    "monthly": 30,
    "three_months": 90,
    "six_months": 180,
}

PLAN_DISPLAY_NAMES = {
    "monthly": "Monthly Plan",
    "three_months": "3 Months Plan",
    "six_months": "6 Months Plan",
}


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def datetime_to_iso(dt: Optional[datetime]) -> Optional[str]:
    if not dt:
        return None
    return dt.astimezone(timezone.utc).isoformat()


def parse_datetime(value: Any) -> Optional[datetime]:
    if not value:
        return None

    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)

    if isinstance(value, str):
        try:
            cleaned = value.replace("Z", "+00:00")
            parsed = datetime.fromisoformat(cleaned)
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed.astimezone(timezone.utc)
        except Exception:
            return None

    return None


def get_plan_duration_days(plan: str) -> int:
    plan_key = (plan or "").strip().lower()

    if plan_key not in PLAN_DURATIONS_DAYS:
        raise ValueError(f"Invalid plan: {plan}")

    return PLAN_DURATIONS_DAYS[plan_key]


def get_user_by_email(email: str) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    if db is None:
        raise RuntimeError("Firestore db is not initialized.")

    email = (email or "").strip().lower()
    if not email:
        return None, None

    docs = (
        db.collection("users")
        .where("email", "==", email)
        .limit(1)
        .stream()
    )

    for doc in docs:
        return doc.id, doc.to_dict()

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


def create_inactive_subscription(uid: str, email: str) -> Dict[str, Any]:
    if db is None:
        raise RuntimeError("Firestore db is not initialized.")

    now = utc_now()

    subscription_data = {
        "uid": uid,
        "email": (email or "").strip().lower(),
        "status": "inactive",
        "plan": None,
        "plan_name": None,
        "started_at": None,
        "active_until": None,
        "provider": None,
        "last_payment_id": None,
        "created_at": datetime_to_iso(now),
        "updated_at": datetime_to_iso(now),
    }

    db.collection("subscriptions").document(uid).set(subscription_data, merge=True)

    db.collection("users").document(uid).set(
        {
            "subscription_status": "inactive",
            "plan": None,
            "plan_name": None,
            "active_until": None,
            "updated_at": datetime_to_iso(now),
        },
        merge=True,
    )

    return subscription_data


def get_subscription(uid: str) -> Optional[Dict[str, Any]]:
    if db is None:
        raise RuntimeError("Firestore db is not initialized.")

    if not uid:
        return None

    doc = db.collection("subscriptions").document(uid).get()
    if not doc.exists:
        return None

    data = doc.to_dict()
    data["uid"] = uid
    return data


def mark_subscription_expired(uid: str) -> None:
    if db is None:
        raise RuntimeError("Firestore db is not initialized.")

    now_iso = datetime_to_iso(utc_now())

    db.collection("subscriptions").document(uid).set(
        {
            "status": "expired",
            "updated_at": now_iso,
        },
        merge=True,
    )

    db.collection("users").document(uid).set(
        {
            "subscription_status": "expired",
            "updated_at": now_iso,
        },
        merge=True,
    )


def is_subscription_active(subscription: Optional[Dict[str, Any]]) -> bool:
    if not subscription:
        return False

    status = (subscription.get("status") or "").lower()
    active_until = parse_datetime(subscription.get("active_until"))

    if status != "active":
        return False

    if not active_until:
        return False

    return active_until > utc_now()


def check_subscription_access(uid: str) -> Tuple[bool, Dict[str, Any]]:
    """
    Returns:
        (allowed, payload)

    allowed = True නම් userට access දෙන්න පුළුවන්.
    allowed = False නම් payload එක frontend/backend response එකට use කරන්න.
    """
    if not uid:
        return False, {
            "error": "unauthorized",
            "message": "Login required.",
            "status_code": 401,
        }

    subscription = get_subscription(uid)

    if not subscription:
        return False, {
            "error": "subscription_required",
            "message": "No subscription found. Please activate a plan.",
            "status_code": 402,
        }

    status = (subscription.get("status") or "inactive").lower()
    active_until = parse_datetime(subscription.get("active_until"))

    if status != "active":
        return False, {
            "error": "subscription_required",
            "message": "Please activate a plan to use ElliotIQ.",
            "subscription_status": status,
            "status_code": 402,
        }

    if not active_until:
        mark_subscription_expired(uid)
        return False, {
            "error": "subscription_expired",
            "message": "Your ElliotIQ plan has expired. Please renew your plan.",
            "subscription_status": "expired",
            "status_code": 402,
        }

    if active_until <= utc_now():
        mark_subscription_expired(uid)
        return False, {
            "error": "subscription_expired",
            "message": "Your ElliotIQ plan has expired. Please renew your plan.",
            "subscription_status": "expired",
            "active_until": datetime_to_iso(active_until),
            "status_code": 402,
        }

    return True, {
        "subscription_status": "active",
        "plan": subscription.get("plan"),
        "plan_name": subscription.get("plan_name"),
        "active_until": datetime_to_iso(active_until),
    }


def activate_subscription(
    *,
    email: str,
    plan: str,
    provider: str = "manual",
    payment_id: Optional[str] = None,
    amount: Optional[float] = None,
    currency: str = "USD",
) -> Dict[str, Any]:
    """
    Manual admin activation / future PayHere webhook both can call this function.
    """
    if db is None:
        raise RuntimeError("Firestore db is not initialized.")

    email = (email or "").strip().lower()
    plan = (plan or "").strip().lower()

    if not email:
        raise ValueError("email is required")

    duration_days = get_plan_duration_days(plan)
    uid, user = get_user_by_email(email)

    if not uid or not user:
        raise ValueError(f"No user found for email: {email}")

    now = utc_now()
    current_subscription = get_subscription(uid)
    current_active_until = None

    if current_subscription:
        current_active_until = parse_datetime(current_subscription.get("active_until"))

    if current_active_until and current_active_until > now:
        base_date = current_active_until
    else:
        base_date = now

    new_active_until = base_date + timedelta(days=duration_days)
    payment_doc_id = payment_id or f"manual_{uid}_{int(now.timestamp())}"

    plan_name = PLAN_DISPLAY_NAMES.get(plan, plan)

    payment_data = {
        "uid": uid,
        "email": email,
        "provider": provider,
        "payment_id": payment_doc_id,
        "plan": plan,
        "plan_name": plan_name,
        "amount": amount,
        "currency": currency,
        "status": "success",
        "paid_at": datetime_to_iso(now),
        "created_at": datetime_to_iso(now),
    }

    subscription_data = {
        "uid": uid,
        "email": email,
        "status": "active",
        "plan": plan,
        "plan_name": plan_name,
        "started_at": datetime_to_iso(now),
        "active_until": datetime_to_iso(new_active_until),
        "provider": provider,
        "last_payment_id": payment_doc_id,
        "updated_at": datetime_to_iso(now),
    }

    user_update = {
        "subscription_status": "active",
        "plan": plan,
        "plan_name": plan_name,
        "active_until": datetime_to_iso(new_active_until),
        "payment_provider": provider,
        "updated_at": datetime_to_iso(now),
    }

    db.collection("payments").document(payment_doc_id).set(payment_data, merge=True)
    db.collection("subscriptions").document(uid).set(subscription_data, merge=True)
    db.collection("users").document(uid).set(user_update, merge=True)

    return {
        "uid": uid,
        "email": email,
        "plan": plan,
        "plan_name": plan_name,
        "subscription_status": "active",
        "active_until": datetime_to_iso(new_active_until),
        "payment_id": payment_doc_id,
    }


def get_subscription_status_by_email(email: str) -> Dict[str, Any]:
    uid, user = get_user_by_email(email)

    if not uid:
        return {
            "exists": False,
            "email": email,
            "subscription_status": "not_found",
        }

    subscription = get_subscription(uid)

    if not subscription:
        return {
            "exists": True,
            "uid": uid,
            "email": email,
            "subscription_status": "inactive",
            "plan": None,
            "active_until": None,
        }

    allowed, access_payload = check_subscription_access(uid)

    return {
        "exists": True,
        "uid": uid,
        "email": email,
        "allowed": allowed,
        **access_payload,
    }


def expire_old_subscriptions() -> Dict[str, int]:
    """
    Daily cron job එකට.
    Active status තියෙන, active_until past වෙච්ච subscriptions expired කරනවා.
    """
    if db is None:
        raise RuntimeError("Firestore db is not initialized.")

    checked = 0
    expired = 0

    docs = (
        db.collection("subscriptions")
        .where("status", "==", "active")
        .stream()
    )

    now = utc_now()

    for doc in docs:
        checked += 1
        data = doc.to_dict()
        active_until = parse_datetime(data.get("active_until"))

        if not active_until or active_until <= now:
            mark_subscription_expired(doc.id)
            expired += 1

    return {
        "checked": checked,
        "expired": expired,
    }