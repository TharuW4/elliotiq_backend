# scripts/expire_subscriptions.py

from __future__ import annotations

import os
import sys
from pathlib import Path
from datetime import datetime, timezone


# ------------------------------------------------------------
# Make project root importable
# ------------------------------------------------------------
# This allows:
#   python scripts/expire_subscriptions.py
# to import app.subscription_service correctly.
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def main() -> int:
    print("=" * 70)
    print(f"[SUBSCRIPTION CLEANUP] Started at {utc_now_iso()}")
    print(f"[SUBSCRIPTION CLEANUP] Project root: {PROJECT_ROOT}")

    try:
        from app.subscription_service import expire_old_subscriptions

        result = expire_old_subscriptions()

        checked = result.get("checked", 0)
        expired = result.get("expired", 0)

        print(f"[SUBSCRIPTION CLEANUP] Checked subscriptions: {checked}")
        print(f"[SUBSCRIPTION CLEANUP] Expired subscriptions: {expired}")
        print(f"[SUBSCRIPTION CLEANUP] Finished at {utc_now_iso()}")
        print("=" * 70)

        return 0

    except Exception as exc:
        print(f"[SUBSCRIPTION CLEANUP] ERROR: {exc}")
        print(f"[SUBSCRIPTION CLEANUP] Failed at {utc_now_iso()}")
        print("=" * 70)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())