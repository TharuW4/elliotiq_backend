import os
from functools import wraps

from dotenv import load_dotenv
load_dotenv()

from flask import Flask, request, jsonify
from flask_cors import CORS

from . import main
from .main import db

from .trade_audit import (
    get_trade_audit_stats,
    list_trade_audit_records,
    check_open_audit_trades
)

from .auth_service import (
    register_user,
    login_user,
    google_login_user,
    get_me,
    get_current_user_from_header,
    create_desktop_login_request,
    approve_desktop_login_request,
    get_desktop_login_status,
)

from .subscription_service import (
    check_subscription_access,
    activate_subscription,
    get_subscription_status_by_email,
)


app = Flask(__name__)

# ================================
# Config
# ================================

REQUIRE_SUBSCRIPTION = os.getenv("REQUIRE_SUBSCRIPTION", "false").lower() == "true"
ADMIN_SECRET = os.getenv("ADMIN_SECRET", "change-this-admin-secret")

# Development වලට open CORS තියාගන්න.
# Production යද්දී origins list එක restrict කරන්න.
ALLOWED_ORIGINS = os.getenv(
    "ALLOWED_ORIGINS",
    "*"
)

if ALLOWED_ORIGINS == "*":
    CORS(app)
else:
    CORS(
        app,
        origins=[origin.strip() for origin in ALLOWED_ORIGINS.split(",") if origin.strip()]
    )


# ================================
# Helpers
# ================================

def get_json_body():
    return request.get_json(silent=True) or {}

def make_json_safe(value):
    """
    Convert NumPy/Pandas scalar values and nested containers
    into standard Python JSON-compatible values.
    """
    if isinstance(value, dict):
        return {
            str(key): make_json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple, set)):
        return [
            make_json_safe(item)
            for item in value
        ]

    # Converts np.float32, np.float64, np.int64,
    # np.bool_ and similar scalar objects.
    item_method = getattr(value, "item", None)

    if callable(item_method):
        try:
            converted = item_method()

            if converted is not value:
                return make_json_safe(converted)

        except Exception:
            pass

    # Converts NumPy arrays if one reaches the response.
    tolist_method = getattr(value, "tolist", None)

    if callable(tolist_method):
        try:
            return make_json_safe(
                tolist_method()
            )

        except Exception:
            pass

    return value

def error_response(message, status_code=400, error="bad_request", extra=None):
    payload = {
        "status": "failed",
        "error": error,
        "message": message,
    }

    if extra:
        payload.update(extra)

    return jsonify(payload), status_code


def require_auth_and_subscription(fn):
    """
    /analyze වගේ protected endpoints වලට.
    
    REQUIRE_SUBSCRIPTION=false නම්:
      - old frontend/testing break නොවෙන්න subscription check skip වෙනවා.
    
    REQUIRE_SUBSCRIPTION=true නම්:
      - Authorization: Bearer <token> ඕන
      - subscription active ඕන
    """
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if not REQUIRE_SUBSCRIPTION:
            return fn(*args, **kwargs)

        user = get_current_user_from_header(request.headers.get("Authorization"))

        if not user:
            return error_response(
                "Login required.",
                status_code=401,
                error="unauthorized"
            )

        allowed, sub_payload = check_subscription_access(user.get("uid"))

        if not allowed:
            status_code = int(sub_payload.get("status_code", 402))
            return error_response(
                sub_payload.get("message", "Subscription required."),
                status_code=status_code,
                error=sub_payload.get("error", "subscription_required"),
                extra={
                    "subscription": sub_payload
                }
            )

        request.current_user = user
        request.subscription = sub_payload

        return fn(*args, **kwargs)

    return wrapper


def require_admin(fn):
    """
    Temporary manual activation endpoint එක protect කරන්න.
    Request header:
      X-Admin-Secret: your-secret
    """
    @wraps(fn)
    def wrapper(*args, **kwargs):
        incoming_secret = request.headers.get("X-Admin-Secret", "")

        if not ADMIN_SECRET or ADMIN_SECRET == "change-this-admin-secret":
            return error_response(
                "ADMIN_SECRET is not configured safely.",
                status_code=500,
                error="admin_secret_not_configured"
            )

        if incoming_secret != ADMIN_SECRET:
            return error_response(
                "Admin access denied.",
                status_code=403,
                error="forbidden"
            )

        return fn(*args, **kwargs)

    return wrapper


# ================================
# Health
# ================================

@app.route("/", methods=["GET"])
def root():
    return jsonify({
        "status": "ok",
        "service": "ElliotIQ API",
        "require_subscription": REQUIRE_SUBSCRIPTION,
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "service": "ElliotIQ API",
    })

# ================================
# App Version
# ================================

@app.route("/app/version", methods=["GET"])
def app_version():
    return jsonify({
        "status": "success",
        "latest_version": "1.0.0",
        "download_url": "https://github.com/TharuW4/elliotiq-app/releases/download/v1.0.0/ElliotIQ-Setup-1.0.0.exe",
        "force_update": False,
        "message": "You are using the latest ElliotIQ version."
    })

# ================================
# Auth Routes
# ================================

@app.route("/auth/register", methods=["POST"])
def auth_register():
    try:
        data = get_json_body()

        email = data.get("email", "")
        password = data.get("password", "")
        name = data.get("name", "")

        result = register_user(
            email=email,
            password=password,
            name=name,
        )

        return jsonify({
            "status": "success",
            **result,
        })

    except ValueError as e:
        return error_response(str(e), status_code=400, error="register_failed")

    except Exception as e:
        return error_response(str(e), status_code=500, error="server_error")

@app.route("/auth/login", methods=["POST"])
def auth_login():
    try:
        data = get_json_body()

        email = data.get("email", "")
        password = data.get("password", "")

        result = login_user(
            email=email,
            password=password,
        )

        return jsonify({
            "status": "success",
            **result,
        })

    except ValueError as e:
        return error_response(str(e), status_code=401, error="login_failed")

    except Exception as e:
        return error_response(str(e), status_code=500, error="server_error")

@app.route("/auth/google", methods=["POST"])
def auth_google():
    try:
        data = get_json_body()

        google_id_token = data.get("id_token", "")

        result = google_login_user(
            google_id_token=google_id_token,
        )

        return jsonify({
            "status": "success",
            **result,
        })

    except ValueError as e:
        return error_response(str(e), status_code=401, error="google_login_failed")

    except Exception as e:
        return error_response(str(e), status_code=500, error="server_error")


@app.route("/auth/me", methods=["GET"])
def auth_me():
    try:
        result = get_me(request.headers.get("Authorization"))

        return jsonify({
            "status": "success",
            **result,
        })

    except ValueError as e:
        return error_response(str(e), status_code=401, error="unauthorized")

    except Exception as e:
        return error_response(str(e), status_code=500, error="server_error")


# ================================
# Desktop Login Bridge Routes
# ================================

@app.route("/auth/desktop/start", methods=["POST"])
def auth_desktop_start():
    try:
        result = create_desktop_login_request()

        return jsonify({
            "status": "success",
            **result,
        })

    except Exception as e:
        return error_response(str(e), status_code=500, error="server_error")


@app.route("/auth/desktop/approve", methods=["POST"])
def auth_desktop_approve():
    try:
        data = get_json_body()
        rid = data.get("rid", "")

        result = approve_desktop_login_request(
            rid=rid,
            auth_header=request.headers.get("Authorization"),
        )

        return jsonify({
            "status": "success",
            **result,
        })

    except ValueError as e:
        return error_response(str(e), status_code=400, error="desktop_login_failed")

    except Exception as e:
        return error_response(str(e), status_code=500, error="server_error")


@app.route("/auth/desktop/status", methods=["GET"])
def auth_desktop_status():
    try:
        rid = request.args.get("rid", "")

        result = get_desktop_login_status(rid=rid)

        return jsonify({
            "status": "success",
            "desktop_login": result,
        })

    except ValueError as e:
        return error_response(str(e), status_code=404, error="desktop_login_not_found")

    except Exception as e:
        return error_response(str(e), status_code=500, error="server_error")

# ================================
# Subscription Routes
# ================================

@app.route("/subscription/status", methods=["GET"])
def subscription_status():
    try:
        user = get_current_user_from_header(request.headers.get("Authorization"))

        if not user:
            return error_response(
                "Login required.",
                status_code=401,
                error="unauthorized"
            )

        allowed, sub_payload = check_subscription_access(user.get("uid"))

        return jsonify({
            "status": "success",
            "allowed": allowed,
            "subscription": sub_payload,
        })

    except Exception as e:
        return error_response(str(e), status_code=500, error="server_error")


@app.route("/admin/activate-subscription", methods=["POST"])
@require_admin
def admin_activate_subscription():
    """
    Temporary manual activation endpoint.
    
    Headers:
      X-Admin-Secret: your-admin-secret
    
    Body:
      {
        "email": "user@gmail.com",
        "plan": "monthly",
        "provider": "manual",
        "payment_id": "optional",
        "amount": 25,
        "currency": "USD"
      }
    """
    try:
        data = get_json_body()

        email = data.get("email", "")
        plan = data.get("plan", "monthly")
        provider = data.get("provider", "manual")
        payment_id = data.get("payment_id")
        amount = data.get("amount")
        currency = data.get("currency", "USD")

        result = activate_subscription(
            email=email,
            plan=plan,
            provider=provider,
            payment_id=payment_id,
            amount=amount,
            currency=currency,
        )

        return jsonify({
            "status": "success",
            "message": "Subscription activated.",
            "subscription": result,
        })

    except ValueError as e:
        return error_response(str(e), status_code=400, error="activation_failed")

    except Exception as e:
        return error_response(str(e), status_code=500, error="server_error")


@app.route("/admin/subscription-by-email", methods=["GET"])
@require_admin
def admin_subscription_by_email():
    try:
        email = request.args.get("email", "")

        result = get_subscription_status_by_email(email)

        return jsonify({
            "status": "success",
            **result,
        })

    except Exception as e:
        return error_response(str(e), status_code=500, error="server_error")


# ================================
# Main Analyze Endpoint
# ================================

@app.route("/analyze", methods=["POST"])
@require_auth_and_subscription
def analyze():
    try:
        data = get_json_body()

        symbol = data.get("symbol", "BTCUSDT")
        interval = data.get("interval", "15m")
        action = data.get("action", "analyze")  # analyze, status, scan, history
        market_type = data.get("market_type", "crypto")

        result = main.run_analysis_backend(symbol, interval, action, market_type)

        return jsonify(
            make_json_safe(result)
        )

    except Exception as e:
        return jsonify({"error": str(e), "status": "failed"}), 500


# ================================
# Trade Audit Endpoints
# ================================

@app.route("/trade-audit/stats", methods=["GET"])
def trade_audit_stats():
    try:
        symbol = request.args.get("symbol")
        timeframe = request.args.get("timeframe")
        limit = int(request.args.get("limit", 1000))

        result = get_trade_audit_stats(
            db=db,
            symbol=symbol,
            timeframe=timeframe,
            limit=limit
        )

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e), "status": "failed"}), 500


@app.route("/trade-audit/trades", methods=["GET"])
def trade_audit_trades():
    try:
        symbol = request.args.get("symbol")
        timeframe = request.args.get("timeframe")
        status = request.args.get("status")
        result_filter = request.args.get("result")
        limit = int(request.args.get("limit", 100))

        result = list_trade_audit_records(
            db=db,
            symbol=symbol,
            timeframe=timeframe,
            status=status,
            result=result_filter,
            limit=limit
        )

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e), "status": "failed"}), 500


@app.route("/trade-audit/check-open", methods=["GET", "POST"])
def trade_audit_check_open():
    try:
        market_type = request.args.get("market_type", "crypto")
        limit = int(request.args.get("limit", 100))
        candle_period = request.args.get("period", "7d")

        result = check_open_audit_trades(
            db=db,
            market_type=market_type,
            limit=limit,
            candle_period=candle_period
        )

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e), "status": "failed"}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)