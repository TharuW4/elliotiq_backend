import json
import math
import re
import threading
import time
from collections.abc import Mapping

from google import genai
from . import config
import pandas as pd

# --- KEY ROTATION & HYBRID LOGIC START ---
CURRENT_KEY_INDEX = 0
_KEY_INDEX_LOCK = threading.Lock()

GEMINI_REQUEST_TIMEOUT_MS = 30_000
GEMINI_PAID_REQUEST_TIMEOUT_MS = 90_000
GEMINI_MIN_CALL_TIMEOUT_MS = 5_000
GEMINI_MAX_CALL_TIMEOUT_MS = max(
    GEMINI_REQUEST_TIMEOUT_MS,
    GEMINI_PAID_REQUEST_TIMEOUT_MS,
)
GEMINI_RETRY_BASE_DELAY_SECONDS = 0.75
GEMINI_RETRY_MAX_DELAY_SECONDS = 3.0
MAX_PROVIDER_ERROR_LENGTH = 1_000
MAX_RAW_RESPONSE_LENGTH = 100_000

_GOOGLE_API_KEY_PATTERN = re.compile(
    r"\bAIza[0-9A-Za-z_-]{25,}\b"
)


def _paid_api_key():
    key = str(
        getattr(
            config,
            "PAID_API_KEY",
            "",
        )
        or ""
    ).strip()

    return key or None


def _free_api_keys():
    raw_pool = getattr(
        config,
        "API_KEY_POOL",
        [],
    )

    if isinstance(raw_pool, str):
        raw_pool = [raw_pool]

    if not isinstance(
        raw_pool,
        (
            list,
            tuple,
            set,
        ),
    ):
        return []

    paid_key = _paid_api_key()

    clean_keys = []
    seen = set()

    for raw_key in raw_pool:
        key = str(
            raw_key or ""
        ).strip()

        if (
            not key
            or key in seen
            or key == paid_key
        ):
            continue

        seen.add(key)
        clean_keys.append(key)

    return clean_keys


def _safe_gemini_error(
    error,
):
    """
    Prevent API keys or very large provider errors from
    reaching logs, API responses or Firestore debug data.
    """

    text = str(
        error or ""
    )

    configured_keys = list(
        _free_api_keys()
    )

    paid_key = _paid_api_key()

    if paid_key:
        configured_keys.append(
            paid_key
        )

    for key in configured_keys:
        text = text.replace(
            key,
            "[REDACTED_API_KEY]",
        )

    text = _GOOGLE_API_KEY_PATTERN.sub(
        "[REDACTED_API_KEY]",
        text,
    )

    text = re.sub(
        r"[\x00-\x08\x0b\x0c\x0e-\x1f]",
        "",
        text,
    )

    return text[
        :MAX_PROVIDER_ERROR_LENGTH
    ]


def _remaining_gemini_timeout_ms(
    deadline_monotonic=None,
    maximum_timeout_ms=None,
):
    try:
        timeout_cap = int(
            maximum_timeout_ms
            if maximum_timeout_ms is not None
            else GEMINI_REQUEST_TIMEOUT_MS
        )

    except Exception:
        timeout_cap = GEMINI_REQUEST_TIMEOUT_MS

    timeout_cap = max(
        GEMINI_MIN_CALL_TIMEOUT_MS,
        min(
            GEMINI_MAX_CALL_TIMEOUT_MS,
            timeout_cap,
        ),
    )

    if deadline_monotonic is None:
        return timeout_cap

    try:
        remaining_ms = int(
            (
                float(deadline_monotonic)
                - time.monotonic()
            )
            * 1000
        )

    except Exception:
        return timeout_cap

    if remaining_ms < GEMINI_MIN_CALL_TIMEOUT_MS:
        return None

    return min(
        timeout_cap,
        remaining_ms,
    )


def _new_gemini_client(
    api_key,
    timeout_ms=None,
):
    try:
        timeout = int(
            timeout_ms
            if timeout_ms is not None
            else GEMINI_REQUEST_TIMEOUT_MS
        )

    except Exception:
        timeout = GEMINI_REQUEST_TIMEOUT_MS

    timeout = max(
        GEMINI_MIN_CALL_TIMEOUT_MS,
        min(
            GEMINI_MAX_CALL_TIMEOUT_MS,
            timeout,
        ),
    )

    return genai.Client(
        api_key=api_key,

        http_options={
            "timeout": timeout,
        },
    )


def _close_gemini_client(
    client,
):
    close_method = getattr(
        client,
        "close",
        None,
    )

    if callable(close_method):
        try:
            close_method()

        except Exception:
            pass


def get_next_free_client(
    timeout_ms=None,
):
    """
    Return the next unique, non-empty free-key client.

    The index update is protected because concurrent requests
    may execute inside the same Flask/Gunicorn worker.
    """

    global CURRENT_KEY_INDEX

    keys = _free_api_keys()

    if not keys:
        return None

    with _KEY_INDEX_LOCK:
        index = (
            CURRENT_KEY_INDEX
            % len(keys)
        )

        CURRENT_KEY_INDEX = (
            index + 1
        ) % len(keys)

        selected_key = keys[index]

    return _new_gemini_client(
        selected_key,
        timeout_ms=timeout_ms,
    )


def _free_request_attempt_limit(
    free_key_count,
):
    """
    Give transient transport failures a few recovery attempts
    even when only one free API key is configured.
    """

    try:
        key_count = max(
            int(free_key_count),
            0,
        )

    except Exception:
        key_count = 0

    if key_count <= 0:
        return 0

    return key_count


def _gemini_retry_delay_seconds(
    attempt_number,
):
    """Return a short capped exponential retry delay."""

    try:
        attempt = max(
            int(attempt_number),
            1,
        )

    except Exception:
        attempt = 1

    return min(
        GEMINI_RETRY_BASE_DELAY_SECONDS
        * (2 ** (attempt - 1)),
        GEMINI_RETRY_MAX_DELAY_SECONDS,
    )

def _sleep_before_gemini_retry(
    delay_seconds,
    deadline_monotonic=None,
):
    try:
        delay = max(
            float(delay_seconds),
            0.0,
        )

    except Exception:
        delay = 0.0

    if delay <= 0:
        return

    if deadline_monotonic is not None:
        try:
            remaining = (
                float(deadline_monotonic)
                - time.monotonic()
            )

        except Exception:
            remaining = delay

        if remaining <= 0:
            return

        delay = min(
            delay,
            remaining,
        )

    time.sleep(delay)

def generate_content_smart(
    model_name,
    prompt,
    *,
    deadline_monotonic=None,
):
    """
    Plain-text Gemini helper used by status, guidance
    and scanner functions.
    """

    selected_model = str(
        model_name or ""
    ).strip()

    prompt_text = str(
        prompt or ""
    ).strip()

    if not selected_model:
        return json.dumps(
            {
                "error": (
                    "Gemini model name is empty."
                ),
            }
        )

    if not prompt_text:
        return json.dumps(
            {
                "error": (
                    "Gemini prompt is empty."
                ),
            }
        )

    last_error = None

    free_attempt_limit = (
        _free_request_attempt_limit(
            len(
                _free_api_keys()
            )
        )
    )

    for attempt_number in range(
        1,
        free_attempt_limit + 1,
    ):
        client = None

        try:
            timeout_ms = (
                _remaining_gemini_timeout_ms(
                    deadline_monotonic
                )
            )

            if timeout_ms is None:
                last_error = (
                    "GEMINI_DEADLINE_EXCEEDED_"
                    "BEFORE_FREE_REQUEST"
                )
                break

            client = get_next_free_client(
                timeout_ms=timeout_ms
            )

            if client is None:
                break

            response = (
                client.models.generate_content(
                    model=selected_model,
                    contents=prompt_text,
                )
            )

            response_text = str(
                getattr(
                    response,
                    "text",
                    "",
                )
                or ""
            ).strip()

            if not response_text:
                raise ValueError(
                    "Gemini returned an empty text response."
                )

            return response_text

        except Exception as exc:
            last_error = (
                _safe_gemini_error(
                    exc
                )
            )

            if _is_retryable_gemini_error(
                exc
            ):
                print(
                    "Gemini free-key request failed "
                    "temporarily; retrying "
                    f"({attempt_number}/"
                    f"{free_attempt_limit})."
                )

                if attempt_number >= free_attempt_limit:
                    break

                _sleep_before_gemini_retry(
                    _gemini_retry_delay_seconds(
                        attempt_number
                    ),
                    deadline_monotonic,
                )
                continue

            print(
                "Gemini request failed: "
                f"{last_error}"
            )

            return json.dumps(
                {
                    "error": (
                        last_error
                        or "GEMINI_REQUEST_FAILED"
                    ),
                }
            )

        finally:
            _close_gemini_client(
                client
            )

    paid_key = _paid_api_key()

    if paid_key:
        client = None

        try:
            timeout_ms = (
                _remaining_gemini_timeout_ms(
                    deadline_monotonic,
                    maximum_timeout_ms=(
                        GEMINI_PAID_REQUEST_TIMEOUT_MS
                    ),
                )
            )

            if timeout_ms is None:
                return json.dumps(
                    {
                        "error": (
                            last_error
                            or (
                                "GEMINI_DEADLINE_EXCEEDED_"
                                "BEFORE_PAID_REQUEST"
                            )
                        ),
                    }
                )

            client = _new_gemini_client(
                paid_key,
                timeout_ms=timeout_ms,
            )

            response = (
                client.models.generate_content(
                    model=selected_model,
                    contents=prompt_text,
                )
            )

            response_text = str(
                getattr(
                    response,
                    "text",
                    "",
                )
                or ""
            ).strip()

            if not response_text:
                raise ValueError(
                    "Gemini paid-key response was empty."
                )

            return response_text

        except Exception as exc:
            last_error = (
                _safe_gemini_error(
                    exc
                )
            )

            print(
                "Gemini paid-key request failed: "
                f"{last_error}"
            )

            return json.dumps(
                {
                    "error": (
                        "PAID_KEY_FAILED: "
                        f"{last_error}"
                    ),
                }
            )

        finally:
            _close_gemini_client(
                client
            )

    return json.dumps(
        {
            "error": (
                last_error
                or "ALL_SYSTEMS_BUSY_NO_KEYS_AVAILABLE"
            ),
        }
    )


def _is_retryable_gemini_error(
    error,
):
    """
    Return True only for quota, temporary service
    and network failures.
    """

    error_code = None

    for attribute in (
        "code",
        "status_code",
        "http_status",
    ):
        value = getattr(
            error,
            attribute,
            None,
        )

        if callable(value):
            try:
                value = value()

            except Exception:
                value = None

        try:
            if value is not None:
                error_code = int(
                    value
                )

                break

        except Exception:
            pass

    text = str(
        error or ""
    )

    if error_code is None:
        match = re.search(
            r"(?<!\d)"
            r"(408|409|429|500|502|503|504)"
            r"(?!\d)",
            text,
        )

        if match:
            error_code = int(
                match.group(1)
            )

    if error_code in {
        408,
        409,
        429,
        500,
        502,
        503,
        504,
    }:
        return True

    lowered = (
        f"{type(error).__name__} {text}"
    ).lower()

    retryable_markers = (
        "resourceexhausted",
        "resource exhausted",
        "quota exceeded",
        "rate limit",
        "too many requests",
        "service unavailable",
        "temporarily unavailable",
        "deadline exceeded",
        "timeout",
        "timed out",
        "connection reset",
        "connection aborted",
        "connection refused",
        "remote protocol error",
        "remoteprotocolerror",
        "server disconnected",
        "without sending a response",
        "connection closed",
        "connection terminated",
        "peer closed connection",
        "incomplete chunked read",
        "unexpected eof",
        "readtimeout",
        "connecttimeout",
        "readerror",
        "connecterror",
    )

    return any(
        marker in lowered
        for marker in retryable_markers
    )


def _reject_non_finite_json(
    value,
):
    raise ValueError(
        "Gemini structured response contains "
        f"non-finite JSON value: {value}"
    )


def _ensure_json_object(
    value,
):
    if not isinstance(
        value,
        Mapping,
    ):
        raise ValueError(
            "Gemini structured response is not a JSON object."
        )

    data = dict(
        value
    )

    json.dumps(
        data,
        ensure_ascii=False,
        allow_nan=False,
    )

    return data


def _parse_structured_gemini_response(
    response,
):
    """
    Convert a Gemini structured response into a finite,
    JSON-serializable dictionary.
    """

    if response is None:
        raise ValueError(
            "Gemini returned no response object."
        )

    parsed = getattr(
        response,
        "parsed",
        None,
    )

    if isinstance(
        parsed,
        Mapping,
    ):
        return _ensure_json_object(
            parsed
        )

    if parsed is not None:
        if hasattr(
            parsed,
            "model_dump",
        ):
            return _ensure_json_object(
                parsed.model_dump()
            )

        if hasattr(
            parsed,
            "dict",
        ):
            return _ensure_json_object(
                parsed.dict()
            )

    raw_text = str(
        getattr(
            response,
            "text",
            "",
        )
        or ""
    ).strip()

    if not raw_text:
        raise ValueError(
            "Gemini returned an empty structured response."
        )

    if len(
        raw_text
    ) > MAX_RAW_RESPONSE_LENGTH:
        raise ValueError(
            "Gemini structured response exceeded "
            "the allowed raw-text size."
        )

    raw_text = raw_text.lstrip(
        "\ufeff"
    )

    if raw_text.startswith(
        "```"
    ):
        raw_text = re.sub(
            r"^```(?:json)?\s*",
            "",
            raw_text,
            flags=re.IGNORECASE,
        )

        raw_text = re.sub(
            r"\s*```$",
            "",
            raw_text,
        ).strip()

    parsed_json = json.loads(
        raw_text,

        parse_constant=(
            _reject_non_finite_json
        ),
    )

    return _ensure_json_object(
        parsed_json
    )


def _extract_gemini_usage_metadata(response):
    """
    Returns token usage metadata without exposing API keys.
    """
    usage = getattr(
        response,
        "usage_metadata",
        None,
    )

    if usage is None:
        return {}

    if isinstance(usage, dict):
        return usage

    if hasattr(usage, "model_dump"):
        try:
            dumped = usage.model_dump()

            if isinstance(dumped, dict):
                return dumped
        except Exception:
            pass

    if hasattr(usage, "dict"):
        try:
            dumped = usage.dict()

            if isinstance(dumped, dict):
                return dumped
        except Exception:
            pass

    return {
        "raw": str(usage),
    }


def _generate_content_with_diagnostics(
    client,
    *,
    model,
    contents,
    config,
    key_source,
    attempt,
    timeout_ms,
):
    started = time.monotonic()
    label = f"{key_source} attempt={attempt}"
    print(
        f"[GEMINI_CALL] START {label} "
        f"timeout_ms={timeout_ms}"
    )

    try:
        response = client.models.generate_content(
            model=model,
            contents=contents,
            config=config,
        )
    except Exception as exc:
        elapsed = time.monotonic() - started
        print(
            f"[GEMINI_CALL] ERROR {label} "
            f"elapsed_s={elapsed:.2f} "
            f"type={type(exc).__name__} "
            f"error={_safe_gemini_error(exc)!r}"
        )
        raise

    elapsed = time.monotonic() - started
    try:
        usage = _extract_gemini_usage_metadata(response)
        candidates = getattr(response, "candidates", None) or []
        finish = (
            getattr(candidates[0], "finish_reason", None)
            if candidates else None
        )
        finish = getattr(finish, "value", finish)
        print(
            f"[GEMINI_CALL] RESPONSE {label} "
            f"elapsed_s={elapsed:.2f} "
            f"finish_reason={finish!r} "
            f"input_tokens={usage.get('prompt_token_count')!r} "
            f"thinking_tokens={usage.get('thoughts_token_count')!r} "
            f"output_tokens={usage.get('candidates_token_count')!r}"
        )
    except Exception:
        print(
            f"[GEMINI_CALL] RESPONSE {label} "
            f"elapsed_s={elapsed:.2f} metadata_unavailable"
        )

    return response

def generate_structured_content_smart(
    model_name,
    prompt,
    response_json_schema,
    *,
    system_instruction=None,
    temperature=0.1,
    top_p=0.2,
    max_output_tokens=12288,
    deadline_monotonic=None,
):
    """
    Generate a strict JSON response using Gemini JSON Schema.

    Key strategy:
    1. Try each configured free key.
    2. Retry another key only for quota/network/service errors.
    3. Fall back to the configured paid key.
    4. Never expose the actual key in the result.

    Returns:
    {
        "ok": True/False,
        "data": {...} or None,
        "raw_text": "...",
        "model": "...",
        "key_source": "FREE_POOL" / "PAID_BACKUP" / None,
        "usage": {...},
        "error": None or "...",
    }
    """

    selected_model = str(
        model_name or ""
    ).strip()

    prompt_text = str(
        prompt or ""
    ).strip()

    if not selected_model:
        return {
            "ok": False,
            "data": None,
            "raw_text": "",
            "model": "",
            "key_source": None,
            "usage": {},
            "error": (
                "Gemini model name is empty."
            ),
        }

    if not prompt_text:
        return {
            "ok": False,
            "data": None,
            "raw_text": "",
            "model": selected_model,
            "key_source": None,
            "usage": {},
            "error": "Prompt is empty.",
        }

    if not isinstance(
        response_json_schema,
        dict,
    ):
        return {
            "ok": False,
            "data": None,
            "raw_text": "",
            "model": selected_model,
            "key_source": None,
            "usage": {},
            "error": (
                "response_json_schema must be "
                "a dictionary."
            ),
        }

    try:
        json.dumps(
            response_json_schema,
            ensure_ascii=False,
            allow_nan=False,
        )

        temperature_value = float(
            temperature
        )

        top_p_value = float(
            top_p
        )

        max_tokens_value = int(
            max_output_tokens
        )

        if not (
            math.isfinite(
                temperature_value
            )
            and 0.0
            <= temperature_value
            <= 2.0
        ):
            raise ValueError(
                "temperature must be between 0 and 2."
            )

        if not (
            math.isfinite(
                top_p_value
            )
            and 0.0
            <= top_p_value
            <= 1.0
        ):
            raise ValueError(
                "top_p must be between 0 and 1."
            )

        if max_tokens_value <= 0:
            raise ValueError(
                "max_output_tokens must be "
                "greater than zero."
            )

    except Exception as exc:
        return {
            "ok": False,
            "data": None,
            "raw_text": "",
            "model": selected_model,
            "key_source": None,
            "usage": {},
            "error": (
                "Invalid Gemini generation configuration: "
                + _safe_gemini_error(
                    exc
                )
            ),
        }

    generation_config = {
        "response_mime_type": "application/json",
        "response_json_schema": response_json_schema,
        "max_output_tokens": max_tokens_value,
    }

    instruction_text = str(
        system_instruction or ""
    ).strip()

    if instruction_text:
        generation_config[
            "system_instruction"
        ] = instruction_text

    last_error = None

    # Allow only one extra Gemini call when the returned
    # structured JSON is malformed, truncated or empty.
    structured_response_retry_used = False

    free_keys = _free_api_keys()

    # --------------------------------------------------
    # Try all configured FREE keys.
    # --------------------------------------------------
    if free_keys:
        attempts = 0
        max_free_attempts = len(
            free_keys
        )

        while attempts < max_free_attempts:
            client = None

            try:
                timeout_ms = (
                    _remaining_gemini_timeout_ms(
                        deadline_monotonic
                    )
                )

                if timeout_ms is None:
                    last_error = (
                        "GEMINI_DEADLINE_EXCEEDED_"
                        "BEFORE_FREE_REQUEST"
                    )
                    break

                client = get_next_free_client(
                    timeout_ms=timeout_ms
                )

                if client is None:
                    break

                attempts += 1

                response = _generate_content_with_diagnostics(
                    client,
                    model=selected_model,
                    contents=prompt_text,
                    config=generation_config,
                    key_source="FREE_POOL",
                    attempt=attempts,
                    timeout_ms=timeout_ms,
                )

                parsed_data = (
                    _parse_structured_gemini_response(
                        response
                    )
                )

                return {
                    "ok": True,
                    "data": parsed_data,
                    "raw_text": str(
                        getattr(
                            response,
                            "text",
                            "",
                        )
                        or ""
                    )[:MAX_RAW_RESPONSE_LENGTH],
                    "model": selected_model,
                    "key_source": "FREE_POOL",
                    "usage": (
                        _extract_gemini_usage_metadata(
                            response
                        )
                    ),
                    "error": None,
                }

            except Exception as exc:
                last_error = (
                    _safe_gemini_error(
                        exc
                    )
                )

                error_text_lower = str(
                    exc or ""
                ).lower()

                structured_response_failed = (
                    isinstance(
                        exc,
                        json.JSONDecodeError,
                    )
                    or (
                        "empty structured response"
                        in error_text_lower
                    )
                    or (
                        "must be a json object"
                        in error_text_lower
                    )
                )

                # Gemini occasionally returns truncated or malformed
                # structured JSON. Retry the request only once.
                if structured_response_failed:
                    if (
                        not structured_response_retry_used
                        and attempts < max_free_attempts
                    ):
                        structured_response_retry_used = True

                        print(
                            "Structured Gemini response was malformed "
                            "or incomplete. Retrying once..."
                        )

                        _sleep_before_gemini_retry(
                            0.25,
                            deadline_monotonic,
                        )
                        continue

                    break

                if _is_retryable_gemini_error(
                    exc
                ):
                    print(
                        "Structured Gemini free-key "
                        "request failed temporarily. "
                        "Retrying "
                        f"({attempts}/"
                        f"{max_free_attempts})..."
                    )

                    if attempts >= max_free_attempts:
                        break

                    _sleep_before_gemini_retry(
                        _gemini_retry_delay_seconds(
                            attempts
                        ),
                        deadline_monotonic,
                    )
                    continue

                print(
                    "Structured Gemini request failed: "
                    f"{last_error}"
                )

                return {
                    "ok": False,
                    "data": None,
                    "raw_text": "",
                    "model": selected_model,
                    "key_source": "FREE_POOL",
                    "usage": {},
                    "error": (
                        last_error
                        or "GEMINI_STRUCTURED_REQUEST_FAILED"
                    ),
                }

            finally:
                _close_gemini_client(
                    client
                )

    # --------------------------------------------------
    # Paid-key fallback.
    # --------------------------------------------------
    print(
        "Structured Gemini free keys unavailable. "
        "Switching to paid-key backup..."
    )

    paid_key = _paid_api_key()

    if paid_key:
        client = None

        try:
            timeout_ms = (
                _remaining_gemini_timeout_ms(
                    deadline_monotonic,
                    maximum_timeout_ms=(
                        GEMINI_PAID_REQUEST_TIMEOUT_MS
                    ),
                )
            )

            if timeout_ms is None:
                return {
                    "ok": False,
                    "data": None,
                    "raw_text": "",
                    "model": selected_model,
                    "key_source": None,
                    "usage": {},
                    "error": (
                        last_error
                        or (
                            "GEMINI_DEADLINE_EXCEEDED_"
                            "BEFORE_PAID_REQUEST"
                        )
                    ),
                }

            client = _new_gemini_client(
                paid_key,
                timeout_ms=timeout_ms,
            )

            response = _generate_content_with_diagnostics(
                client,
                model=selected_model,
                contents=prompt_text,
                config=generation_config,
                key_source="PAID_BACKUP",
                attempt=1,
                timeout_ms=timeout_ms,
            )

            parsed_data = (
                _parse_structured_gemini_response(
                    response
                )
            )

            print(
                "Structured Gemini request succeeded "
                "using paid-key backup."
            )

            return {
                "ok": True,
                "data": parsed_data,
                "raw_text": str(
                    getattr(
                        response,
                        "text",
                        "",
                    )
                    or ""
                )[:MAX_RAW_RESPONSE_LENGTH],
                "model": selected_model,
                "key_source": "PAID_BACKUP",
                "usage": (
                    _extract_gemini_usage_metadata(
                        response
                    )
                ),
                "error": None,
            }

        except Exception as exc:
            last_error = (
                _safe_gemini_error(
                    exc
                )
            )

            print(
                "Structured Gemini paid-key request "
                f"failed: {last_error}"
            )

            return {
                "ok": False,
                "data": None,
                "raw_text": "",
                "model": selected_model,
                "key_source": "PAID_BACKUP",
                "usage": {},
                "error": (
                    "PAID_KEY_FAILED: "
                    f"{last_error}"
                ),
            }

        finally:
            _close_gemini_client(
                client
            )

    return {
        "ok": False,
        "data": None,
        "raw_text": "",
        "model": selected_model,
        "key_source": None,
        "usage": {},
        "error": (
            last_error
            or "ALL_SYSTEMS_BUSY_NO_KEYS_AVAILABLE"
        ),
    }

# --- KEY ROTATION LOGIC END ---


# --- HELPER FUNCTIONS ---

def format_waves(waves, timeframe="Unknown", limit=None, max_age_days=None, reference_date=None):
    """
    Formats recent wave pivot points into a readable string for AI.
    Backend-controlled:
    - filters stale waves by max_age_days
    - limits number of wave points sent
    - adds summary so AI knows total_detected, shown, latest pattern/date
    """
    if not waves:
        return f"Wave Summary: timeframe={timeframe}, total_detected=0, shown=0\nNo clear structure found."

    original_total = len(waves)
    filtered = waves

    # 1) Time-window filter
    try:
        if max_age_days is not None and reference_date is not None:
            ref_date = pd.to_datetime(reference_date)
            cutoff = ref_date - pd.Timedelta(days=max_age_days)

            recent = []
            for w in waves:
                try:
                    d = pd.to_datetime(w.get("p", {}).get("date"))
                    if d >= cutoff:
                        recent.append(w)
                except Exception:
                    continue

            # If recent waves exist, use them.
            # If no recent waves found, keep original waves but summary will still show latest date.
            if recent:
                filtered = recent
    except Exception:
        filtered = waves

    # 2) Max wave-count limit
    if limit is not None:
        filtered = filtered[-limit:]

    shown_total = len(filtered)

    latest_pattern = "Unknown"
    latest_label = "?"
    latest_date = "Unknown"

    try:
        last = filtered[-1]
        latest_pattern = last.get("pattern_type", "Unknown")
        latest_label = last.get("lbl", "?")
        latest_date = str(last.get("p", {}).get("date", "Unknown"))
    except Exception:
        pass

    txt = (
        f"Wave Summary: timeframe={timeframe}, "
        f"total_detected={original_total}, shown={shown_total}, "
        f"latest_pattern={latest_pattern}, latest_label={latest_label}, latest_date={latest_date}\n"
    )

    for w in filtered:
        try:
            p = w.get("p", {})
            d = p.get("date")
            date_str = d if isinstance(d, str) else d.strftime("%Y-%m-%d %H:%M")

            pattern_type = w.get("pattern_type", "UnknownPattern")
            pivot_type = p.get("type", "pivot")
            price = _format_trade_price(p.get("price"))
            label = w.get("lbl", "?")

            txt += f"- [{pattern_type}] Wave {label} {pivot_type} at {price} on {date_str}\n"

        except Exception:
            continue

    return txt

def get_last_macro_label(waves, reference_date=None, max_age_days=365):
    """
    Returns latest macro label with stale awareness.
    If latest detected macro wave is too old compared to latest candle date,
    mark it as stale so AI does not over-weight it.
    """
    if not waves:
        return "Unknown"

    last = waves[-1]
    pattern_type = last.get("pattern_type", "UnknownPattern")
    label = last.get("lbl", "?")
    last_date = last.get("p", {}).get("date")

    try:
        if reference_date is not None and last_date is not None:
            ref_date = pd.to_datetime(reference_date)
            wave_date = pd.to_datetime(last_date)
            age_days = (ref_date - wave_date).days

            if age_days > max_age_days:
                return f"STALE {pattern_type} Wave {label} ({age_days} days old)"
    except Exception:
        pass

    return f"{pattern_type} Wave {label}"

def ensure_datetime(d):
    """Ensures the date is a comparable datetime object."""
    if isinstance(d, str):
        return pd.to_datetime(d)
    return d


def dataframe_to_csv_string(df, limit=None):
    """
    Converts DataFrame to a clean CSV-like string for AI.
    Uses one Date column only.
    """
    if df is None or df.empty:
        return "No Data Available"

    subset = df.tail(limit) if limit else df.copy()

    csv_str = "Date,Open,High,Low,Close\n"

    for idx, row in subset.iterrows():
        try:
            # If Date column exists, use it. Otherwise use dataframe index.
            date_value = row["Date"] if "Date" in subset.columns else idx
            d_str = str(date_value)

            o = _format_trade_price(row["Open"])
            h = _format_trade_price(row["High"])
            l = _format_trade_price(row["Low"])
            c = _format_trade_price(row["Close"])

            csv_str += f"{d_str},{o},{h},{l},{c}\n"

        except Exception:
            continue

    return csv_str


def _extract_float(val):
    if isinstance(val, (int, float)):
        return float(val)

    if isinstance(val, str):
        try:
            text = val.replace(",", "").strip()
            match = re.search(r"-?\d+(?:\.\d+)?", text)
            if match:
                return float(match.group(0))
        except Exception:
            return None

    return None



def _format_trade_price(value):
    """
    Dynamic price formatting.
    Keeps enough decimals for low-price coins like APE, DOGE, PEPE.
    """
    value = _extract_float(value)
    if value is None:
        return ""

    if abs(value) >= 1000:
        return f"{value:.2f}"
    elif abs(value) >= 100:
        return f"{value:.2f}"
    elif abs(value) >= 10:
        return f"{value:.3f}"
    elif abs(value) >= 1:
        return f"{value:.4f}"
    elif abs(value) >= 0.01:
        return f"{value:.4f}"
    elif abs(value) >= 0.0001:
        return f"{value:.6f}"
    else:
        return f"{value:.8f}"


def _force_wait_scenario(summary="Setup unclear. Wait for confirmation."):
    return {
        "name": "Sniper Setup",
        "trade_type": "WAIT",
        "probability": "Low",
        "summary": summary,
        "entry_zone": "",
        "targets": [],
        "invalidation": ""
    }


def _normalize_ai_review(
    data,
):
    """
    Keep the legacy backend-review path fail-closed.

    Missing, malformed or unsupported review decisions must
    become REJECT_TO_WAIT rather than accidental approval.
    """

    if not isinstance(
        data,
        dict,
    ):
        return data

    review = data.get(
        "ai_review"
    )

    if not isinstance(
        review,
        dict,
    ):
        review = {}

    allowed = {
        "APPROVE_BACKEND_PRIMARY",
        "REJECT_TO_WAIT",
    }

    decision = str(
        review.get(
            "decision"
        )
        or "REJECT_TO_WAIT"
    ).upper().strip()

    preferred_rank = review.get(
        "preferred_candidate_rank"
    )

    if decision.startswith(
        "PREFER_CANDIDATE_"
    ):
        preferred_rank = decision.replace(
            "PREFER_CANDIDATE_",
            "",
        ).strip()

        decision = "REJECT_TO_WAIT"

    if decision not in allowed:
        decision = "REJECT_TO_WAIT"

    review["decision"] = decision

    review[
        "preferred_candidate_rank"
    ] = preferred_rank

    review["reason"] = str(
        review.get("reason")
        or (
            "AI review was missing or invalid; "
            "defaulted to WAIT."
        )
    )

    review["risk_notes"] = str(
        review.get(
            "risk_notes"
        )
        or ""
    )

    review["tp_sl_notes"] = str(
        review.get(
            "tp_sl_notes"
        )
        or ""
    )

    review["is_safe_to_show"] = (
        decision
        == "APPROVE_BACKEND_PRIMARY"
    )

    data["ai_review"] = review

    return data


def get_timeframe_profile_text(user_tf, market_type="crypto"):
    """
    LEAD AI v2: Universal Timeframe Profile
    Tells AI how to respect the Trigger, Confirm, and Swing timeframes.
    """
    tf = str(user_tf or "").lower().strip()

    if tf in ["15m"]:
        return """
        TIMEFRAME PROFILE FOR 15m SETUP (DAY TRADING/SCALPING):
        - Requested timeframe (15m) is the ENTRY TRIGGER.
        - 1H is your CONFIRM timeframe (Must align for safety).
        - 4H is your SWING timeframe (Macro direction).
        - Do NOT counter-trade the 1H/4H trend unless a clear CHoCH is present.
        """
    if tf in ["1h", "60m"]:
        return """
        TIMEFRAME PROFILE FOR 1H SETUP (SWING TRADING):
        - Requested timeframe (1H) is the ENTRY TRIGGER.
        - 4H is your CONFIRM timeframe.
        - 1D is your MACRO timeframe.
        - Stop losses will naturally be wider. Ensure structural invalidation is logical.
        """
    if tf in ["4h", "240m"]:
        return """
        TIMEFRAME PROFILE FOR 4H SETUP (POSITION TRADING):
        - Requested timeframe (4H) is the ENTRY TRIGGER.
        - 1D is your CONFIRM timeframe.
        - Do not overreact to minor lower-timeframe noise. Focus on major zones.
        """
    if tf in ["1d", "d", "daily"]:
        return """
        TIMEFRAME PROFILE FOR DAILY SETUP (MACRO INVESTING):
        - Daily is the primary signal timeframe.
        - Weekly is the macro bias.
        - Signal frequency should be low, probability must be extremely high.
        """

    return f"""
    TIMEFRAME PROFILE FOR {user_tf} SETUP:
    - {user_tf} is the entry trigger.
    - Strictly follow the Dominant Trend Context for direction.
    """


def _normalize_ai_output(data):
    """
    Enforce one sniper setup only.
    If LONG/SHORT is missing required numeric fields, convert it to WAIT.
    """
    if not isinstance(data, dict):
        return {
            "macro_analysis": {
                "trend": "Unknown",
                "detailed_breakdown": "Invalid AI response.",
                "key_levels": "N/A"
            },
            "micro_analysis": {
                "wave_count_status": "Unknown",
                "wave_personality": "Unknown"
            },
            "action_advice": "Wait.",
            "trade_scenarios": [_force_wait_scenario("Invalid AI response.")]
        }

    if "macro_analysis" not in data or not isinstance(data.get("macro_analysis"), dict):
        data["macro_analysis"] = {
            "trend": "Unknown",
            "detailed_breakdown": "No macro analysis provided.",
            "key_levels": "N/A"
        }

    if "micro_analysis" not in data or not isinstance(data.get("micro_analysis"), dict):
        data["micro_analysis"] = {
            "wave_count_status": "Unknown",
            "wave_personality": "Unknown"
        }

    if "action_advice" not in data or not data.get("action_advice"):
        data["action_advice"] = "Wait."

    scenarios = data.get("trade_scenarios", [])
    if not isinstance(scenarios, list) or not scenarios:
        data["trade_scenarios"] = [_force_wait_scenario("No valid setup returned.")]
        return data

    s = scenarios[0] if isinstance(scenarios[0], dict) else {}
    s.setdefault("name", "Sniper Setup")
    s.setdefault("probability", "Low")
    s.setdefault("summary", "Setup unclear. Wait for confirmation.")
    s.setdefault("entry_zone", "")
    s.setdefault("targets", [])
    s.setdefault("invalidation", "")

    trade_type = str(s.get("trade_type", "WAIT")).upper().strip()
    if trade_type not in ["LONG", "SHORT", "WAIT"]:
        trade_type = "WAIT"

    if trade_type in ["LONG", "SHORT"]:
        raw_entry_zone = s.get("entry_zone", "")
        entry = _extract_float(raw_entry_zone)
        stop = _extract_float(s.get("invalidation"))

        entry_text = str(raw_entry_zone or "").strip()
        entry_has_range = bool(
            re.search(r"\d+(?:\.\d+)?\s*(?:-|–|to)\s*\d+(?:\.\d+)?", entry_text)
        )
        raw_targets = s.get("targets", [])
        parsed_targets = []
        for t in raw_targets:
            val = _extract_float(t)
            if val is not None:
                parsed_targets.append(val)

        if entry is None or stop is None or not parsed_targets:
            s = _force_wait_scenario("Incomplete setup data. Wait for confirmation.")
            data["action_advice"] = "Wait for clear setup."
        else:
            s["trade_type"] = trade_type

            # Actual entry used by backend = FIRST value of entry_zone/range.
            s["entry_zone"] = _format_trade_price(entry)
            s["actual_entry"] = _format_trade_price(entry)

            # Keep original AI range only as reference/debug/display.
            if entry_has_range:
                s["entry_range"] = entry_text

            s["invalidation"] = _format_trade_price(stop)
            s["targets"] = [_format_trade_price(t) for t in parsed_targets[:2]]
            if not s["targets"]:
                s = _force_wait_scenario("Targets missing. Wait for confirmation.")
    else:
        s = _force_wait_scenario(s.get("summary", "Setup unclear. Wait for confirmation."))

        current_advice = str(data.get("action_advice", "")).strip()
        if current_advice.lower() in ["", "none", "null", "nan"]:
            data["action_advice"] = "Wait for clear setup."

    if str(s.get("trade_type", "")).upper() == "WAIT":
        advice = str(data.get("action_advice", "")).lower()
        # If final output is WAIT, do not let advice sound like an approved Elliott signal.
        raw_advice = str(data.get("action_advice", "")).strip()
        if raw_advice.lower().startswith("elliott wave: yes"):
            data["action_advice"] = re.sub(
                r"^\s*elliott\s+wave\s*:\s*yes",
                "Elliott Wave: NO",
                raw_advice,
                count=1,
                flags=re.IGNORECASE,
            )

        blocked_words = [
            "buy at",
            "sell at",
            "enter at",
            "place buy",
            "place sell",
            "open long",
            "open short",
            "take long",
            "take short",
            "go long",
            "go short",
            "entry at"
        ]

        if any(word in advice for word in blocked_words):
            data["action_advice"] = "Wait for clear setup."

        s["entry_zone"] = ""
        s["targets"] = []
        s["invalidation"] = ""

    data["trade_scenarios"] = [s]
    return data


def enforce_risk_reward(ai_data, ratio=5.0):
    """
    LIGHT SANITY CHECK ONLY:
    - DO NOT heavily overwrite AI targets.
    - AI should decide:
        TP1 = exact 1.5R
        TP2 = structure-based final target
    - Backend only:
        1. validates target direction/order
        2. creates missing TP2 when only one target exists
        3. caps absurd TP2 beyond max ratio
    """
    try:
        if "trade_scenarios" not in ai_data:
            return ai_data

        scenarios = ai_data["trade_scenarios"]
        if not scenarios:
            return ai_data

        main_setup = scenarios[0]
        trade_type = str(main_setup.get("trade_type", "")).upper()

        if trade_type == "WAIT":
            return ai_data

        entry = _extract_float(main_setup.get("entry_zone"))
        stop = _extract_float(main_setup.get("invalidation"))

        if entry is None or stop is None or entry == stop:
            ai_data["trade_scenarios"] = [_force_wait_scenario("Invalid entry or stop. Wait for confirmation.")]
            return ai_data

        risk = abs(entry - stop)

        if "LONG" in trade_type:
            tp1_expected = entry + (risk * 1.5)
            tp2_cap = entry + (risk * ratio)
        elif "SHORT" in trade_type:
            tp1_expected = entry - (risk * 1.5)
            tp2_cap = entry - (risk * ratio)
        else:
            ai_data["trade_scenarios"] = [_force_wait_scenario("Invalid trade type. Wait for confirmation.")]
            return ai_data

        raw_targets = main_setup.get("targets", [])
        parsed_targets = []
        for t in raw_targets:
            val = _extract_float(t)
            if val is not None:
                parsed_targets.append(val)

        if not parsed_targets:
            ai_data["trade_scenarios"] = [_force_wait_scenario("Targets missing. Wait for confirmation.")]
            return ai_data

        if len(parsed_targets) >= 2:
            tp1 = parsed_targets[0]
            tp2 = parsed_targets[1]
        else:
            # If AI returns only one target, treat it as TP2 and create TP1 from 1.5R
            tp1 = tp1_expected
            tp2 = parsed_targets[0]

        if "LONG" in trade_type:
            # TP1 must be above entry
            tp1 = tp1_expected

            # If AI accidentally made TP2 too close / below TP1, push it farther
            if tp2 <= tp1:
                tp2 = max(tp1 + (risk * 0.5), tp2)

            # Final cap
            if tp2 > tp2_cap:
                tp2 = tp2_cap

        else:  # SHORT
            # TP1 must always be exact 1.5R
            tp1 = tp1_expected

            # If AI accidentally made TP2 too close / above TP1, push it farther down
            if tp2 >= tp1:
                tp2 = min(tp1 - (risk * 0.5), tp2)

            # Final cap
            if tp2 < tp2_cap:
                tp2 = tp2_cap

        main_setup["targets"] = [_format_trade_price(tp1), _format_trade_price(tp2)]

        # Keep backend actual entry as first value only.
        main_setup["entry_zone"] = _format_trade_price(entry)
        main_setup["actual_entry"] = _format_trade_price(entry)

        main_setup["invalidation"] = _format_trade_price(stop)

    except Exception as e:
        print(f"RR Adjustment Logic Failed: {e}")

    return ai_data


# --- STATUS UPDATE (SIMPLIFIED FOR BEGINNERS) ---
def generate_status_update(scenario, df_recent):
    has_pool = hasattr(config, "API_KEY_POOL") and config.API_KEY_POOL
    has_paid = hasattr(config, "PAID_API_KEY") and config.PAID_API_KEY

    if not has_pool and not has_paid:
        return "AI Unreachable"

    try:
        current_price = df_recent.iloc[-1]["Close"]
        price_action = ""
        for _, row in df_recent.tail(15).iterrows():
            price_action += f"C:{_format_trade_price(row['Close'])} "

        prompt = f"""
        You are a Trading Mentor active on a live trade.
        SCENARIO: {scenario['trade_type']} on {scenario.get('name', 'Setup')}
        ENTRY: {scenario.get('entry_zone', scenario.get('entry', '0'))}
        TARGETS: {scenario.get('targets', [scenario.get('target', '0')])}
        STOP: {scenario.get('invalidation', scenario.get('stop_loss', '0'))}
        CURRENT PRICE: {current_price}
        RECENT CANDLES: {price_action}

        TASK:
        Provide an extremely SHORT, SIMPLE status update for a beginner.
        Max 15 words.
        Example:
        "Price is moving well, hold for target."
        "Momentum slowing, secure profits."
        "Limit order still pending."
        """

        response_text = generate_content_smart(config.MODEL_FAST, prompt)

        if '{"error"' in response_text:
            return "Market data analyzed. Monitor key levels."

        return response_text.strip()
    except Exception:
        return "Market data analyzed. Monitor key levels."

def generate_trade_guidance_advice(scenario, guidance_facts, guidance_context_text):
    """
    AI explanation layer for existing locked/global trade guidance.

    Backend owns facts and hard status.
    AI only writes user-friendly advice.
    AI must NOT create/modify entry, SL, TP1, TP2.
    """
    has_pool = hasattr(config, "API_KEY_POOL") and config.API_KEY_POOL
    has_paid = hasattr(config, "PAID_API_KEY") and config.PAID_API_KEY

    if not has_pool and not has_paid:
        return {
            "ai_guidance_available": False,
            "ai_guidance": guidance_facts.get(
                "user_guidance",
                "Market data analyzed. Follow original TP/SL plan."
            ),
            "ai_guidance_action": guidance_facts.get("action", "HOLD"),
            "ai_risk_level": guidance_facts.get("risk_level", "UNKNOWN"),
            "source": "BACKEND_FALLBACK_NO_AI_KEY",
        }

    try:
        scenario = scenario or {}
        guidance_facts = guidance_facts or {}
        facts = guidance_facts.get("facts") or {}

        prompt = f"""
You are a trade mentor giving advice for an EXISTING GLOBAL SIGNAL.

IMPORTANT:
- The backend already calculated the trade status and risk facts.
- You MUST use backend facts as truth.
- You MUST NOT create a new trade.
- You MUST NOT change entry, stop loss, TP1, or TP2.
- You MUST NOT suggest a new entry, new SL, or new target.
- You are only writing user guidance for the existing signal.

EXISTING SIGNAL:
Symbol/Name: {scenario.get("symbol", "")} {scenario.get("name", "")}
Side: {scenario.get("trade_type") or scenario.get("side")}
Entry: {scenario.get("entry") or scenario.get("entry_zone")}
Stop Loss: {scenario.get("stop_loss") or scenario.get("invalidation")}
Targets: {scenario.get("targets") or [scenario.get("target")]}
Current Scenario Status: {scenario.get("status", "")}
Original Summary: {scenario.get("summary", "")}

BACKEND HARD GUIDANCE:
Guidance Status: {guidance_facts.get("guidance_status", "")}
Risk Level: {guidance_facts.get("risk_level", "")}
Backend Action: {guidance_facts.get("action", "")}
Danger Score: {guidance_facts.get("danger_score", 0)}
Danger Flags: {guidance_facts.get("danger_flags", [])}
Backend Advice: {guidance_facts.get("backend_advice", "")}
Backend User Guidance: {guidance_facts.get("user_guidance", "")}
Management Mode: {guidance_facts.get("management_mode", "")}
TP2 Probability: {guidance_facts.get("tp2_probability", "")}
Reversal Risk: {guidance_facts.get("reversal_risk", "")}
Profit Action: {guidance_facts.get("profit_action", "")}
SL Management: {guidance_facts.get("sl_management", "")}
Short Action Text: {guidance_facts.get("short_action_text", "")}
Exit Trigger: {guidance_facts.get("exit_trigger", "")}

BACKEND FACTS:
{guidance_context_text}

CRITICAL RULES:
1. Backend facts are the source of truth. Follow Management Mode, Profit Action, SL Management, TP2 Probability, Reversal Risk, and Exit Trigger.
2. Start the advice with the backend Short Action Text when available.
3. If Guidance Status is TP1_PARTIAL_WIN, say TP1 profit was secured and the remaining runner later closed at SL. Treat it as a closed partial win, not a loss.
4. If Guidance Status is TP2_HIT or TP2 Hit is True, say the main target is completed.
5. If backend says CLOSE_CONFIRMED or SL_HIT_CONFIRMED, say the trade is invalidated/closed. Do not say hold.
6. If TP1 Hit is False, clearly say TP1 is not confirmed yet. Do not suggest partial profit.
7. If Management Mode is HOLD_FOR_TP2, say TP1 partial profit is secured and only the remaining runner should be held toward TP2.
8. If Management Mode is HOLD_WITH_CAUTION, say TP1 partial is secured and only the runner should be held with caution.
9. If Management Mode is PROTECT_PROFIT, say TP1 partial is secured and the remaining runner needs defensive management. Do not mention any fixed percentage.
10. If Management Mode is DEFENSIVE_RUNNER_TO_MAGNET_TEST, say TP1 partial is secured, do not add, and manage only the remaining runner around the backend exit trigger.
11. If Management Mode is CLOSE_FULL, say close the remaining runner after TP1. Do not imply that the original full position is still open.
12. Do NOT advise automatic breakeven just because TP1 is hit.
13. Do NOT tell the user to move SL to entry unless backend SL Management explicitly says so.
14. If SL Management is STRUCTURE_TRAIL_ONLY, say trail only by market structure.
15. If Entry Hit is False and backend action is WAIT_FOR_ENTRY/WAIT_ENTRY, say the limit order is still pending and do not chase.
16. If backend says MISSED / DO_NOT_CHASE, clearly say the setup is missed and should not be chased.
17. Keep advice practical for a beginner.
18. Keep advice short: maximum 2 short sentences.
19. Do not mention hidden calculations, internal code, or percentages that backend did not provide.

OUTPUT JSON ONLY:
{{
  "ai_guidance_available": true,
  "ai_guidance_action": "WAIT_ENTRY/HOLD/HOLD_TO_TP1/HOLD_FOR_TP2/HOLD_WITH_CAUTION/PROTECT_PROFIT/DEFENSIVE_RUNNER_TO_MAGNET_TEST/DO_NOT_CHASE/CLOSE_FULL/CLOSE_CONFIRMED/PARTIAL_WIN_CONFIRMED/TAKE_PROFIT_CONFIRMED",
  "ai_risk_level": "LOW/MEDIUM/HIGH/CLOSED",
  "ai_guidance": "Maximum 2 short sentences. Start with ACTION. Follow backend Management Mode exactly.",
  "key_warning": "One short warning if needed, otherwise empty string.",
  "should_user_chase": false,
  "can_hold_for_tp2": true,
  "breakeven_note": "Explain whether breakeven is appropriate or not, based only on backend facts."
}}
"""

        raw_text = generate_content_smart(config.MODEL_FAST, prompt)

        if not raw_text or '{"error"' in str(raw_text):
            return {
                "ai_guidance_available": False,
                "ai_guidance": guidance_facts.get(
                    "user_guidance",
                    "Market data analyzed. Follow original TP/SL plan."
                ),
                "ai_guidance_action": guidance_facts.get("action", "HOLD"),
                "ai_risk_level": guidance_facts.get("risk_level", "UNKNOWN"),
                "source": "BACKEND_FALLBACK_AI_ERROR",
            }

        cleaned_text = str(raw_text).replace("```json", "").replace("```", "").strip()
        match = re.search(r"\{.*\}", cleaned_text, re.DOTALL)

        if not match:
            return {
                "ai_guidance_available": False,
                "ai_guidance": guidance_facts.get(
                    "user_guidance",
                    "Market data analyzed. Follow original TP/SL plan."
                ),
                "ai_guidance_action": guidance_facts.get("action", "HOLD"),
                "ai_risk_level": guidance_facts.get("risk_level", "UNKNOWN"),
                "source": "BACKEND_FALLBACK_NO_JSON",
            }

        data = json.loads(match.group(0))

        # Safety normalization
        allowed_actions = {
            "NO_ACTION",
            "WAIT",
            "WAIT_ENTRY",
            "WAIT_FOR_ENTRY",
            "HOLD",
            "HOLD_TO_TP1",
            "HOLD_FOR_TP2",
            "HOLD_WITH_CAUTION",
            "TAKE_PARTIAL",
            "TAKE_PARTIAL_HOLD_TP2",
            "PROTECT_PROFIT",
            "DEFENSIVE_RUNNER_TO_MAGNET_TEST",
            "REDUCE_OR_CLOSE",
            "DO_NOT_CHASE",
            "CANCEL_PENDING",
            "PARTIAL_WIN_CONFIRMED",
            "CLOSE_CONFIRMED",
            "SL_HIT_CONFIRMED",
            "TAKE_PROFIT_CONFIRMED",
            "CLOSE_FULL",
        }

        action = str(data.get("ai_guidance_action") or guidance_facts.get("action") or "HOLD").upper()
        if action not in allowed_actions:
            action = str(guidance_facts.get("action") or "HOLD").upper()
            if action not in allowed_actions:
                action = "HOLD"

        risk = str(data.get("ai_risk_level") or guidance_facts.get("risk_level") or "UNKNOWN").upper()
        if risk not in ["LOW", "MEDIUM", "HIGH", "CLOSED", "UNKNOWN"]:
            risk = str(guidance_facts.get("risk_level") or "UNKNOWN").upper()

        ai_guidance = str(data.get("ai_guidance") or "").strip()
        if not ai_guidance:
            ai_guidance = guidance_facts.get(
                "user_guidance",
                "Market data analyzed. Follow original TP/SL plan."
            )

        # Hard override safety:
        # If backend says TP2 hit, AI cannot say keep holding.
        if guidance_facts.get("guidance_status") == "TP2_HIT":
            action = "TAKE_PROFIT_CONFIRMED"
            risk = "CLOSED"
            ai_guidance = guidance_facts.get(
                "user_guidance",
                "TP2 hit. This trade idea has completed its main target."
            )

        # Hard override safety:
        # TP1 was secured and the remaining runner later stopped.
        if guidance_facts.get("guidance_status") == "TP1_PARTIAL_WIN":
            action = "PARTIAL_WIN_CONFIRMED"
            risk = "CLOSED"

            ai_guidance = guidance_facts.get(
                "user_guidance",
                (
                    "ACTION: PARTIAL WIN CLOSED. "
                    "TP1 profit was secured and the remaining "
                    "runner later hit SL."
                ),
            )

        # Hard override safety:
        # If backend says SL hit, AI cannot say hold.
        if guidance_facts.get("guidance_status") == "CLOSE_CONFIRMED":
            action = "CLOSE_CONFIRMED"
            risk = "CLOSED"
            if "hold" in ai_guidance.lower():
                ai_guidance = guidance_facts.get(
                    "user_guidance",
                    "SL hit confirmed. This setup is invalidated/closed."
                )

        # Hard override safety:
        # If backend says close after TP1 reversal,
        # only the remaining runner is being closed.
        if guidance_facts.get("management_mode") == "CLOSE_FULL":
            action = "CLOSE_FULL"
            risk = "HIGH"

            ai_guidance = guidance_facts.get(
                "user_guidance",
                (
                    "ACTION: CLOSE REMAINING RUNNER. "
                    "TP1 partial profit was secured, but reversal "
                    "is now confirmed. Do not wait for TP2."
                ),
            )

        # Hard override safety:
        # Every open post-TP1 mode manages only the remaining runner.
        runner_management_modes = {
            "HOLD_FOR_TP2",
            "HOLD_WITH_CAUTION",
            "PROTECT_PROFIT",
            "DEFENSIVE_RUNNER_TO_MAGNET_TEST",
        }

        management_mode = str(
            guidance_facts.get("management_mode")
            or ""
        ).upper().strip()

        if management_mode in runner_management_modes:
            backend_action = str(
                guidance_facts.get("action")
                or management_mode
            ).upper().strip()

            if backend_action in allowed_actions:
                action = backend_action

            risk = guidance_facts.get(
                "risk_level",
                risk,
            )

            lower_text = ai_guidance.lower()

            blocked_full_position_phrases = (
                "do not take partial",
                "without taking partial",
                "hold the full position",
                "no partial",
            )

            has_partial_policy = (
                "partial" in lower_text
            )

            has_runner_policy = any(
                marker in lower_text
                for marker in (
                    "runner",
                    "remaining",
                    "the rest",
                )
            )

            if (
                not has_partial_policy
                or not has_runner_policy
                or any(
                    phrase in lower_text
                    for phrase in blocked_full_position_phrases
                )
            ):
                ai_guidance = guidance_facts.get(
                    "user_guidance",
                    (
                        "ACTION: SECURE TP1 PARTIAL, MANAGE RUNNER. "
                        "Keep only the remaining runner active and "
                        "follow the backend exit trigger."
                    ),
                )

        data["ai_guidance_available"] = True
        data["ai_guidance_action"] = action
        data["ai_risk_level"] = risk
        data["ai_guidance"] = ai_guidance
        data["source"] = "AI_GUIDANCE_WITH_BACKEND_FACTS"

        return data

    except Exception as e:
        return {
            "ai_guidance_available": False,
            "ai_guidance": guidance_facts.get(
                "user_guidance",
                "Market data analyzed. Follow original TP/SL plan."
            ) if isinstance(guidance_facts, dict) else "Market data analyzed. Follow original TP/SL plan.",
            "ai_guidance_action": guidance_facts.get("action", "HOLD") if isinstance(guidance_facts, dict) else "HOLD",
            "ai_risk_level": guidance_facts.get("risk_level", "UNKNOWN") if isinstance(guidance_facts, dict) else "UNKNOWN",
            "source": "BACKEND_FALLBACK_EXCEPTION",
            "error": str(e),
        }


# --- MAIN ANALYSIS FUNCTION ---
def generate_mtf_analysis(
    symbol,
    macro_waves,
    swing_waves,
    micro_waves,
    df_micro,
    df_macro,
    user_tf,
    user_plan="Free",
    locked_scenario=None,
    market_bias="Analyze neutrally",
    market_type="crypto",
    cross_tf_context="",
    market_context_text="",
    confirm_1h_waves=None,
    df_confirm_1h=None,
    df_swing=None
):
    has_pool = hasattr(config, "API_KEY_POOL") and config.API_KEY_POOL
    has_paid = hasattr(config, "PAID_API_KEY") and config.PAID_API_KEY

    if not has_pool and not has_paid:
        return json.dumps({"error": "No API Keys Configured"})

    if user_plan == "Free":
        return json.dumps({
            "macro_analysis": {"trend": "LOCKED", "detailed_breakdown": "Upgrade to Pro", "key_levels": "---"},
            "micro_analysis": {"wave_count_status": "Locked", "wave_personality": "Locked"},
            "trade_scenarios": []
        })

    selected_model = config.MODEL_FAST if locked_scenario else config.MODEL_SMART

    try:
        current_price = df_micro.iloc[-1]["Close"]
        timeframe_profile_text = get_timeframe_profile_text(user_tf, market_type)

        # --- 1. PREPARE MACRO CONTEXT ---
        macro_csv_data = "No Macro Data"
        if macro_waves and df_macro is not None and not df_macro.empty:
            try:
                last_wave_date = ensure_datetime(macro_waves[-1]["p"]["date"])
                macro_subset = df_macro[df_macro.index >= last_wave_date]
                if len(macro_subset) < 30:
                    macro_subset = df_macro.tail(30)
                macro_csv_data = dataframe_to_csv_string(macro_subset, limit=None)
            except Exception as e:
                print(f"Macro Data Error: {e}")
                macro_csv_data = dataframe_to_csv_string(df_macro, limit=100)

        # --- 2. PREPARE MICRO CONTEXT ---
        micro_csv_data = dataframe_to_csv_string(df_micro, limit=500)
        # --- 2B. PREPARE 1H CONFIRMATION CONTEXT ---
        confirm_1h_csv_data = "No 1H confirmation data supplied."

        if df_confirm_1h is not None and not df_confirm_1h.empty:
            confirm_1h_csv_data = dataframe_to_csv_string(df_confirm_1h, limit=300)


        # --- 2C. REFERENCE DATES FOR WAVE FRESHNESS ---
        macro_ref_date = df_macro.index[-1] if df_macro is not None and not df_macro.empty else None
        micro_ref_date = df_micro.index[-1] if df_micro is not None and not df_micro.empty else None
        confirm_1h_ref_date = df_confirm_1h.index[-1] if df_confirm_1h is not None and not df_confirm_1h.empty else None
        swing_ref_date = df_swing.index[-1] if df_swing is not None and not df_swing.empty else macro_ref_date

        last_macro_label = get_last_macro_label(
            macro_waves,
            reference_date=macro_ref_date,
            max_age_days=365
        )


        # --- 3. FORMAT WAVE STRUCTURES ---
        macro_structure_text = format_waves(
            macro_waves,
            timeframe="1D",
            limit=40,
            max_age_days=365,
            reference_date=macro_ref_date
        )

        swing_structure_text = format_waves(
            swing_waves,
            timeframe="4H",
            limit=30,
            max_age_days=90,
            reference_date=swing_ref_date
        )

        confirm_1h_structure_text = format_waves(
            confirm_1h_waves,
            timeframe="1H",
            limit=25,
            max_age_days=30,
            reference_date=confirm_1h_ref_date
        )

        micro_structure_text = format_waves(
            micro_waves,
            timeframe=str(user_tf),
            limit=20,
            max_age_days=7,
            reference_date=micro_ref_date
        )

        market_context_note = ""
        if market_type == "forex":
            market_context_note = """
            *** FOREX MARKET ALERT ***
            - This is a Fiat Pair or Commodity (Gold/Oil).
            - Market closes on Weekends (expect gaps).
            - High sensitivity to London/New York session overlaps.
            - Volatility is measured in Pips.
            - Consider USD Index (DXY) correlation implicitly.
            """

        base_context = f"""
        CONTEXT:
        Symbol: {symbol}
        Market Type: {market_type.upper()}
        {market_context_note}

        --- BACKEND TIMEFRAME RULE PROFILE ---
        {timeframe_profile_text}

        --- ALGORITHMIC WAVE ANALYSIS (GUIDE) ---

        IMPORTANT ELLIOTT PATTERN TYPE GUIDE:
        - [Impulse] means bullish 1-2-3-4-5 structure: Up, Down, Up, Down, Up.
        - [BearishImpulse] means bearish 1-2-3-4-5 structure: Down, Up, Down, Up, Down.
        - [Correction] means bullish A-B-C corrective movement (A-Down, B-Up, C-Down).
        - [BearishCorrection] means bearish A-B-C corrective movement (A-Up, B-Down, C-Up).
        - IMPORTANT: [BearishImpulse] and [BearishCorrection] are STRICTLY BEARISH signals.
        - Do NOT treat Bearish labels as bullish.
        [MACRO WAVES DETECTED - Daily]:
        {macro_structure_text}

        [SWING WAVES DETECTED - 4H]:
        {swing_structure_text}

        [1H CONFIRMATION WAVES DETECTED]:
        {confirm_1h_structure_text}

        [MICRO WAVES DETECTED - {user_tf}]:
        {micro_structure_text}

        Last Macro Label detected: **{last_macro_label}**.

        --- DATA LAYER A: MACRO STRUCTURE (Daily 1D) ---
        MACRO CSV (Daily):
        {macro_csv_data}

        --- DATA LAYER B: 1H CONFIRMATION STRUCTURE ---
        1H CONFIRMATION CSV:
        {confirm_1h_csv_data}

        --- DATA LAYER C: MICRO STRUCTURE ({user_tf}) ---
        MICRO CSV ({user_tf}):
        {micro_csv_data}

        --- EXISTING SETUPS ON OTHER TIMEFRAMES ---
        The following are ACTIVE Locked trades on different timeframes for this coin.
        Use them for CONFLUENCE. Do NOT contradict them unless there is a strong reversal signal.
        DATA: {cross_tf_context if cross_tf_context else "None"}

        --- MARKET STRUCTURE / FIB / ATR / VOLUME CONTEXT ---
        {market_context_text if market_context_text else "No additional market context available."}

        CURRENT PRICE: {current_price}
        """

        if locked_scenario:
            prompt = f"""
            Act as an expert Elliott Wave analyst and live trade manager.

            {base_context}

            LOCKED SCENARIO (ACTIVE TRADE): {json.dumps(locked_scenario)}

            TASK:
            1. Validate whether the LOCKED SCENARIO is still valid using macro structure, micro structure, market context, and recent price action.
            2. Respect and monitor the locked scenario unless it is clearly invalidated.
            3. Do NOT generate a new trade while the locked scenario is active.
            4. Only update the status/advice for the locked scenario.
            5. Provide a VERY SHORT beginner-friendly action advice.

            STRICT RULES:
            1. Return ONLY ONE scenario in trade_scenarios.
            2. The returned scenario must represent the LOCKED SCENARIO only, not a new trade.
            3. Use OLD SCHEMA ONLY: entry_zone, targets, invalidation.
            4. Keep the original locked entry_zone, targets, and invalidation unless the scenario is clearly invalidated.
            5. If the locked scenario is still valid, action_advice must say one of:
                - "Limit order still pending."
                - "Hold position."
                - "TP1 hit. Secure partial profit and move SL near entry."
                - "Protect remaining profit or trail SL tighter."
                - "Monitor closely."
            6. If active entry has not been reached yet, action_advice must say "Limit order still pending."
            7. If price moved away from entry before fill, action_advice must say "No chase. Entry missed."
            8. If current price is closer to TP1 than entry before fill, action_advice must be "No chase. Entry missed."
            9. If locked scenario is invalidated, clearly say invalidated in summary.
            10. Keep action_advice extremely short and beginner friendly.
            11. Summary must be concise and professional.
            12. Use the 1H confirmation layer to judge whether the locked scenario still has immediate trend support.
            13. If TP1 has already been hit, action_advice must clearly say to secure partial profit and move SL near entry/breakeven.
            14. If price is between TP1 and TP2 but momentum weakens or reversal signs appear, action_advice should say to protect remaining profit or trail SL tighter.
            15. If reversal risk is strong after TP1, advise closing remaining position or securing most profit.
            16. Avoid vague advice like "Secure profits" alone. Be specific and beginner-friendly.

            OUTPUT JSON ONLY:
            {{
                "macro_analysis": {{
                    "trend": "Bullish/Bearish/Sideways",
                    "detailed_breakdown": "Short macro update on the larger structure.",
                    "key_levels": "Important support and resistance levels."
                }},
                "micro_analysis": {{
                    "wave_count_status": "Short update on immediate wave condition.",
                    "wave_personality": "Impulsive/Corrective/Unclear"
                }},
                "action_advice": "VERY SHORT beginner-friendly advice.",
                "trade_scenarios": [
                    {{
                        "name": "Sniper Setup",
                        "trade_type": "LONG/SHORT/WAIT",
                        "probability": "High/Medium/Low",
                        "summary": "Locked scenario status update only. No new trade generated.",
                        "entry_zone": "SPECIFIC_PRICE_OR_RANGE_FIRST_VALUE_IS_ENTRY",
                        "targets": ["FIRST_TARGET_PRICE", "SECOND_TARGET_PRICE"],
                        "invalidation": "STOP_LOSS_PRICE"
                    }}
                ]
            }}
            """
        else:
            prompt = f"""
            Act as an expert Elliott Wave analyst, setup reviewer, and beginner-friendly trade mentor.

            {base_context}

            IMPORTANT BACKEND-FIRST + AI-REVIEWER ARCHITECTURE:
            The backend has already analyzed trend regime, built structured context,
            generated candidate entries, applied hard validation, and selected ONE primary setup
            in the section:
            --- BACKEND SETUP PLANNER ---

            Your job is NOT to create a trade.
            Your job is to REVIEW the backend-selected setup and explain it.

            YOU MAY ONLY CHOOSE ONE REVIEW DECISION:
            1. APPROVE_BACKEND_PRIMARY
               - Use when the selected backend setup is logical, realistic, and aligned.
            2. REJECT_TO_WAIT
               - Use when the selected backend setup is unsafe, too early, too late, wrong-side,
                 has unrealistic SL/TP, CHoCH conflict, range/chop conflict, or lacks real confluence.

            Do NOT choose alternate candidates.
            Do NOT output PREFER_CANDIDATE_N.
            Candidate preferences are advisory only and must not change final entry/SL/TP.

            ABSOLUTE HARD RULES:
            1. You MUST NOT create new entry, stop loss, or target prices.
            2. You MUST NOT modify entry_zone, targets, invalidation, trade_type, side, or order direction.
            3. You MUST NOT choose a side blocked by backend_trend_regime.
            4. You MUST NOT invent a trade if backend planner says WAIT / no valid setup.
            5. If Dominant Trend says CHoCH YES or Preferred Side WAIT, strongly prefer REJECT_TO_WAIT unless the backend explicitly explains a valid exception.
            6. If both support and resistance rejection are confirmed, treat the market as choppy/rangy and prefer WAIT unless setup is at a clean edge.
            7. If SL is too tight, too wide, or not beyond real structure, set ai_review.decision = REJECT_TO_WAIT.
            8. If TP2 is unrealistically far beyond nearby S/R, visible extensions, or realistic ATR range, set ai_review.decision = REJECT_TO_WAIT or mention TP/SL concern.
            9. If a lower-timeframe setup is earlier/weaker than the 1H/HTF entry zone, prefer WAIT or a better backend candidate.
            10. If selected setup is LIMIT_ORDER, action_advice must say wait for the exact backend entry. Do not chase.
            11. Prefer Wave 2 / Wave ii / Wave B pullback continuation only when the backend setup and Fib/S/R context support it.
            12. Avoid top/bottom reversal language unless backend explicitly selected a reversal.
            13. Keep advice short, practical, and beginner-friendly.

            REVIEW CHECKLIST BEFORE APPROVAL:
            - Direction: Does selected side match HTF / trend_regime / dominant trend?
            - Entry: Is entry at a real Fib/S/R/EMA/retest zone, not random?
            - SL: Is SL beyond real structure and within realistic ATR/risk limits?
            - TP2: Is TP2 reachable based on S/R, visible extensions, and ATR realism?
            - Timing: Is current price too close to TP1 or already moved too far?
            - Timeframe hierarchy: For 15m, does entry respect 1H/4H zone?

            ELLIOTT WAVE REVIEW RULES:
            - If BACKEND SETUP PLANNER shows "Elliott Wave: YES" or "ELLIOTT WAVE SETUP CONFIRMATION",
              treat it as a higher-priority setup than normal EMA/Fib/S/R confluence.
            - If an Elliott Wave template is confirmed, prefer APPROVE_BACKEND_PRIMARY unless
              SL is clearly invalid, RR is poor, price already chased too far, or CHoCH/hard conflict exists.
            - For Wave 2 -> Wave 3 setups, explain that the setup is trying to enter the Wave 2 pullback
              and target the next Wave 3 continuation.
            - For Wave 4 -> Wave 5 setups, explain that the setup is trying to enter the Wave 4 pullback
              and target the final Wave 5 continuation.
            - If Elliott Wave is not confirmed, say the setup is a standard backend confluence setup only.
            - Only say "Elliott Wave: YES" if the FINAL backend-selected setup has elliott_confirmed=true or setup_source=ELLIOTT_WAVE_TEMPLATE.
            - If an Elliott candidate appears in Candidate Debug but is rejected, you MUST say "Elliott Wave: NO".
            - Never mark Elliott Wave YES just because Candidate Debug contains an Elliott candidate.
            - Backend may provide Wave 3 Status as ANTICIPATION or CONFIRMED.
            - ANTICIPATION means a valid Wave 1-2 template exists, but breakout/breakdown confirmation is not proven yet.
            - CONFIRMED means the backend detected candle-body breakout/breakdown and momentum confirmation.
            - If Wave 3 Status is ANTICIPATION, explain it as an early Wave 3 setup, not a fully confirmed Wave 3.
            - If Wave 3 Status is CONFIRMED, explain it as a stronger confirmed Wave 3 continuation setup.
            - If Planner Status is WAIT_FOR_RECLAIM, WAIT_FOR_BREAKDOWN, WAIT_FOR_RETEST_HOLD, or WAIT_FOR_RETEST_REJECTION, do NOT call it a trade signal. Return WAIT and explain the required break/retest/hold/rejection condition.
            - If TP1 wall is very close to entry and backend still selected LIMIT_ORDER, mention it only as a caution, not a rejection.

            MTF ELLIOTT CONFIRMATION RULES:
            - Backend may provide mtf_elliott_alignment, mtf_wave_side, mtf_wave_phase, mtf_confirmation_score, and mtf_confirmation_reason.
            - Treat these fields as backend-owned confirmation data. Do NOT invent or modify them.
            - STRONG means the local Elliott setup aligns well with higher-timeframe Elliott wave context.
            - MEDIUM means acceptable higher-timeframe Elliott support with normal risk control.
            - WEAK means caution. Do not reject by yourself unless backend action is WAIT or backend candidate is already rejected.
            - CONFLICT means the local Elliott setup conflicts with higher-timeframe Elliott structure.
            - If mtf_elliott_alignment is CONFLICT, do NOT approve a trade. Use REJECT_TO_WAIT unless backend already returned WAIT.
            - In micro_analysis.wave_count_status, mention MTF alignment and score when Elliott Wave is approved.
            - In action_advice, if Elliott Wave is YES, include MTF status briefly, for example: "MTF: STRONG (82)".
            - Never mark MTF as STRONG/MEDIUM unless backend context explicitly says so.

            VERY IMPORTANT:
            Backend will ignore any invented levels. If you output fake levels, they will be rejected.
            If you approve, copy backend-selected setup values exactly from BACKEND SETUP PLANNER.
            If you reject, return a WAIT scenario and explain the reason briefly.

            OUTPUT JSON ONLY:
            {{
                "macro_analysis": {{
                    "trend": "Bullish/Bearish/Sideways based on backend trend regime and dominant trend context",
                    "detailed_breakdown": "Brief explanation of the larger trend and whether the backend side is safe.",
                    "key_levels": "Mention only important support/resistance/Fib/EMA levels already present in the context."
                }},
                "micro_analysis": {{
                    "wave_count_status": "If backend confirms Elliott, mention the exact Wave template, Wave 3 Status, and MTF Elliott alignment/score if available, e.g. 'Wave 2 -> Wave 3 ANTICIPATION | MTF MEDIUM 74' or 'Wave 2 -> Wave 3 CONFIRMED | MTF STRONG 88'. If backend rejected the Elliott candidate, say the Elliott idea exists but is not an approved trade. Otherwise explain the immediate pullback, continuation, range, or unclear phase.",
                    "wave_personality": "Impulsive/Corrective/Range/Unclear"
                }},
                "ai_review": {{
                    "decision": "APPROVE_BACKEND_PRIMARY or REJECT_TO_WAIT",
                    "preferred_candidate_rank": null,
                    "reason": "One concise reason for the review decision.",
                    "risk_notes": "Comment on SL realism. Mention if SL is valid/too tight/too wide.",
                    "tp_sl_notes": "Comment on TP2 realism and whether target respects structure.",
                    "is_safe_to_show": true
                }},
                "action_advice": "Short beginner-friendly advice. Start with 'Elliott Wave: YES' only if the FINAL backend-selected setup has elliott_confirmed=true or setup_source=ELLIOTT_WAVE_TEMPLATE. Also mention Wave 3 Status and MTF alignment/score if available, e.g. 'MTF: STRONG (82)' or 'MTF: MEDIUM (74)'. If LIMIT_ORDER, include the exact backend entry price. If Planner Status is WAIT_FOR_RECLAIM/WAIT_FOR_BREAKDOWN/WAIT_FOR_RETEST_HOLD/WAIT_FOR_RETEST_REJECTION or backend rejected the setup, start with 'Elliott Wave: NO' and say WAIT with the backend reason.",
                "trade_scenarios": [
                    {{
                        "name": "Sniper Setup",
                        "trade_type": "COPY_BACKEND_TRADE_TYPE_EXACTLY_OR_WAIT_IF_REJECTED",
                        "probability": "COPY_OR_MATCH_BACKEND_PROBABILITY",
                        "summary": "Explain/review the backend-selected setup only. Do not create new levels.",
                        "entry_zone": "COPY_BACKEND_ENTRY_EXACTLY_OR_EMPTY_IF_WAIT",
                        "targets": ["COPY_BACKEND_TP1_EXACTLY", "COPY_BACKEND_TP2_EXACTLY"],
                        "invalidation": "COPY_BACKEND_STOP_EXACTLY_OR_EMPTY_IF_WAIT"
                    }}
                ]
            }}
            """

        # try:
        #     with open("debug_ai_prompt.txt", "w", encoding="utf-8") as f:
        #         f.write(prompt)
        #     print("✅ AI prompt saved to debug_ai_prompt.txt")
        # except Exception as e:
        #     print(f"Prompt debug save failed: {e}")

        raw_text = generate_content_smart(selected_model, prompt)

        # try:
        #     with open("debug_ai_raw_response.txt", "w", encoding="utf-8") as f:
        #         f.write(raw_text)
        # except Exception as e:
        #     print(f"Raw AI debug save failed: {e}")


        if '{"error"' in raw_text:
            return raw_text

        cleaned_text = raw_text.replace("```json", "").replace("```", "").strip()
        match = re.search(r"\{.*\}", cleaned_text, re.DOTALL)

        if match:
            json_str = match.group(0)
            try:
                data = json.loads(json_str)
                if not locked_scenario:
                    # Backend setup_planner remains the source of truth
                    # only for this legacy reviewer flow.
                    pass

                data = _normalize_ai_output(
                    data
                )

                data = _normalize_ai_review(
                    data
                )

                return json.dumps(
                    data
                )
            except Exception as e:
                print(f"JSON Parsing/Adjustment Error: {e}")
                return json.dumps({
                    "macro_analysis": {
                        "trend": "Unknown",
                        "detailed_breakdown": "AI parsing failed.",
                        "key_levels": "N/A"
                    },
                    "micro_analysis": {
                        "wave_count_status": "Unknown",
                        "wave_personality": "Unknown"
                    },
                    "action_advice": "Wait.",
                    "trade_scenarios": [_force_wait_scenario("AI parsing failed.")]
                })
        else:
            return json.dumps({"error": "No JSON found in AI response"})

    except Exception as e:
        return json.dumps({"error": str(e)})


# --- SCANNER ---
def analyze_scanner_candidates(candidates):
    has_pool = hasattr(config, "API_KEY_POOL") and config.API_KEY_POOL
    has_paid = hasattr(config, "PAID_API_KEY") and config.PAID_API_KEY

    if not has_pool and not has_paid:
        return json.dumps({"error": "No API Keys Configured"})

    candidates_str = ""
    for c in candidates:
        candidates_str += f"- {c['symbol']}: Price {c['price']}, RSI {c['rsi']:.1f}, Signal {c['signal']}\n"

    prompt = f"""
    Analyze these candidates. Pick Top 5.
    DATA:
    {candidates_str}
    OUTPUT JSON Array: [{{ "ticker": "SYM", "price": 0, "rsi": 0, "signal": "LONG/SHORT", "reason": "..." }}]
    """

    try:
        raw_text = generate_content_smart(config.MODEL_FAST, prompt)

        if '{"error"' in raw_text:
            return json.dumps([])

        cleaned = raw_text.replace("```json", "").replace("```", "").strip()
        match = re.search(r"\[.*\]", cleaned, re.DOTALL)
        return match.group(0) if match else json.dumps([])
    except Exception as e:
        return json.dumps({"error": str(e)})