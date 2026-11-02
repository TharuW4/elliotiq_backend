import re


def extract_price(price_str):
    """
    Extract first numeric price from int/float/string/range.

    Examples:
      "84.180 - 84.300" -> 84.180
      "83,500.25" -> 83500.25
    """
    try:
        if isinstance(
            price_str,
            (int, float),
        ):
            return float(price_str)

        match = re.search(
            r"[\d,]+(\.\d+)?",
            str(price_str),
        )

        if match:
            return float(
                match.group(0).replace(
                    ",",
                    "",
                )
            )

    except Exception:
        return None

    return None


def format_signal_price(value):
    """
    Dynamic price formatting for signal values.

    Retained for backward compatibility with any
    module that may import this helper directly.
    """
    value = extract_price(value)

    if value is None:
        return ""

    if abs(value) >= 1000:
        return f"{value:.2f}"

    if abs(value) >= 100:
        return f"{value:.2f}"

    if abs(value) >= 10:
        return f"{value:.3f}"

    if abs(value) >= 1:
        return f"{value:.4f}"

    if abs(value) >= 0.01:
        return f"{value:.4f}"

    if abs(value) >= 0.0001:
        return f"{value:.6f}"

    return f"{value:.8f}"


def calculate_running_pnl_percent(
    scenario,
    current_price,
):
    """
    Calculate unrealized price-movement PnL percentage
    for a running LONG/SHORT trade.

    This is not leverage-adjusted PnL.
    """
    try:
        scenario = scenario or {}

        trade_type = str(
            scenario.get("trade_type")
            or scenario.get("side")
            or ""
        ).upper().strip()

        entry = extract_price(
            scenario.get("entry_zone")
            or scenario.get("entry")
        )

        current = float(
            current_price
        )

        if (
            entry is None
            or entry <= 0
        ):
            return None

        if trade_type == "LONG":
            return (
                (current - entry)
                / entry
            ) * 100.0

        if trade_type == "SHORT":
            return (
                (entry - current)
                / entry
            ) * 100.0

    except Exception:
        pass

    return None


def format_running_pnl_advice(
    pnl_percent,
):
    """
    Basic RUNNING-state PnL text only.

    TP1/TP2/SL management is handled by
    models.trade_guidance_manager.
    """
    if pnl_percent is None:
        return (
            "Trade running. Waiting for "
            "TP/SL."
        )

    if pnl_percent > 0:
        return (
            f"🟢 In Profit "
            f"(+{pnl_percent:.2f}%). "
            "Follow the active trade "
            "management guidance."
        )

    if pnl_percent < 0:
        return (
            f"🔴 In Loss "
            f"({pnl_percent:.2f}%). "
            "Do not add; monitor the "
            "original invalidation."
        )

    return (
        "⚪ At entry price (0.00%). "
        "Trade is running."
    )