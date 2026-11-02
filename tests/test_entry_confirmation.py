import pandas as pd

from app.models.entry_confirmation import evaluate_entry_confirmation


def _df(rows):
    return pd.DataFrame(rows, columns=["Open", "High", "Low", "Close"])


def test_long_entry_confirmation_can_confirm_after_reclaim():
    rows = [[100, 101, 99.5, 100.5] for _ in range(205)]
    rows[-1] = [100.8, 101.6, 98.5, 101.2]

    result = evaluate_entry_confirmation(
        side="LONG",
        entry_price=100,
        stop_loss=95,
        user_timeframe="1h",
        lower_df=_df(rows),
        entry_zone_touched=True,
    )

    assert result["enabled"] is True
    assert result["confirmation_timeframe"] == "15m"
    assert result["confirmation_status"] == "CONFIRMED"
    assert result["would_activate_by_confirmation"] is True
    assert result["signals"]["close_reclaim"] is True
    assert result["signals"]["wick_rejection"] is True


def test_confirmation_waits_when_entry_zone_not_touched():
    result = evaluate_entry_confirmation(
        side="LONG",
        entry_price=100,
        stop_loss=95,
        user_timeframe="4h",
        lower_df=_df([[101, 103, 100.5, 102]]),
        entry_zone_touched=False,
    )

    assert result["confirmation_timeframe"] == "1h"
    assert result["confirmation_status"] == "WAITING"
    assert result["would_activate_by_confirmation"] is False
    assert "has not been touched" in result["advice"]


def test_short_entry_confirmation_invalidates_on_close_above_stop():
    result = evaluate_entry_confirmation(
        side="SHORT",
        entry_price=100,
        stop_loss=105,
        user_timeframe="4h",
        lower_df=_df([[101, 107, 100, 106]]),
        entry_zone_touched=True,
    )

    assert result["confirmation_status"] == "INVALIDATED"
    assert result["confirmation_score"] == 0
    assert result["signals"]["invalid_close"] is True
    assert result["would_activate_by_confirmation"] is False


def test_entry_confirmation_not_applicable_for_other_timeframes():
    result = evaluate_entry_confirmation(
        side="LONG",
        entry_price=100,
        stop_loss=95,
        user_timeframe="15m",
        lower_df=_df([[100, 101, 99, 100.5]]),
        entry_zone_touched=True,
    )

    assert result["enabled"] is False
    assert result["confirmation_status"] == "NOT_APPLICABLE"
