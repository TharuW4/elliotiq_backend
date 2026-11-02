# app/models/ai_signal_engine.py

"""
ElliotIQ AI-First Signal Engine
===============================

Creates a structured Gemini trade draft from the complete AI market packet.

The final deterministic SL / TP / path validation is handled by a separate
backend validator before a signal is published.

This module:
- Receives the complete AI market packet.
- Sends the packet to Gemini 3.5 Flash.
- Generates LONG / SHORT / WAIT / NO_TRADE.
- Generates Entry, Stop Loss and TP2.
- Keeps TP1 R multiple fixed at exactly 1.5R.
- Generates frontend STATUS and short SETUP SUMMARY advice.
- Supports one correction call after backend validation errors.

This module does NOT:
- Fetch candles.
- Fetch Binance data.
- Calculate final TP1 price.
- Save Firestore signals.
- Publish a signal without backend validation.
"""

from __future__ import annotations

import json
import math
import re
from typing import Any, Dict, Mapping, Optional, Sequence

from .. import config
from ..ai_engine import generate_structured_content_smart
from .ai_market_packet import (
    STOP_LOSS_MINIMUM_ATR,
    STOP_LOSS_MAXIMUM_ATR_BY_TIMEFRAME,
    TP1_R_MULTIPLE,
    TP2_MINIMUM_R,
    TP2_PREFERRED_R,
    TP2_MAXIMUM_R,
    TP2_MAXIMUM_ATR_BY_TIMEFRAME,
    market_packet_to_json,
    validate_ai_market_packet,
)


ENGINE_VERSION = "1.1.0"
PROMPT_VERSION = "1.3.0"


# ==========================================================
# ALLOWED OUTPUT VALUES
# ==========================================================

ALLOWED_DECISIONS = {
    "LONG",
    "SHORT",
    "WAIT",
    "NO_TRADE",
}


ALLOWED_SIDES = {
    "LONG",
    "SHORT",
    "NONE",
}


ALLOWED_EXECUTION_STATES = {
    "ENTER_NOW",
    "PLACE_LIMIT",
    "WAIT_FOR_ENTRY",
    "WAIT_FOR_RETEST",
    "WAIT_FOR_RECLAIM",
    "WAIT_FOR_BREAKOUT",
    "WAIT_FOR_FAILED_RECLAIM",
    "NO_TRADE",
}


ALLOWED_ORDER_TYPES = {
    "MARKET",
    "LIMIT",
    "STOP",
    "NONE",
}


ALLOWED_TRADE_QUALITIES = {
    "A",
    "B",
    "C",
    "NO_TRADE",
}


ALLOWED_RISK_LEVELS = {
    "LOW",
    "MEDIUM",
    "HIGH",
    "EXTREME",
}


ALLOWED_TIMEFRAME_BIASES = {
    "STRONG_BULLISH",
    "BULLISH",
    "NEUTRAL",
    "BEARISH",
    "STRONG_BEARISH",
    "UNCLEAR",
}


# ==========================================================
# GEMINI STRUCTURED RESPONSE SCHEMA
# ==========================================================

AI_SIGNAL_RESPONSE_SCHEMA: Dict[str, Any] = {
    "type": "object",

    "properties": {
        "decision": {
            "type": "string",
            "enum": [
                "LONG",
                "SHORT",
                "WAIT",
                "NO_TRADE",
            ],
        },

        "side": {
            "type": "string",
            "enum": [
                "LONG",
                "SHORT",
                "NONE",
            ],
        },

        "execution_state": {
            "type": "string",
            "enum": [
                "ENTER_NOW",
                "PLACE_LIMIT",
                "WAIT_FOR_ENTRY",
                "WAIT_FOR_RETEST",
                "WAIT_FOR_RECLAIM",
                "WAIT_FOR_BREAKOUT",
                "WAIT_FOR_FAILED_RECLAIM",
                "NO_TRADE",
            ],
        },

        "order_type": {
            "type": "string",
            "enum": [
                "MARKET",
                "LIMIT",
                "STOP",
                "NONE",
            ],
        },

        "entry_price": {
            "type": "number",
            "description": (
                "The planned trade entry price. "
                "Use 0 only when no concrete trade plan exists."
            ),
        },

        "stop_loss": {
            "type": "number",
            "description": (
                "The packet-supported structural invalidation price. "
                f"The global minimum is {STOP_LOSS_MINIMUM_ATR:.2f} ATR "
                "as a floor, not a fixed distance. The stop must also "
                "remain below the selected-timeframe hard maximum. "
                "Use 0 only when no concrete trade plan exists."
            ),
        },

        "tp1_r_multiple": {
            "type": "number",
            "description": (
                f"Must always be exactly {TP1_R_MULTIPLE:.1f}. "
                "The backend calculates and owns the exact TP1 price."
            ),
        },

        "tp2": {
            "type": "number",
            "description": (
                "Packet-supported extended target. "
                f"The hard minimum is {TP2_MINIMUM_R:.2f}R, "
                f"but prefer at least {TP2_PREFERRED_R:.2f}R. "
                "A TP2 between the hard minimum and preferred target "
                "is allowed only when its exact supplied structure anchor "
                "is valid and extending to the preferred target would be "
                "unsupported. The effective maximum is the lower of "
                f"{TP2_MAXIMUM_R:.2f}R and the selected-timeframe "
                "TP2 ATR cap; the backend may cap an excess target "
                "downward. Use 0 only when no concrete trade plan exists."
            ),
        },

        "setup_type": {
            "type": "string",
            "description": (
                "Concise setup type such as WAVE_2_CONTINUATION, "
                "BREAKOUT_RETEST, LIQUIDITY_SWEEP_REVERSAL, "
                "CRASH_CONTINUATION, RANGE_SWEEP or NO_TRADE."
            ),
        },

        "market_regime": {
            "type": "string",
            "description": (
                "Current market regime such as TREND, RANGE, CRASH, "
                "HIGH_VOLATILITY, PULLBACK_IN_TREND or UNCLEAR."
            ),
        },

        "confidence": {
            "type": "integer",
            "minimum": 0,
            "maximum": 100,
        },

        "trade_quality": {
            "type": "string",
            "enum": [
                "A",
                "B",
                "C",
                "NO_TRADE",
            ],
        },

        "trap_risk": {
            "type": "string",
            "enum": [
                "LOW",
                "MEDIUM",
                "HIGH",
                "EXTREME",
            ],
        },

        "sl_hit_risk": {
            "type": "string",
            "enum": [
                "LOW",
                "MEDIUM",
                "HIGH",
                "EXTREME",
            ],
        },

        "tp2_quality": {
            "type": "string",
            "enum": [
                "GOOD",
                "AVERAGE",
                "POOR",
                "NOT_APPLICABLE",
            ],
        },

        "expected_path": {
            "type": "string",
            "description": (
                "Expected market path such as RETEST_THEN_CONTINUATION, "
                "IMMEDIATE_CONTINUATION, FAILED_RECLAIM_THEN_DROP, "
                "SWEEP_THEN_REVERSAL, RANGE or UNCLEAR."
            ),
        },

        "timeframe_bias": {
            "type": "object",

            "properties": {
                "lower": {
                    "type": "string",
                    "enum": sorted(
                        ALLOWED_TIMEFRAME_BIASES
                    ),
                },

                "main": {
                    "type": "string",
                    "enum": sorted(
                        ALLOWED_TIMEFRAME_BIASES
                    ),
                },

                "confirm": {
                    "type": "string",
                    "enum": sorted(
                        ALLOWED_TIMEFRAME_BIASES
                    ),
                },

                "swing": {
                    "type": "string",
                    "enum": sorted(
                        ALLOWED_TIMEFRAME_BIASES
                    ),
                },

                "macro": {
                    "type": "string",
                    "enum": sorted(
                        ALLOWED_TIMEFRAME_BIASES
                    ),
                },
            },

            "required": [
                "lower",
                "main",
                "confirm",
                "swing",
                "macro",
            ],

            "additionalProperties": False,
        },

        "entry_anchor_path": {
            "type": "string",
            "description": (
                "Exact dotted path in the supplied market packet "
                "that supports the entry price. "
                "Use NONE only when no concrete plan exists."
            ),
        },

        "sl_anchor_path": {
            "type": "string",
            "description": (
                "Exact dotted path in the supplied market packet "
                "that supports structural invalidation. "
                "Use NONE only when no concrete plan exists."
            ),
        },

        "tp2_anchor_path": {
            "type": "string",
            "description": (
                "Exact dotted path in the supplied market packet "
                "that supports TP2. "
                "Use NONE only when no concrete plan exists."
            ),
        },

        "evidence_paths": {
            "type": "array",

            "items": {
                "type": "string",
            },

            "description": (
                "Exact dotted paths from the market packet that "
                "support this decision."
            ),
        },

        "counter_evidence_paths": {
            "type": "array",

            "items": {
                "type": "string",
            },

            "description": (
                "Exact dotted paths containing important opposing "
                "or risk evidence."
            ),
        },

        "reason_codes": {
            "type": "array",

            "items": {
                "type": "string",
            },
        },

        "decision_reason": {
            "type": "string",
            "description": (
                "Evidence-based explanation of the trade decision. "
                "Generic reasoning is forbidden."
            ),
        },

        "invalidation_condition": {
            "type": "string",
            "description": (
                "Concrete market condition that invalidates the setup, "
                "or NONE when no plan exists."
            ),
        },

        "status_text": {
            "type": "string",
            "description": (
                "One short frontend STATUS line."
            ),
        },

        "advice": {
            "type": "string",
            "description": (
                "One compact frontend advice paragraph containing "
                "no more than two short sentences."
            ),
        },
    },

    "required": [
        "decision",
        "side",
        "execution_state",
        "order_type",
        "entry_price",
        "stop_loss",
        "tp1_r_multiple",
        "tp2",
        "setup_type",
        "market_regime",
        "confidence",
        "trade_quality",
        "trap_risk",
        "sl_hit_risk",
        "tp2_quality",
        "expected_path",
        "timeframe_bias",
        "entry_anchor_path",
        "sl_anchor_path",
        "tp2_anchor_path",
        "evidence_paths",
        "counter_evidence_paths",
        "reason_codes",
        "decision_reason",
        "invalidation_condition",
        "status_text",
        "advice",
    ],

    "additionalProperties": False,
}


# ==========================================================
# PROFESSIONAL TRADER SYSTEM PROMPT
# ==========================================================

def _build_risk_policy_instruction() -> str:
    """
    Build the exact AI-visible risk contract from the same
    constants used by the market-packet policy.
    """

    sl_maximums = ", ".join(
        f"{timeframe}={value:.2f} ATR"
        for timeframe, value
        in STOP_LOSS_MAXIMUM_ATR_BY_TIMEFRAME.items()
    )

    tp2_atr_maximums = ", ".join(
        f"{timeframe}={value:.2f} ATR"
        for timeframe, value
        in TP2_MAXIMUM_ATR_BY_TIMEFRAME.items()
    )

    return (
        "HARD_RISK_POLICY\n"
        f"SL minimum floor: {STOP_LOSS_MINIMUM_ATR:.2f} ATR. "
        "This is not a fixed SL distance.\n"
        f"SL maximums: {sl_maximums}.\n"
        f"TP1: exactly {TP1_R_MULTIPLE:.2f}R.\n"
        f"TP2 hard minimum: {TP2_MINIMUM_R:.2f}R.\n"
        f"TP2 preferred target: at least {TP2_PREFERRED_R:.2f}R when supported.\n"
        "A TP2 between hard minimum and preferred is allowed only with "
        "an exact supplied structure anchor; never invent an extension.\n"
        f"TP2 global maximum: {TP2_MAXIMUM_R:.2f}R.\n"
        f"TP2 ATR maximums: {tp2_atr_maximums}.\n"
        "Effective TP2 maximum: the lower of the global R maximum "
        "and the selected-timeframe TP2 ATR maximum.\n"
        "An SL outside its range must be structurally rebuilt or the "
        "decision must become WAIT/NO_TRADE; never clamp SL blindly.\n"
        "A TP2 below the minimum is invalid. A TP2 above the effective "
        "maximum may be capped downward by the backend and then the "
        "remaining validation continues."
    )


SYSTEM_INSTRUCTION = r"""
You are ElliotIQ's final AI-first crypto futures signal decision engine.

Act as a disciplined professional trader, multi-timeframe market-structure
analyst and risk manager.

Your objective is to find high-quality positive-expectancy opportunities while:

- avoiding weak trades,
- avoiding liquidity traps,
- avoiding late entries,
- avoiding invalid stop losses,
- avoiding unrealistic targets,
- reducing unnecessary losses,
- and not becoming so restrictive that valid opportunities are missed.

The goal is not to generate the maximum number of signals.
The goal is not to reject every trade.
The goal is to select the strongest realistically tradable opportunities.


SOURCE-OF-TRUTH RULES

1. The supplied JSON market packet is the only market truth.

2. Never invent:

- candles,
- timestamps,
- price levels,
- support or resistance,
- liquidity pools,
- Fibonacci levels,
- Elliott Wave labels,
- BOS or CHoCH events,
- FVG or imbalance zones,
- indicator values,
- volume states,
- probabilities,
- or historical outcomes.

3. Independently analyse the market packet.

4. Do not assume that a backend-generated signal already exists.

5. Historical analog results are statistical evidence, not guaranteed outcomes.

6. If essential market evidence is missing or contradictory, return WAIT or
NO_TRADE.

7. Never approve a trade because it merely:

- looks good,
- seems bullish,
- seems bearish,
- feels strong,
- or probably works.

Every trade requires concrete evidence from the supplied packet.


MANDATORY ANALYSIS PROCESS

Before generating any signal:

1. Read the lower timeframe direction.

2. Read the main execution timeframe direction.

3. Read the confirmation timeframe direction.

4. Read the swing timeframe direction.

5. Read the macro timeframe direction.

6. Determine whether the current market is:

- a clean trend,
- a pullback inside a trend,
- a range,
- a crash,
- high volatility,
- low volatility,
- a reversal,
- a transition,
- or unclear.

7. Combine:

- recent closed candles,
- candle bodies,
- wick behavior,
- volume,
- ATR,
- volatility,
- Elliott Wave structure,
- Elliott phase,
- SMC context,
- BOS,
- CHoCH,
- order blocks,
- FVG and imbalance,
- liquidity pools,
- liquidity sweeps,
- support and resistance,
- Fibonacci retracement,
- Fibonacci extension,
- momentum,
- divergence,
- rejection candles,
- and Historical Analog Engine statistics.

8. Evaluate both LONG and SHORT objectively before choosing a side.

9. Distinguish between:

- genuine continuation,
- pullback continuation,
- confirmed reversal,
- liquidity sweep,
- fake breakout,
- failed reclaim,
- breakout retest,
- range sweep,
- crash continuation,
- late chase,
- and exhausted Wave 5.

10. Generate a trade only when:

- a valid entry exists,
- a structural invalidation exists,
- the stop is realistic,
- TP1 at 1.5R is realistic,
- TP2 has a real market anchor,
- and the trade is not already too late.


MARKET REGIME RULES

TREND:

- Prioritize confirmed pullbacks and continuation in the higher-timeframe
  direction.
- Avoid chasing extended candles.
- Avoid entering after most of the move to TP1 has already occurred.
- Avoid exhausted Wave-5 continuation unless a new structure confirms.

RANGE:

- Avoid taking trades in the middle of the range.
- Prefer a range-boundary liquidity sweep and reclaim.
- Prefer a clear boundary rejection.
- A breakout requires a confirmed close and retest.
- Treat fake-breakout risk as important.

CRASH OR EXTREME VOLATILITY:

- Do not create a LONG merely because price is oversold.
- Crash detection alone is never a valid trade setup.
- Default to WAIT or NO_TRADE during an active crash.
- Consider continuation only after a completed pullback or retest, a closed-candle failed reclaim or rejection, higher-timeframe alignment, a structural Stop Loss, a non-chased entry and a clear TP path.
- If any of these confirmations is missing, do not generate Entry, SL or TP prices; return WAIT with side NONE or NO_TRADE.
- Do not SHORT an already overextended drop directly into major support.
- Stop distance must respect extreme volatility.
- A normal trend-market stop may be invalid in a crash regime.

REVERSAL:

- Require an actual structural change.
- Require supporting liquidity and candle evidence.
- Require BOS or CHoCH confirmation when appropriate.
- A single RSI, divergence or Elliott label is not enough.


HISTORICAL ANALOG ENGINE RULES

Use all available analog information:

- likely market direction,
- LONG probability,
- SHORT probability,
- range probability,
- expected favorable movement,
- expected adverse movement,
- expected market path,
- move-start timing,
- time to target,
- fake-breakout probability,
- reversal probability,
- entry-touch-then-adverse probability,
- sample count,
- independent match count,
- average similarity,
- multi-timeframe alignment,
- and prediction confidence.

Historical analog results must not be used alone.

Unavailable historical analog data is neutral missing optional evidence.
Do not treat unavailable history as bearish evidence, proof of negative
expectancy, or an independent reason for WAIT/NO_TRADE. Use current structure,
candles, liquidity and the supplied risk limits. Do not invent statistics.

If analog confidence is LOW or INSUFFICIENT_HISTORY:

- give analog evidence low weight,
- and rely more heavily on current structure and current candles.

If analog evidence indicates RETEST_THEN_CONTINUATION:

- do not chase the current price;
- use LONG/SHORT with PLACE_LIMIT for an already supported structural
  pullback entry when only the future price touch remains;
- use WAIT only if a separate essential confirmation event is missing;
- unavailable analog evidence must not create an extra confirmation requirement.

If current Elliott, SMC, trend or liquidity conditions strongly contradict the
analog direction, do not blindly follow analog statistics.


ENTRY RULES

Entry must be supported by at least one supplied market event or level:

- support or resistance,
- swing structure,
- reclaim,
- retest,
- breakout,
- failed reclaim,
- liquidity sweep,
- candle confirmation,
- or validated pullback structure.

Do not invent a new entry price between unsupported levels.

For a MARKET entry:

- current structure must already confirm entry,
- the trade must not be chased,
- and immediate trap risk must be acceptable.

For a LIMIT entry:

- the entry must have a clear supplied structure, S/R, Fibonacci, order-block,
  liquidity or retest anchor;
- if the setup is already supported and only price reaching entry is pending,
  use decision LONG/SHORT, execution_state PLACE_LIMIT, and order_type LIMIT;
- an unfilled limit order is not a WAIT idea;
- assess the stop and target path for the proposed entry price, not for an
  immediate market entry at the current price;
- resistance near the current price can justify a lower supported LONG entry,
  and support near current price can justify a higher supported SHORT entry,
  but the proposed entry-to-target path must still be valid.

Advisory preferred-side WAIT labels, minor CHoCH, or a lower-timeframe pullback
alone are not automatic vetoes. Respect explicit backend hard blocks and
confirmed major structure conflicts. Do not pretend that an essential future
reversal or breakout confirmation is already complete.

Before selecting WAIT:

- distinguish a price-only return to a supported level from a later candle
  hold/rejection or structural event;
- if only the price return remains, use a supported pending LIMIT plan;
- if a later close, rejection, failed reclaim, or structure break is essential,
  keep WAIT, set order_type NONE, and state the exact event, level and timeframe;
- use "WAIT FOR RETEST" only when a specific reaction/hold beyond mere price
  touch is essential; explain that reaction in decision_reason;
- never switch WAIT to LONG/SHORT merely to satisfy a signal quota.




STOP-LOSS RULES

Read the exact selected-timeframe limits from:

non_negotiable_trade_rules.selected_timeframe_risk_policy

The minimum ATR value is a floor, not a fixed stop distance. Do not place every
stop at the minimum. Select the real packet-supported structural invalidation
level inside the permitted range.

LONG:

- stop_loss must be below entry_price,
- and should sit beyond valid support, swing low or structural invalidation.

SHORT:

- stop_loss must be above entry_price,
- and should sit beyond valid resistance, swing high or structural invalidation.

Never expect the backend to blindly widen, tighten or clamp an invalid SL.
An SL below the minimum or above the selected-timeframe maximum fails
validation and may receive one correction pass.

When correcting an invalid SL:

- reconsider Entry and structural invalidation together,
- choose another genuine packet-supported structure only when one exists,
- rebuild TP1 and TP2 consistently from the corrected risk distance,
- never move SL to the minimum or maximum merely to pass validation,
- and return WAIT or NO_TRADE when no valid structural plan exists.

The stop must not be widened merely to force a trade.
The stop must not be artificially tightened merely to satisfy the maximum.

The stop must also respect:

- ATR,
- volatility,
- recent wick behavior,
- and historically normal adverse movement.

If no valid structural stop exists inside the permitted range, return WAIT or
NO_TRADE.


LIQUIDITY EXECUTION POLICY

Read liquidity evidence from:
quantitative_context.institutional_evidence.local_liquidity

Liquidity is evidence, not a universal entry gate.
A pool near entry or between entry and stop is not a rejection reason alone.
Assess whether the actual structural stop has adequate clearance beyond the
relevant liquidity level and satisfies the packet risk policy.

An unswept pool, an early/minor CHoCH, or a lower-timeframe pullback alone
does not require WAIT. Assess the main/confirmation-timeframe structure,
invalidation, current closed candles, and realistic target path together.
Respect explicit backend hard blocks and confirmed major structure conflicts.

A supportive closed-candle sweep/reclaim is positive evidence. Do not require
another retest solely because a supportive sweep exists. The reclaim candle
may provide reaction evidence if it is closed and structurally valid; it does
not prove that a separate retest occurred.

For both FULL_TRADE and CONFIRMATION_REQUIRED contexts, decide whether the
specific entry is already supported. CONFIRMATION_REQUIRED is not an
automatic instruction to demand a liquidity sweep plus an additional retest.
Any confirmation genuinely required by the chosen setup must still exist.

If a supported structural LIMIT entry is valid on price touch without another
unobserved event, use LONG/SHORT with PLACE_LIMIT and order_type LIMIT.
Price not yet reaching that entry is not, by itself, a reason for WAIT.
ENTER_NOW requires current executable price and completed required evidence.

Use WAIT only when a specific essential event is still missing. State the
exact level/event needed. A plan requiring a future failed reclaim, breakout
hold, or reversal confirmation must not be presented as an active limit order.
Never relabel an unconfirmed WAIT as LONG/SHORT just to increase signal count.

A stop inside a likely sweep zone needs a structurally justified alternative
within the existing ATR/RR limits. If none exists, use WAIT or NO_TRADE.
Do not widen a stop or invent a target merely to make the trade pass.


TARGET RULES

Read the exact selected-timeframe target limits from:

non_negotiable_trade_rules.selected_timeframe_risk_policy

TP1:

- tp1_r_multiple must always be exactly 1.5.
- The backend calculates the exact TP1 price from Entry and Stop Loss.
- Do not change the 1.5R requirement.

TP2:

- must meet the packet-supplied minimum R requirement,
- and its effective maximum is the lower of the global R maximum and the
  selected-timeframe TP2 ATR cap.

A TP2 below the minimum is invalid. Do not automatically extend an unsupported
target to the minimum. Rebuild the plan from real structure or return WAIT or
NO_TRADE.

A requested TP2 above the effective maximum is not rejected for excess alone.
The backend may cap it downward and continue validation. However, the requested
TP2 must still have an actual supplied market anchor and must not be invented.

TP2 also:

- must be beyond TP1 in the trade direction,
- must have an actual supplied market anchor,
- must be supported by S/R, swing structure, liquidity, Fibonacci extension
  or measured movement,
- must be realistic compared with historical expected movement,
- and must not blindly extend beyond major opposing higher-timeframe structure.

If a realistic TP2 does not exist, do not force one.
Return WAIT or NO_TRADE when the setup cannot satisfy the required rules.


EVIDENCE CONTRACT

For a concrete LONG, SHORT or directional WAIT plan:

- include at least four evidence_paths,
- use exact dotted paths from the supplied market packet,
- include important opposing evidence in counter_evidence_paths,
- provide an entry_anchor_path,
- provide a sl_anchor_path,
- provide a tp2_anchor_path.

Do not hide conflicting evidence.

The decision_reason must explain:

- the strongest supporting evidence,
- the most important opposing risk,
- and why the selected action is appropriate.

Use NONE for anchor paths only when no concrete plan exists.


DECISION MEANING

LONG:

- A supported LONG trade plan exists.
- It may be immediate or a pending order.

SHORT:

- A supported SHORT trade plan exists.
- It may be immediate or a pending order.

For LONG/SHORT, a supported pending LIMIT order uses PLACE_LIMIT.
Do not use WAIT merely because the order has not filled.
Do not use a pending LIMIT order when an essential confirmation event is
still missing; that remains WAIT.

WAIT:

- A directional idea may exist,
- but a specific confirmation, retest, reclaim, breakout or failed-reclaim
  event is still required.
- WAIT may contain valid planned Entry, Stop Loss and TP2 prices.

NO_TRADE:

- No sufficiently supported trade exists.
- Use side NONE.
- Use order_type NONE.
- Use execution_state NO_TRADE.
- Use entry_price 0.
- Use stop_loss 0.
- Use tp2 0.


FRONTEND STATUS RULES

status_text must:

- be one short line,
- normally remain under 40 characters,
- contain no long explanation,
- contain no markdown,
- contain no heading.

Good examples:

LONG | Entry Confirmed
SHORT | Entry Confirmed
WAIT | Retest Required
LIMIT ORDER | Fresh Setup
WAIT | Failed Reclaim Needed
NO TRADE | Weak Structure


FRONTEND ADVICE RULES

advice must:

- be one compact paragraph,
- contain no more than two short sentences,
- state the immediate action first,
- clearly tell the user what to do now,
- and briefly explain the strongest reason or required confirmation.

Entry, Stop Loss, TP1 and TP2 are displayed separately in the frontend.

Mention the Entry price in advice only when:

- the trade has not yet entered,
- and the price is useful for a pending entry, limit entry or retest instruction.

Do not repeat:

- Stop Loss price,
- TP1 price,
- or TP2 price

inside advice.

Do not use generic statements such as:

- trade carefully,
- use risk management,
- be cautious,
- looks good,
- seems strong.

Do not use:

- markdown,
- headings,
- bullet points,
- or multiple advice sections.

Return only the structured object required by the schema.
""".strip()


# ==========================================================
# NORMALIZATION HELPERS
# ==========================================================

def _safe_float(value: Any) -> float:
    try:
        if isinstance(value, str):
            value = (
                value
                .replace(",", "")
                .replace("$", "")
                .strip()
            )

        number = float(value)

        if not math.isfinite(number):
            return 0.0

        return number

    except Exception:
        return 0.0


def _clean_text(
    value: Any,
    max_length: int,
) -> str:
    text = str(
        value or ""
    ).strip()

    text = re.sub(
        r"[`*_#]+",
        "",
        text,
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    ).strip()

    return text[:max_length].strip()

def _clean_packet_path(
    value: Any,
    max_length: int = 300,
) -> str:
    if not isinstance(value, str):
        return ""

    text = value.strip().strip("`\"'")
    text = re.sub(r"\[\s*(\d+)\s*\]", r".\1", text)
    return text[:max_length].strip()


def _clean_packet_paths(
    value: Any,
    max_items: int = 12,
) -> list[str]:
    if not isinstance(value, (list, tuple, set)):
        return []

    result: list[str] = []

    for item in value:
        path = _clean_packet_path(item)

        if path and path not in result:
            result.append(path)

        if len(result) >= max_items:
            break

    return result



def _clean_list(
    value: Any,
    max_items: int = 12,
) -> list[str]:
    if not isinstance(
        value,
        (
            list,
            tuple,
            set,
        ),
    ):
        return []

    result: list[str] = []

    for item in value:
        text = _clean_text(
            item,
            300,
        )

        if (
            text
            and text not in result
        ):
            result.append(text)

        if len(result) >= max_items:
            break

    return result


def _enum(
    value: Any,
    allowed: set[str],
    default: str,
) -> str:
    text = (
        str(
            value or ""
        )
        .strip()
        .upper()
        .replace(" ", "_")
    )

    if text in allowed:
        return text

    return default


def _normalize_timeframe_bias(
    value: Any,
) -> Dict[str, str]:
    source = (
        value
        if isinstance(
            value,
            Mapping,
        )
        else {}
    )

    return {
        role: _enum(
            source.get(role),
            ALLOWED_TIMEFRAME_BIASES,
            "UNCLEAR",
        )

        for role in (
            "lower",
            "main",
            "confirm",
            "swing",
            "macro",
        )
    }


# ==========================================================
# AI RESPONSE NORMALIZATION
# ==========================================================

def normalize_ai_signal(
    raw: Mapping[str, Any],
) -> Dict[str, Any]:
    decision = _enum(
        raw.get("decision"),
        ALLOWED_DECISIONS,
        "NO_TRADE",
    )

    signal: Dict[str, Any] = {
        "decision": decision,

        "side": _enum(
            raw.get("side"),
            ALLOWED_SIDES,
            "NONE",
        ),

        "execution_state": _enum(
            raw.get("execution_state"),
            ALLOWED_EXECUTION_STATES,
            "NO_TRADE",
        ),

        "order_type": _enum(
            raw.get("order_type"),
            ALLOWED_ORDER_TYPES,
            "NONE",
        ),

        "entry_price": _safe_float(
            raw.get("entry_price")
        ),

        "stop_loss": _safe_float(
            raw.get("stop_loss")
        ),

        # TP1 is always fixed at exactly 1.5R.
        "tp1_r_multiple": TP1_R_MULTIPLE,

        "tp2": _safe_float(
            raw.get("tp2")
        ),

        "setup_type": (
            _clean_text(
                raw.get("setup_type"),
                100,
            )
            or "NO_TRADE"
        ),

        "market_regime": (
            _clean_text(
                raw.get("market_regime"),
                100,
            )
            or "UNCLEAR"
        ),

        "confidence": max(
            0,
            min(
                100,
                int(
                    _safe_float(
                        raw.get("confidence")
                    )
                ),
            ),
        ),

        "trade_quality": _enum(
            raw.get("trade_quality"),
            ALLOWED_TRADE_QUALITIES,
            "NO_TRADE",
        ),

        "trap_risk": _enum(
            raw.get("trap_risk"),
            ALLOWED_RISK_LEVELS,
            "HIGH",
        ),

        "sl_hit_risk": _enum(
            raw.get("sl_hit_risk"),
            ALLOWED_RISK_LEVELS,
            "HIGH",
        ),

        "tp2_quality": _enum(
            raw.get("tp2_quality"),
            {
                "GOOD",
                "AVERAGE",
                "POOR",
                "NOT_APPLICABLE",
            },
            "NOT_APPLICABLE",
        ),

        "expected_path": (
            _clean_text(
                raw.get("expected_path"),
                160,
            )
            or "UNCLEAR"
        ),

        "timeframe_bias": (
            _normalize_timeframe_bias(
                raw.get("timeframe_bias")
            )
        ),

        "entry_anchor_path": (
            _clean_packet_path(raw.get("entry_anchor_path")) or "NONE"
        ),
        "sl_anchor_path": (
            _clean_packet_path(raw.get("sl_anchor_path")) or "NONE"
        ),
        "tp2_anchor_path": (
            _clean_packet_path(raw.get("tp2_anchor_path")) or "NONE"
        ),
        "evidence_paths": _clean_packet_paths(
            raw.get("evidence_paths")
        ),
        "counter_evidence_paths": _clean_packet_paths(
            raw.get("counter_evidence_paths")
        ),


        "reason_codes": _clean_list(
            raw.get("reason_codes")
        ),

        "decision_reason": _clean_text(
            raw.get("decision_reason"),
            900,
        ),

        "invalidation_condition": (
            _clean_text(
                raw.get(
                    "invalidation_condition"
                ),
                400,
            )
            or "NONE"
        ),

        "status_text": _clean_text(
            raw.get("status_text"),
            80,
        ),

        "advice": _clean_text(
            raw.get("advice"),
            500,
        ),
    }

    # Prevent stale or hallucinated prices from leaking into
    # a NO_TRADE response.
    if decision == "NO_TRADE":
        signal.update(
            {
                "side": "NONE",
                "execution_state": "NO_TRADE",
                "order_type": "NONE",
                "entry_price": 0.0,
                "stop_loss": 0.0,
                "tp2": 0.0,
                "trade_quality": "NO_TRADE",
                "entry_anchor_path": "NONE",
                "sl_anchor_path": "NONE",
                "tp2_anchor_path": "NONE",
            }
        )

    return signal


# ==========================================================
# PROMPT BUILDER
# ==========================================================

def _build_prompt(
    market_packet: Mapping[str, Any],
    correction_context: Optional[
        Mapping[str, Any]
    ] = None,
) -> str:
    prompt = (
        "Analyse the following ElliotIQ market packet. "
        "Evaluate LONG and SHORT independently, then create "
        "the highest-quality valid decision. Obey the exact "
        "selected-timeframe risk policy. When a matching "
        "evidence_catalog item exists, copy its exact stored "
        "path into evidence and anchor fields; never invent or "
        "edit catalog IDs or paths. Use another exact packet "
        "path only when the catalog has no matching evidence."
        "\n\nMARKET_PACKET_JSON\n"
        + market_packet_to_json(
            market_packet,
            pretty=False,
        )
    )

    if (
        isinstance(
            correction_context,
            Mapping,
        )
        and correction_context
    ):
        prompt += (
            "\n\nCORRECTION_PASS\n"
            "The previous draft failed deterministic backend "
            "validation. Correct it without forcing a trade. "
            "When Entry, SL or target geometry changes, rebuild "
            "Entry, structural SL and TP2 consistently. Never move "
            "SL blindly to the minimum floor or timeframe maximum. "
            "Use a real packet-supported structural invalidation. "
            "A TP2 above the effective maximum may be capped by the "
            "backend, but a TP2 below the minimum is invalid. Return "
            "WAIT or NO_TRADE if all rules cannot be satisfied.\n"
            + json.dumps(
                correction_context,
                ensure_ascii=False,
                separators=(
                    ",",
                    ":",
                ),
            )
        )

    return prompt


# ==========================================================
# FIRST GEMINI SIGNAL CALL
# ==========================================================

def generate_ai_signal_draft(
    market_packet: Mapping[str, Any],
    *,
    correction_context: Optional[
        Mapping[str, Any]
    ] = None,
    model_name: Optional[str] = None,
    deadline_monotonic: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Generates an independent Gemini signal draft.

    The returned signal has not yet passed the final deterministic
    backend SL / TP / S/R / liquidity / path validator.
    """

    packet_validation = (
        validate_ai_market_packet(
            market_packet
        )
    )

    if not packet_validation.get(
        "valid"
    ):
        return {
            "ok": False,

            "signal": None,

            "provider": {},

            "packet_validation": (
                packet_validation
            ),

            "engine_version": (
                ENGINE_VERSION
            ),

            "prompt_version": (
                PROMPT_VERSION
            ),

            "error": (
                "INVALID_AI_MARKET_PACKET"
            ),
        }

    selected_model = str(
        model_name
        or getattr(
            config,
            "MODEL_SMART",
            "gemini-3.5-flash",
        )
    )

    provider_result = (
        generate_structured_content_smart(
            selected_model,

            _build_prompt(
                market_packet,
                correction_context,
            ),

            AI_SIGNAL_RESPONSE_SCHEMA,

            system_instruction=(
                SYSTEM_INSTRUCTION
            ),

            temperature=0.1,

            top_p=0.2,

            max_output_tokens=12288,

            deadline_monotonic=(
                deadline_monotonic
            ),
        )
    )

    if not provider_result.get(
        "ok"
    ):
        return {
            "ok": False,

            "signal": None,

            "provider": {
                "model": (
                    provider_result.get(
                        "model"
                    )
                ),

                "key_source": (
                    provider_result.get(
                        "key_source"
                    )
                ),

                "usage": (
                    provider_result.get(
                        "usage"
                    )
                    or {}
                ),
            },

            "packet_validation": (
                packet_validation
            ),

            "engine_version": (
                ENGINE_VERSION
            ),

            "prompt_version": (
                PROMPT_VERSION
            ),

            "error": str(
                provider_result.get(
                    "error"
                )
                or (
                    "GEMINI_STRUCTURED_"
                    "REQUEST_FAILED"
                )
            ),
        }

    raw_data = provider_result.get(
        "data"
    )

    if not isinstance(
        raw_data,
        Mapping,
    ):
        return {
            "ok": False,

            "signal": None,

            "provider": {
                "model": (
                    provider_result.get(
                        "model"
                    )
                ),

                "key_source": (
                    provider_result.get(
                        "key_source"
                    )
                ),

                "usage": (
                    provider_result.get(
                        "usage"
                    )
                    or {}
                ),
            },

            "packet_validation": (
                packet_validation
            ),

            "engine_version": (
                ENGINE_VERSION
            ),

            "prompt_version": (
                PROMPT_VERSION
            ),

            "error": (
                "GEMINI_RETURNED_"
                "NON_OBJECT_RESPONSE"
            ),
        }

    normalized_signal = (
        normalize_ai_signal(
            raw_data
        )
    )

    return {
        "ok": True,

        "signal": (
            normalized_signal
        ),

        "provider": {
            "model": (
                provider_result.get(
                    "model"
                )
            ),

            "key_source": (
                provider_result.get(
                    "key_source"
                )
            ),

            "usage": (
                provider_result.get(
                    "usage"
                )
                or {}
            ),
        },

        "packet_validation": (
            packet_validation
        ),

        "engine_version": (
            ENGINE_VERSION
        ),

        "prompt_version": (
            PROMPT_VERSION
        ),

        "error": None,
    }


# ==========================================================
# ONE-TIME CORRECTION / REGENERATION CALL
# ==========================================================

def regenerate_ai_signal(
    market_packet: Mapping[str, Any],
    previous_signal: Mapping[str, Any],
    validation_errors: Sequence[str],
    *,
    validation_warnings: Optional[
        Sequence[str]
    ] = None,
    model_name: Optional[str] = None,
    deadline_monotonic: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Gives Gemini one correction attempt after the deterministic
    backend validator identifies invalid SL / TP / geometry /
    evidence or market-structure issues.

    The caller should allow only one regeneration attempt.
    """

    safe_previous_signal = (
        normalize_ai_signal(
            previous_signal
            if isinstance(
                previous_signal,
                Mapping,
            )
            else {}
        )
    )

    error_source = (
        validation_errors
        if isinstance(
            validation_errors,
            (
                list,
                tuple,
                set,
            ),
        )
        else []
    )

    warning_source = (
        validation_warnings
        if isinstance(
            validation_warnings,
            (
                list,
                tuple,
                set,
            ),
        )
        else []
    )

    safe_validation_errors = [
        _clean_text(
            item,
            500,
        )

        for item in list(
            error_source
        )[:12]

        if str(item).strip()
    ]

    safe_validation_warnings = [
        _clean_text(
            item,
            500,
        )

        for item in list(
            warning_source
        )[:12]

        if str(item).strip()
    ]

    correction_context = {
        "previous_signal": (
            safe_previous_signal
        ),

        "validation_errors": (
            safe_validation_errors
        ),

        "validation_warnings": (
            safe_validation_warnings
        ),

        "required_action": (
            "Return one corrected valid signal or WAIT/NO_TRADE. "
            "For SL errors, rebuild Entry and structural invalidation "
            "together; never clamp SL blindly to a numeric boundary. "
            "Recalculate target geometry from the corrected risk. "
            "Do not invent unsupported prices, levels or evidence."
        ),
    }

    return generate_ai_signal_draft(
        market_packet,

        correction_context=(
            correction_context
        ),

        model_name=model_name,

        deadline_monotonic=(
            deadline_monotonic
        ),
    )