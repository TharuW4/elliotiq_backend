import json
import re
import time
from google import genai
from . import config
import pandas as pd
from .signal_quality import enforce_backend_quality_score

# --- KEY ROTATION & HYBRID LOGIC START ---
CURRENT_KEY_INDEX = 0


def get_next_free_client():
    """Returns a client using the next available FREE key from the pool."""
    global CURRENT_KEY_INDEX

    if not hasattr(config, "API_KEY_POOL") or not config.API_KEY_POOL:
        return None

    key = config.API_KEY_POOL[CURRENT_KEY_INDEX]
    CURRENT_KEY_INDEX = (CURRENT_KEY_INDEX + 1) % len(config.API_KEY_POOL)
    return genai.Client(api_key=key)


def generate_content_smart(model_name, prompt):
    """
    SMART HYBRID STRATEGY:
    1. Try all FREE keys in the pool (Round-Robin).
    2. If ALL free keys fail (Rate Limited 429), switch to PAID key.
    """
    if hasattr(config, "API_KEY_POOL") and config.API_KEY_POOL:
        attempts = 0
        max_free_retries = len(config.API_KEY_POOL)

        while attempts < max_free_retries:
            try:
                client = get_next_free_client()
                if not client:
                    break

                response = client.models.generate_content(model=model_name, contents=prompt)
                return response.text

            except Exception as e:
                error_msg = str(e)
                if "429" in error_msg or "ResourceExhausted" in error_msg or "Quota" in error_msg:
                    print(f"⚠️ Free Key {CURRENT_KEY_INDEX} Rate Limited. Switching to next...")
                    attempts += 1
                    time.sleep(0.5)
                    continue
                else:
                    print(f"❌ AI Logic Error (Non-Limit): {error_msg}")
                    return json.dumps({"error": str(e)})

    print("⚠️ All Free Keys Exhausted! Switching to PAID KEY Backup...")

    if hasattr(config, "PAID_API_KEY") and config.PAID_API_KEY:
        try:
            client = genai.Client(api_key=config.PAID_API_KEY)
            response = client.models.generate_content(model=model_name, contents=prompt)
            print("✅ Success using PAID KEY.")
            return response.text
        except Exception as e:
            print(f"❌ Paid Key Failed: {e}")
            return json.dumps({"error": f"PAID_KEY_FAILED: {str(e)}"})

    return json.dumps({"error": "ALL_SYSTEMS_BUSY (No Keys Available)"})


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
        - [Correction] means A-B-C corrective movement.
        - Do NOT treat BearishImpulse waves as bullish waves.
        - If Elliott pattern type conflicts with macro trend, support/resistance, or current price action, reduce confidence or return WAIT.
        - Wave lists are backend-filtered by timeframe recency and max shown count. Prefer the latest shown wave summary over older labels.
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
            Act as an expert Elliott Wave analyst and professional crypto signal strategist.

            {base_context}

            TASK:
            Analyze the market using:
            1. Elliott Wave structure
            2. Macro trend vs micro setup alignment
            3. Support and resistance behavior
            4. Fibonacci / ATR / volume context as supporting evidence
            5. Continuation vs correction logic
            6. Entry quality (do not force trades)

            ---> OPTIONAL GLOBAL BIAS / USER INSTRUCTION: {market_bias} <---
            Use this only as a directional preference if Elliott structure, support/resistance, Fibonacci confluence, ATR, volume, and timeframe alignment support it.
            Never force a trade only because of this instruction. If the technical setup is unclear, return WAIT.

            STRICT RULES:
            1. Return ONLY ONE final setup in trade_scenarios.
            2. Use OLD SCHEMA ONLY: entry_zone, targets, invalidation.
            3. trade_type must be LONG, SHORT, or WAIT.
            4. If setup is weak, unclear, already missed, or incomplete, return WAIT.
            5. Do NOT return LONG or SHORT without entry_zone, exactly TWO targets, and invalidation.
            6. entry_zone may be a single exact price OR a price range.
                - If entry_zone is a range, the FIRST value is the actual limit entry.
                - The first value will be used for TP/SL/RR, no-chase checks, trade status, and user display.
                - Do NOT choose a middle value from the zone as the actual entry.
                - Example: "77379.92 - 77800.00" means actual entry is 77379.92.
            7. invalidation must be ONE exact price only.
            8. targets must contain TWO prices:
                - target 1 = exact 1.5R based on the FIRST value of entry_zone and invalidation.
                - target 2 = STRICTLY the next "Major Support/Resistance Zone" or "Long Lookback" level provided in the Market Context.
            9. Never use the structure target as target 1. Only target 1 is formula-based.
            10. TP2 (Target 2) LOGIC (STRICT MARKET STRUCTURE ALIGNMENT):
                - For LONG setups: TP2 MUST be the exact price of the "Major Resistance Zone" or "Long Lookback Major Resistance Zone". Do not invent arbitrary prices.
                - For SHORT setups: TP2 MUST be the exact price of the "Major Support Zone" or "Long Lookback Major Support Zone". Do not invent arbitrary prices.
                - If the required Major Zone is too close to the entry (e.g., Risk/Reward for TP2 is less than 2.0R), the setup is trapped. You MUST return WAIT.
            11. Elliott Wave structure is the PRIMARY model. Market Structure / Fib / ATR / Volume Context is supporting evidence only.
            12. Do not create a trade only because market context shows support/resistance or Fibonacci levels.
            13. Use market context only to confirm, reject, or refine Elliott-based setups.
            14. Re-check whether the correction is complete or still ongoing before giving direction.
            15. If macro trend strongly conflicts with the setup, lower confidence or return WAIT.
            16. Entry must be realistic relative to CURRENT PRICE.
            17. If LONG entry_zone is below CURRENT PRICE, treat it as a limit pullback entry and action_advice must include the exact entry price, e.g. "Wait for pullback to 77750."
            18. If SHORT entry_zone is above CURRENT PRICE, treat it as a limit bounce entry and action_advice must include the exact entry price, e.g. "Wait for bounce to 78500."
            19. If CURRENT PRICE is closer to target 1 than to entry_zone, or price has already moved more than 50% from entry_zone toward target 1, return WAIT with action_advice = "No chase. Wait for new setup."
            20. STOP LOSS / STRUCTURAL INVALIDATION RULE (SMART S/R PLACEMENT):
                - Never set stop loss from ATR alone. It MUST be protected by a major structure or zone.
                - For LONG setups: 
                    Find the "Major Support Zone" or "Long Lookback Major Support Zone". Your SL MUST be placed strictly BELOW the BOTTOM price of this zone. Do NOT just place it below the nearest minor swing. 
                    Also, ensure it is below the "Confirm TF Swing Low" if provided in the Critical Direction Rules.
                - For SHORT setups: 
                    Find the "Major Resistance Zone" or "Long Lookback Major Resistance Zone". Your SL MUST be placed strictly ABOVE the TOP price of this zone. Do NOT just place it above the nearest minor swing.
                    Also, ensure it is above the "Confirm TF Swing High" if provided in the Critical Direction Rules.
                - Finally, add a 1.0x to 1.5x ATR buffer beyond this chosen level to prevent Liquidity Sweeps.
                - If placing the SL this far away makes the Risk/Reward unacceptable (e.g., Target 1 is closer than the SL distance), you MUST return WAIT. Do not artificially tighten the SL just to force a trade.
            21. If a LONG setup targets directly into nearby major resistance and higher timeframe bias is bearish, prefer WAIT or a SHORT bounce setup.
            22. If 1h/4h bias strongly conflicts with the 15m setup, do not take a countertrend setup unless reversal confirmation is clear.
            23. Keep action_advice short and beginner friendly. Summary must clearly explain entry, SL, TP1, and TP2 logic.
            24. For 15m or 30m setups, use the 1H confirmation layer as the immediate trend filter.
            25. If 15m entry direction conflicts with 1H structure, lower confidence or return WAIT.
            26. If 15m setup aligns with 1H and 4H structure, confidence can improve.
            27. Do not create a trade from 1H alone; 1H is confirmation, not the entry trigger.
            28. The requested timeframe is the entry trigger timeframe. Confirmation timeframes cannot generate a trade alone.
            29. For 15m/30m setups, 1H confirmation can confirm, reject, or downgrade the setup, but the actual entry must still come from 15m/30m micro structure.
            30. If 1H looks directional but the requested timeframe entry is unclear, late, or unsupported by current market context, return WAIT.
            31. FIBONACCI CONFLUENCE (THE HOLY TRINITY):
                - Elliott Waves structure is the primary model. However, you MUST look for "Confluence" (overlap) between the Fibonacci Pullback Zones (0.382, 0.5, 0.618, 0.786) and the Major Support/Resistance Zones.
                - If the planned Entry Zone aligns with a printed Fibonacci level AND a Major S/R Zone, increase confidence and explain this "Holy Trinity Confluence" in the summary.
            32. FIBONACCI PULLBACK LOGIC:
                - For Shallow Pullbacks (Fast Markets): Look for entries near the 0.382 or 0.5 Fib levels IF volume is strong.
                - For Deep Pullbacks (Normal/Slow Markets): Strictly wait for the 0.618 Golden Pocket or 0.786 Fib levels for entries. Do NOT chase early entries.
            33. FIBONACCI EXTENSIONS FOR TP:
                - Target 1 (TP1) is always the safe 1.5R exit (Formula-based).
                - Target 2 (TP2) MUST be selected based on Major Support/Resistance. However, if there is no clear nearby S/R, use the "Continuation Extension Levels" (1.0, 1.272, 1.618) as the final target.
            34. FIBONACCI AS SUPPORTING EVIDENCE ONLY:
                - Do not claim an entry is at a Fibonacci level unless the entry_zone is actually close to one of the printed Fibonacci zones from the data block.
                - If Elliott Fib Context Status is STALE or WEAK, ignore it and prioritize the "Visible Swing" Fibonacci context.
            35. PURE PRICE ACTION ENTRIES (NO VOLUME DEPENDENCY):
                - Do NOT use volume to filter out, delay, or reject new entries. 
                - If the planned entry aligns with a Major Support/Resistance Zone and a printed Fibonacci level (like the 0.618 Golden Pocket), treat it as a valid Limit Order setup.
                - Your action_advice should directly instruct the user to place a limit order at the chosen entry zone.
            36. LIMIT ORDER CONFIRMATION RULE (SNIPER ENTRIES):
                - A support/resistance zone touch is valid for an entry if confluence exists.
                - You do NOT need to wait for a "rejection candle" or wick to form if the setup has good Confluence (e.g., S/R + Fibonacci). 
                - Limit orders are placed in advance. If the Elliott structure and S/R context are valid, return the LONG/SHORT setup directly.
                - You may remind the user to "monitor the zone" in the advice, but do NOT return WAIT just because a price rejection hasn't happened yet.
            37. DOMINANT TREND PRIORITY RULE:
                - Use the DOMINANT TREND CONTEXT as the main direction filter.
                - For 15m/30m setups, 1H and 4H structure override minor 15m EMA/swing breakouts.
                - If Dominant Direction is BEARISH or Preferred Trade Side is SHORT, prefer SHORT corrective pullback setups.
                - If Dominant Direction is BULLISH or Preferred Trade Side is LONG, prefer LONG corrective pullback setups.
                - If Preferred Trade Side is WAIT, do not force LONG/SHORT unless the setup is extremely clean and confirmed.
                - Countertrend trades require confirmed reversal/rejection and cannot be High probability.
                - Do not reverse direction from a minor 15m EMA/swing breakout unless confirmed reversal exists.
            38. LEAD AI v2 GATEKEEPER RULE (ABSOLUTE PRIORITY):
                - Read the "DOMINANT TREND CONTEXT (LEAD AI v2)" block carefully.
                - If the text says "Structure Broken (CHoCH): YES", you MUST return WAIT. Do not attempt a trade.
                - If the text says "NO LONGs below EMA200", and you find a LONG setup, you MUST override it and return WAIT.
                - If the text says "NO SHORTs above Confirm TF Swing High", and you find a SHORT setup, you MUST override it and return WAIT.
                - Your output trade_type CANNOT contradict the "Preferred Trade Side" provided in the Dominant Trend Context.
            39. EXHAUSTION / DEEP PULLBACK RULE:
                - Do NOT sell the bottom or buy the top.
                - Do NOT use volume as a hard rejection filter for fresh trade decisions.
                - A 0.786 deep pullback is NOT automatically a WAIT signal.
                - If price is already extended away from the safe entry zone, or already moved more than 50% toward TP1, return WAIT / NO CHASE.
                - If price is at or near a 0.786 deep pullback, only consider a trade if Elliott structure, support/resistance, candle rejection, and structural SL are valid.
                - If the 0.786 area shows unclear reaction, weak structure, or no safe invalidation, return WAIT.
                - Volume may be treated only as soft supporting context when available, never as the main reason to reject a fresh setup.
            40. DYNAMIC TIMEFRAME RULE:
                - When analyzing, refer to the "Confirm TF" and "Swing TF" as provided in the meta data, rather than assuming hardcoded 1H or 4H values.
            41. GOLDEN CONFLUENCE (HIGH CONFIDENCE) TAG:
                - Use this tag ONLY if the setup is exceptionally strong. 
                - The entry MUST perfectly align with BOTH a "Major Support/Resistance Zone" that has a (strength 2) or (strength 3) rating, AND a printed 0.786 Deep Pullback level from the provided data.
                - Do NOT use this tag for (strength 1) zones, and do NOT mathematically invent Fibonacci levels to force this tag.
                - If it naturally meets this strict criteria, explicitly include this exact tag at the beginning of your 'summary': '🔥 GOLDEN SNIPER ENTRY: Major Structure + 0.786 Fib Confluence!'.


            OUTPUT JSON ONLY:
            {{
                "macro_analysis": {{
                    "trend": "Bullish/Bearish/Sideways",
                    "detailed_breakdown": "Explain the larger structure seen in macro data and detected waves.",
                    "key_levels": "Key support and resistance levels from macro structure."
                }},
                "micro_analysis": {{
                    "wave_count_status": "Current sub-wave count based on micro structure.",
                    "wave_personality": "Impulsive/Corrective/Unclear"
                }},
                "action_advice": "Simple, short, beginner-friendly advice.",
                "trade_scenarios": [
                    {{
                        "name": "Sniper Setup",
                        "trade_type": "LONG/SHORT/WAIT",
                        "probability": "High/Medium/Low",
                        "summary": "High-quality sniper setup aligned with macro and micro structure. TP1 is the safer 1.5R exit, while TP2 is the structure-based final target.",
                        "entry_zone": "SPECIFIC_PRICE_OR_RANGE_FIRST_VALUE_IS_ENTRY",
                        "targets": ["FIRST_TARGET_PRICE", "SECOND_TARGET_PRICE"],
                        "invalidation": "STOP_LOSS_PRICE"
                    }}
                ]
            }}
            """

        try:
            with open("debug_ai_prompt.txt", "w", encoding="utf-8") as f:
                f.write(prompt)
            print("✅ AI prompt saved to debug_ai_prompt.txt")
        except Exception as e:
            print(f"Prompt debug save failed: {e}")

        raw_text = generate_content_smart(selected_model, prompt)

        try:
            with open("debug_ai_raw_response.txt", "w", encoding="utf-8") as f:
                f.write(raw_text)
        except Exception as e:
            print(f"Raw AI debug save failed: {e}")


        if '{"error"' in raw_text:
            return raw_text

        cleaned_text = raw_text.replace("```json", "").replace("```", "").strip()
        match = re.search(r"\{.*\}", cleaned_text, re.DOTALL)

        if match:
            json_str = match.group(0)
            try:
                data = json.loads(json_str)
                data = _normalize_ai_output(data)

                if not locked_scenario:
                    data = enforce_risk_reward(data, ratio=8.0)

                    data = enforce_backend_quality_score(
                        data,
                        user_tf=user_tf,
                        market_context_text=market_context_text,
                        macro_structure_text=macro_structure_text,
                        swing_structure_text=swing_structure_text,
                        confirm_1h_structure_text=confirm_1h_structure_text,
                        micro_structure_text=micro_structure_text
                    )

                data = _normalize_ai_output(data)
                return json.dumps(data)
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