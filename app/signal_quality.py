import re

# ==========================================================
# HELPER FUNCTIONS (Numeric & Text Extraction)
# ==========================================================

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

def _extract_context_number(market_context_text, label):
    try:
        if not market_context_text: return None
        pattern = rf"{re.escape(label)}:\s*(-?\d+(?:\.\d+)?)"
        match = re.search(pattern, market_context_text)
        if match: return float(match.group(1))
    except Exception: pass
    return None

def _extract_context_line(market_context_text, label):
    try:
        if not market_context_text: return ""
        # Supports both:
        #   Normal Pullback Zone: 1.234
        #   Normal Pullback Zone (0.5): 1.234
        pattern = rf"^\s*{re.escape(label)}(?:\s*\([^:\n]*\))?\s*:\s*(.+)$"
        match = re.search(pattern, market_context_text, flags=re.MULTILINE)
        if match: return match.group(1).strip()
    except Exception: pass
    return ""

def _format_price(value):
    value = _extract_float(value)
    if value is None: return "N/A"
    if abs(value) >= 1000: return f"{value:.2f}"
    elif abs(value) >= 100: return f"{value:.2f}"
    elif abs(value) >= 10: return f"{value:.3f}"
    elif abs(value) >= 1: return f"{value:.4f}"
    elif abs(value) >= 0.01: return f"{value:.4f}"
    elif abs(value) >= 0.0001: return f"{value:.6f}"
    else: return f"{value:.8f}"

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

# ==========================================================
# ZONE & STRUCTURAL EXTRACTION
# ==========================================================

def _extract_context_zone_bounds(market_context_text, label):
    try:
        line = _extract_context_line(market_context_text, label)
        if not line: return None
        range_match = re.search(r"(-?\d+(?:\.\d+)?)\s*-\s*(-?\d+(?:\.\d+)?)", line)
        if range_match:
            a, b = _extract_float(range_match.group(1)), _extract_float(range_match.group(2))
            if a is not None and b is not None: return (min(a, b), max(a, b))
        single_match = re.search(r"(-?\d+(?:\.\d+)?)", line)
        if single_match:
            v = _extract_float(single_match.group(1))
            if v is not None: return (v, v)
    except Exception: pass
    return None

def _get_first_zone_bounds(market_context_text, labels):
    for label in labels:
        zone = _extract_context_zone_bounds(market_context_text, label)
        if zone is not None: return zone
    return None

def _zone_low(zone):
    return min(float(zone[0]), float(zone[1])) if zone else None

def _zone_high(zone):
    return max(float(zone[0]), float(zone[1])) if zone else None

def _collect_structural_sl_reference(market_context_text, trade_type, entry, atr=None):
    try:
        trade_type = str(trade_type or "").upper()
        entry = _extract_float(entry)
        atr = _extract_float(atr)
        if entry is None: return None

        min_gap = max(atr * 0.80, abs(entry) * 0.0004) if atr and atr > 0 else abs(entry) * 0.0008

        def get_val(label, is_zone=False, use_low=True):
            if is_zone:
                z = _extract_context_zone_bounds(market_context_text, label)
                return _zone_low(z) if use_low else _zone_high(z)
            else:
                return _extract_context_number(market_context_text, label)

        if trade_type == "LONG":
            # Priority order as per logical hierarchy
            priorities = [
                ("Recent Swing Low", False, True),
                ("Visible Deep Pullback", True, True),
                ("Deep Pullback", True, True),
                ("Long Lookback Major Support Zone", True, True),
                ("Major Support Zone", True, True),
                ("Visible Normal Pullback Zone", True, True),
                ("Normal Pullback Zone", True, True),
                ("Nearest Support", False, True) # Fallback
            ]
            for label, is_zone, use_low in priorities:
                v = get_val(label, is_zone, use_low)
                if v is not None and v < entry and (entry - v) >= min_gap:
                    return v

        if trade_type == "SHORT":
            # Priority order as per logical hierarchy
            priorities = [
                ("Recent Swing High", False, False),
                ("Visible Deep Pullback", True, False),
                ("Deep Pullback", True, False),
                ("Long Lookback Major Resistance Zone", True, False),
                ("Major Resistance Zone", True, False),
                ("Visible Normal Pullback Zone", True, False),
                ("Normal Pullback Zone", True, False),
                ("Nearest Resistance", False, False) # Fallback
            ]
            for label, is_zone, use_low in priorities:
                v = get_val(label, is_zone, use_low)
                if v is not None and v > entry and (v - entry) >= min_gap:
                    return v

    except Exception: pass
    return None

def _zone_margin(current_price, atr):
    try:
        price = _extract_float(current_price) or 0
        atr_val = _extract_float(atr) or 0
        return max(atr_val * 0.15, abs(price) * 0.0005)
    except Exception: return 0

def _price_in_or_near_zone(price, zone, margin=0):
    price = _extract_float(price)
    return False if price is None or zone is None else (zone[0] - margin) <= price <= (zone[1] + margin)

def _price_broke_above_zone(price, zone, margin=0):
    price = _extract_float(price)
    return False if price is None or zone is None else price > zone[1] + margin

def _price_broke_below_zone(price, zone, margin=0):
    price = _extract_float(price)
    return False if price is None or zone is None else price < zone[0] - margin

def _extract_context_bool(market_context_text, label):
    try:
        line = _extract_context_line(market_context_text, label).upper().strip()
        if line.startswith("YES"): return True
        if line.startswith("NO"): return False
    except Exception: pass
    return None

def _extract_last_closed_price(market_context_text):
    """Extracts the Close price of the last completed candle to check for body breaks vs wicks."""
    try:
        if not market_context_text: return None
        # Looks for: Last Closed Candle O/H/L/C: 8.43 / 8.45 / 8.43 / 8.44
        pattern = r"Last Closed Candle O/H/L/C:\s*[\d\.]+\s*/\s*[\d\.]+\s*/\s*[\d\.]+\s*/\s*([\d\.]+)"
        match = re.search(pattern, market_context_text)
        if match:
            return float(match.group(1))
    except Exception: pass
    return None

def _force_backend_wait(ai_data, reason, market_context_text=""):
    ai_data["action_advice"] = f"WAIT. {reason}"
    ai_data["trade_scenarios"] = [_force_wait_scenario(reason)]
    return ai_data

def _downgrade_probability(setup, reason):
    if str(setup.get("probability", "")).lower() == "high":
        setup["probability"] = "Medium"
    summary = str(setup.get("summary", ""))
    if reason and reason not in summary:
        setup["summary"] = f"{summary} Backend note: {reason}"

# ==========================================================
# LEAD AI v2: CHOCH AND EXHAUSTION FILTER (UNIVERSAL)
# ==========================================================
def _is_trend_exhausted(market_context_text, trade_type):
    """
    Fresh trade hard-block.

    IMPORTANT:
    - New trades must NOT be filtered/rejected by volume because volume changes quickly.
    - Volume is used only in locked/running trade management inside main.py.
    - Here we only hard-block confirmed structure break / CHoCH.
    """
    try:
        is_broken = _extract_context_line(market_context_text, "Structure Broken (CHoCH)")
        if is_broken and "YES" in is_broken.upper():
            return True, "Structure broken (CHoCH detected). Wait for trend reset."

        return False, ""
    except Exception:
        return False, ""

# ==========================================================
# MAIN QUALITY ENFORCEMENT ENGINE
# ==========================================================
def enforce_backend_quality_score(
    ai_data,
    user_tf="15m",
    market_context_text="",
    macro_structure_text="",
    swing_structure_text="",
    confirm_1h_structure_text="",
    micro_structure_text=""
):
    try:
        if not isinstance(ai_data, dict): return ai_data
        scenarios = ai_data.get("trade_scenarios", [])
        if not scenarios or not isinstance(scenarios[0], dict): return ai_data

        setup = scenarios[0]
        trade_type = str(setup.get("trade_type", "")).upper().strip()

        if trade_type not in ["LONG", "SHORT"]: return ai_data

        # --- LEAD AI v2: EXHAUSTION / CHOCH HARD BLOCK ---
        is_exhausted, exhaust_reason = _is_trend_exhausted(market_context_text, trade_type)
        if is_exhausted:
            return _force_backend_wait(ai_data, f"LEAD AI FILTER: Setup rejected! {exhaust_reason}", market_context_text)
        # -------------------------------------------------

        entry = _extract_float(setup.get("entry_zone"))
        stop = _extract_float(setup.get("invalidation"))
        raw_targets = setup.get("targets", [])
        tp1 = _extract_float(raw_targets[0]) if raw_targets else None

        current_price = _extract_context_number(market_context_text, "Current Price")
        atr = _extract_context_number(market_context_text, "ATR(14)")
        last_close_price = _extract_last_closed_price(market_context_text) # For Wick vs Body detection

        if entry is None or stop is None or abs(entry - stop) <= 0:
            return _force_backend_wait(ai_data, "Invalid entry or stop loss.", market_context_text)
            
        risk = abs(entry - stop)
        score = 0
        notes = []
        hard_wait_reasons = []

        # --- 0. FRONT-RUNNING (LIQUIDITY GAP) LOGIC FOR WEAK S/R BREAKOUTS ---
        # අපි මෙතැනදී බලනවා මේක "Minor S/R Override" එකක්ද කියලා AI එකෙන් එන Summary එකෙන්.
        if atr and atr > 0 and "Minor S/R overridden" in str(setup.get("summary", "")):
            gap = atr * 0.20 # <-- මෙතැන 0.15 වෙනුවට 0.20 දැම්මා (20% of ATR as gap)
            if trade_type == "LONG":
                # Breakout එකට කලින් ෆිල් වෙන්න පොඩ්ඩක් පල්ලෙහයින් තියනවා
                entry = entry - gap
            elif trade_type == "SHORT":
                entry = entry + gap
            
            setup["entry_zone"] = _format_price(entry)
            setup["summary"] += f" [Liquidity Gap Applied: Entry adjusted to {_format_price(entry)} to avoid wicks]"
            notes.append("Liquidity Gap (Front-run) applied for weak S/R (0.2 ATR).")

        # --- 0.5 SHALLOW ENTRY / RELIEF BOUNCE FILTER ---
        swing_high = _extract_context_number(market_context_text, "Recent Swing High")
        swing_low = _extract_context_number(market_context_text, "Recent Swing Low")
        
        if swing_high is not None and swing_low is not None and atr and atr > 0:
            impulse_size = abs(swing_high - swing_low)
            
            # If recent move is very strong (> 2 ATR), block shallow entries to avoid traps
            if impulse_size > (2.0 * atr):
                # 1. Check Fib Shallow Zones
                shallow_zone = _extract_context_zone_bounds(market_context_text, "Shallow Pullback Zone")
                if not shallow_zone:
                    shallow_zone = _extract_context_zone_bounds(market_context_text, "Visible Shallow Pullback Zone")
                
                if shallow_zone and entry is not None:
                    # Allow 15% ATR margin to check if it's strictly inside the shallow area
                    if (shallow_zone[0] - (atr*0.15)) <= entry <= (shallow_zone[1] + (atr*0.15)):
                        hard_wait_reasons.append(
                            f"Strong momentum move detected ({impulse_size:.6f} move, {impulse_size/atr:.2f}x ATR). "
                            "Shallow entry rejected to avoid relief-bounce trap. Wait for deeper pullback."
                        )

                # 2. EXTENDED: Check Nearest S/R only if entry is NOT also near deeper/major zones
                deep_zone = _extract_context_zone_bounds(market_context_text, "Deep Pullback")
                visible_deep_zone = _extract_context_zone_bounds(market_context_text, "Visible Deep Pullback")
                normal_zone = _extract_context_zone_bounds(market_context_text, "Normal Pullback Zone")
                visible_normal_zone = _extract_context_zone_bounds(market_context_text, "Visible Normal Pullback Zone")
                major_res_zone = _extract_context_zone_bounds(market_context_text, "Major Resistance Zone")
                major_sup_zone = _extract_context_zone_bounds(market_context_text, "Major Support Zone")

                def near_any_good_zone(entry_price, zones, margin):
                    for z in zones:
                        if z and (z[0] - margin) <= entry_price <= (z[1] + margin):
                            return True
                    return False

                good_zone_margin = atr * 0.25

                if trade_type == "SHORT":
                    nearest_res = _extract_context_number(market_context_text, "Nearest Resistance")
                    is_near_good_zone = near_any_good_zone(
                        entry,
                        [deep_zone, visible_deep_zone, normal_zone, visible_normal_zone, major_res_zone],
                        good_zone_margin
                    )

                    if nearest_res is not None and entry is not None and not is_near_good_zone:
                        if abs(entry - nearest_res) <= (atr * 0.20):
                            hard_wait_reasons.append(
                                "Strong bearish momentum detected. Entry is only near minor nearest resistance, "
                                "not a normal/deep pullback or major resistance. Wait for deeper pullback."
                            )

                elif trade_type == "LONG":
                    nearest_sup = _extract_context_number(market_context_text, "Nearest Support")
                    is_near_good_zone = near_any_good_zone(
                        entry,
                        [deep_zone, visible_deep_zone, normal_zone, visible_normal_zone, major_sup_zone],
                        good_zone_margin
                    )

                    if nearest_sup is not None and entry is not None and not is_near_good_zone:
                        if abs(entry - nearest_sup) <= (atr * 0.20):
                            hard_wait_reasons.append(
                                "Strong bullish momentum detected. Entry is only near minor nearest support, "
                                "not a normal/deep pullback or major support. Wait for deeper pullback."
                            )

        # 1. No-Chase Logic
        progress = 0
        is_missed = False
        if current_price is not None and tp1 is not None:
            if trade_type == "LONG" and tp1 > entry and current_price > entry:
                progress = (current_price - entry) / (tp1 - entry)
            elif trade_type == "SHORT" and tp1 < entry and current_price < entry:
                progress = (entry - current_price) / (entry - tp1)
            
            if progress > 0.50:
                is_missed = True
                hard_wait_reasons.append(f"Price already moved {progress * 100:.1f}% toward TP1 (Missed Setup).")

        # 2. Structural SL Adjustment & FLIP LOGIC
        structural_ref = _collect_structural_sl_reference(market_context_text, trade_type, entry, atr=atr)
        
        if structural_ref and atr and atr > 0:
            structural_buffer = max(atr * 0.35, abs(entry) * 0.0002)
            pivot_gap = atr * 0.15 # 15% ATR Gap to distinguish wick from body break

            required_stop = (structural_ref - structural_buffer) if trade_type == "LONG" else (structural_ref + structural_buffer)
            stop_is_valid = (stop <= required_stop) if trade_type == "LONG" else (stop >= required_stop)
            
            # --- MISSED TRADE FLIP LOGIC (Candle Body vs Wick) ---
            if is_missed and last_close_price is not None:
                if trade_type == "LONG" and last_close_price < (structural_ref - pivot_gap):
                    setup["trade_type"] = "SHORT"
                    setup["entry_zone"] = _format_price(last_close_price) 
                    setup["invalidation"] = _format_price(structural_ref + structural_buffer)
                    new_risk = abs(last_close_price - (structural_ref + structural_buffer))
                    setup["targets"] = [_format_price(last_close_price - (new_risk * 1.5)), _format_price(last_close_price - (new_risk * 3.0))]
                    setup["summary"] = "🔥 FLIP ALERT: Missed LONG flipped to SHORT due to structural support breaking with a candle body close."
                    setup["probability"] = "High"
                    hard_wait_reasons.clear() 
                    score += 5
                    notes.append("Setup flipped to SHORT.")
                
                elif trade_type == "SHORT" and last_close_price > (structural_ref + pivot_gap):
                    setup["trade_type"] = "LONG"
                    setup["entry_zone"] = _format_price(last_close_price)
                    setup["invalidation"] = _format_price(structural_ref - structural_buffer)
                    new_risk = abs(last_close_price - (structural_ref - structural_buffer))
                    setup["targets"] = [_format_price(last_close_price + (new_risk * 1.5)), _format_price(last_close_price + (new_risk * 3.0))]
                    setup["summary"] = "🔥 FLIP ALERT: Missed SHORT flipped to LONG due to structural resistance breaking with a candle body close."
                    setup["probability"] = "High"
                    hard_wait_reasons.clear()
                    score += 5
                    notes.append("Setup flipped to LONG.")

            # Normal SL Check (If not flipped)
            if not is_missed or len(hard_wait_reasons) > 0:
                if stop_is_valid:
                    score += 1
                    notes.append(f"Structural SL valid.")
                else:
                    adjusted_risk = abs(entry - required_stop)
                    adjusted_sl_atr = adjusted_risk / atr if atr else 0
                    
                    if adjusted_sl_atr <= 3.5:
                        setup["invalidation"] = _format_price(required_stop)
                        setup["structural_sl_adjusted"] = True
                        setup["summary"] = str(setup.get("summary", "")) + f" [SL adjusted to structural {required_stop}]"
                        score += 1
                        notes.append("Structural SL auto-adjusted.")
                    else:
                        hard_wait_reasons.append("SL is not beyond structural support/resistance and adjustment is too wide.")
        else:
            notes.append("Structural SL reference unavailable.")

        # 3. Final Backend Decision
        setup["quality_score"] = score
        setup["quality_notes"] = " | ".join(notes)

        if hard_wait_reasons:
            return _force_backend_wait(ai_data, "Backend Quality Filter: " + "; ".join(hard_wait_reasons), market_context_text)

        if score < 0:
            return _force_backend_wait(ai_data, f"Low quality score ({score}). Trade rejected.", market_context_text)

        if score in [0, 1, 2]:
            setup["probability"] = "Medium"
            setup["summary"] += " [Backend note: Medium confidence setup. Wait for confirmation.]"
            ai_data["trade_scenarios"][0] = setup
            return ai_data

        if score >= 3:
            setup["probability"] = "High"
            setup["summary"] += " [Backend note: Strong setup.]"
            ai_data["trade_scenarios"][0] = setup
            return ai_data

    except Exception as e:
        print(f"Backend quality score filter failed: {e}")
        return ai_data