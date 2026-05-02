import streamlit as st
import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import ccxt
import pandas as pd
import google.generativeai as genai
import json
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import time
import firebase_admin
from firebase_admin import credentials, firestore

# --- FIREBASE SETUP ---
FIREBASE_CREDENTIALS_PATH = os.getenv("FIREBASE_CREDENTIALS_PATH", os.path.join(PROJECT_ROOT, "serviceAccountKey.json"))

db = None
FIREBASE_AVAILABLE = False

try:
    if not firebase_admin._apps:
        cred = credentials.Certificate(FIREBASE_CREDENTIALS_PATH)
        firebase_admin.initialize_app(cred)
    db = firestore.client()
    FIREBASE_AVAILABLE = True
except Exception as e:
    FIREBASE_AVAILABLE = False

# --- PDF LIBRARY ---
try:
    from fpdf import FPDF
except ImportError:
    st.error("❌ 'fpdf' library not found! Please add 'fpdf' to your requirements.txt file.")
    st.stop()

# --- IMPORT KNOWLEDGE BASE ---
try:
    from app.knowledge import ELLIOTT_KNOWLEDGE
except Exception as e:
    st.error(f"❌ Failed to import knowledge.py: {e}")
    st.stop()


def load_api_key_from_firebase():
    if not FIREBASE_AVAILABLE or db is None:
        return None

    try:
        doc_ref = db.collection("app_secrets").document("config")
        doc = doc_ref.get()

        if not doc.exists:
            return None

        data = doc.to_dict() or {}

        pool = data.get("gemini_api_key_pool", [])
        if isinstance(pool, list) and len(pool) > 0:
            return pool[0]

        paid_key = data.get("gemini_paid_key")
        if paid_key:
            return paid_key

    except Exception as e:
        st.warning(f"Firebase API key load failed: {e}")

    return None

# ==========================================
# 1. CONFIGURATION
# ==========================================

st.set_page_config(page_title="Elliott Wave Dashboard", layout="wide", page_icon="🌊")

st.markdown("""
<style>
    .stApp { background-color: #0E1117; }
    .metric-card {
        background-color: #1e2130; border: 1px solid #2b2f42;
        padding: 15px; border-radius: 8px; color: white; margin-bottom: 10px;
    }
    .deep-analysis-box {
        background-color: #151922; border-left: 4px solid #7c4dff;
        padding: 15px; border-radius: 5px; margin-bottom: 10px;
    }
    .scenario-card {
        background-color: #262730; padding: 20px; border-radius: 10px;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# --- API KEY ---
API_KEY = load_api_key_from_firebase()

if not API_KEY:
    API_KEY = st.sidebar.text_input("Enter Google API Key", type="password")

if not API_KEY:
    st.warning("⚠️ API Key Required. Firebase or Sidebar එකෙන් key එක දෙන්න.")
    st.stop()

try:
    genai.configure(api_key=API_KEY)
    MODEL_NAME = 'gemini-2.5-flash'
except Exception as e:
    st.error(f"API Configuration Error: {e}")
    st.stop()

try:
    genai.configure(api_key=API_KEY)
    MODEL_NAME = 'gemini-2.5-flash' 
except Exception as e:
    st.error(f"API Configuration Error: {e}")
    st.stop()

# ==========================================
# 2. STATE & DATA FUNCTIONS
# ==========================================

if "ai_data" not in st.session_state: st.session_state.ai_data = None
if "df_micro" not in st.session_state: st.session_state.df_micro = None
if "last_update" not in st.session_state: st.session_state.last_update = None
if "chat_history" not in st.session_state: st.session_state.chat_history = []

# --- CACHING ADDED FOR SPEED ---
@st.cache_data(ttl=300) # Data stays in memory for 300 seconds (5 mins)
def get_crypto_data(symbol, timeframe, limit=1000):
    """
    Fetches crypto data using multiple exchanges with Caching.
    """
    exchanges_to_try = [
        ccxt.binance(),
        ccxt.okx(),
        ccxt.kraken(),
        ccxt.kucoin()
    ]

    for exchange in exchanges_to_try:
        try:
            exchange.enableRateLimit = True
            bars = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
            if bars:
                df = pd.DataFrame(bars, columns=['time', 'open', 'high', 'low', 'close', 'volume'])
                df['time'] = pd.to_datetime(df['time'], unit='ms')
                return df
        except Exception as e:
            continue
    return pd.DataFrame()

def analyze_deep_wave(symbol, micro_tf, df_1w, df_1d, df_micro, language, previous=None):
    
    lang_inst = "Explain in English."
    if language == "Singlish":
        lang_inst = "Explain in 'Singlish' (Sinhala mixed with English). Use technical terms freely."

    json_1w = df_1w.tail(30).to_json(orient="records", date_format='iso')
    json_1d = df_1d.tail(60).to_json(orient="records", date_format='iso')

    cols = ['open', 'high', 'low', 'close', 'volume']
    micro_subset = df_micro.tail(300).copy()
    micro_subset[cols] = micro_subset[cols].round(4)
    json_micro = micro_subset.to_json(orient="records", date_format='iso')

    task = "### TASK: DETAILED ELLIOTT WAVE STRUCTURE ANALYSIS"
    
    scalp_instruction = ""
    if micro_tf in ['1m', '3m', '5m']:
        scalp_instruction = """
        **⚠️ SCALPING MODE ACTIVE (1m-5m):**
        - You are provided with 300 candles of micro data.
        - Analyze the immediate wave structure carefully.
        - Check for Volume Divergence on the last wave.
        - Ensure the trade direction aligns with the 1D Swing Trend.
        """

    prompt = f"""
    You are a Master Elliott Wave Analyst.
    {task}
    
    ### DATA INPUTS
    * **1W (Trend Context):** {json_1w}
    * **1D (Swing Context):** {json_1d}
    * **MICRO ({micro_tf} - Last 300 Candles):** {json_micro}
    
    ### INSTRUCTIONS
    1.  **Structure:** Analyze the provided 300 candles to determine the exact wave count.
    2.  **Validations:** Check High/Low relationships, Fibonacci Time cycles, and Volume.
    3.  **Consistency:** Ensure the Micro count fits into the 1D Swing structure.
    4.  **rule** Only use elliott wave and support and resistance. also make sure give the datiled scenario in "trade_scenarios"
    {scalp_instruction}
    
    ### LANGUAGE
    {lang_inst}
    
    ### OUTPUT JSON FORMAT (MANDATORY)
    {{
        "macro_analysis": {{
            "trend": "Bullish/Bearish",
            "current_structure": "Primary Wave Count",
            "detailed_breakdown": "History of the move.",
            "key_levels": "Key Support/Res"
        }},
        "micro_analysis": {{
            "timeframe": "{micro_tf}",
            "current_wave_degree": "Degree (e.g. Sub-Minuette)",
            "wave_count_status": "Status (e.g. Wave iii extending)",
            "sub_wave_structure": "Internal structure description.",
            "fib_confluence": "Fib levels aligned.",
            "volume_validation": "Volume analysis."
        }},
        "trade_scenarios": [
            {{
                "name": "Primary Setup",
                "trade_type": "Long/Short",
                "probability": "High",
                "summary": "Reasoning.",
                "entry_zone": "0.00 - 0.00",
                "target": 0.00,
                "invalidation": 0.00,
                "color": "#00E676"
            }},
            {{
                "name": "Alternative",
                "trade_type": "Long/Short",
                "probability": "Low",
                "summary": "Alt count.",
                "entry_zone": "0.00 - 0.00",
                "target": 0.00,
                "invalidation": 0.00,
                "color": "#FFAB00"
            }}
        ]
    }}
    """
    
    try:
        model = genai.GenerativeModel(MODEL_NAME, system_instruction=ELLIOTT_KNOWLEDGE)
        response = model.generate_content(prompt, generation_config={"temperature": 0.2, "response_mime_type": "application/json"})
        return json.loads(response.text)
    except Exception as e:
        st.error(f"AI Analysis Error: {e}")
        return None

# --- PDF GENERATOR FUNCTION (FIXED) ---
def create_pdf(macro, micro, scenarios, symbol, tf):
    class PDF(FPDF):
        def header(self):
            self.set_font('Arial', 'B', 15)
            self.cell(0, 10, 'Elliott Wave AI - Analysis Report', 0, 1, 'C')
            self.ln(10)
            
    # --- Text Cleaning Helper ---
    # මෙය මගින් Emojis සහ සිංහල අකුරු ඉවත් කර PDF එක Crash වීම වලක්වයි
    def clean_text(text):
        if text:
            # Latin-1 වලට හරවන්න උත්සාහ කරයි. බැරි අකුරු (Emojis/Sinhala) '?' ලකුණෙන් ආදේශ කරයි.
            return str(text).encode('latin-1', 'replace').decode('latin-1')
        return ""
    
    pdf = PDF()
    pdf.add_page()
    pdf.set_font("Arial", size=11)
    
    # 1. Header Info
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, clean_text(f"Symbol: {symbol}  |  Timeframe: {tf}"), 0, 1)
    pdf.cell(0, 10, clean_text(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}"), 0, 1)
    pdf.line(10, 35, 200, 35)
    pdf.ln(5)
    
    # 2. Macro Analysis
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "1. Macro Trend (1W/1D)", 0, 1)
    pdf.set_font("Arial", size=10)
    
    # Multi-cell භාවිතා කරන විට clean_text අනිවාර්යයි
    macro_text = (
        f"Trend: {macro.get('trend')}\n"
        f"Structure: {macro.get('current_structure')}\n"
        f"Details: {macro.get('detailed_breakdown')}\n"
        f"Key Levels: {macro.get('key_levels')}"
    )
    pdf.multi_cell(0, 7, clean_text(macro_text))
    pdf.ln(5)
    
    # 3. Micro Analysis
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, clean_text(f"2. Micro Structure ({tf})"), 0, 1)
    pdf.set_font("Arial", size=10)
    
    micro_text = (
        f"Wave Degree: {micro.get('current_wave_degree')}\n"
        f"Status: {micro.get('wave_count_status')}\n"
        f"Structure: {micro.get('sub_wave_structure')}\n"
        f"Volume: {micro.get('volume_validation')}"
    )
    pdf.multi_cell(0, 7, clean_text(micro_text))
    pdf.ln(5)
    
    # 4. Scenarios
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "3. Trade Setups", 0, 1)
    
    for s in scenarios:
        pdf.set_font("Arial", 'B', 10)
        pdf.set_text_color(0, 0, 139) # Dark Blue
        pdf.cell(0, 10, clean_text(f"> {s.get('name')} ({s.get('trade_type')})"), 0, 1)
        pdf.set_text_color(0, 0, 0) # Black
        pdf.set_font("Arial", size=10)
        
        scenario_text = (
            f"Reasoning: {s.get('summary')}\n"
            f"Entry: {s.get('entry_zone')}\n"
            f"Target: {s.get('target')}\n"
            f"Invalidation: {s.get('invalidation')}"
        )
        pdf.multi_cell(0, 7, clean_text(scenario_text))
        pdf.ln(2)
        
    pdf.ln(10)
    pdf.set_font("Arial", 'I', 8)
    pdf.cell(0, 10, "Generated by Deep Wave AI. Not financial advice.", 0, 1, 'C')

    return pdf.output(dest='S').encode('latin-1', 'replace')
# ==========================================
# 3. UI LAYOUT
# ==========================================

with st.sidebar:
    st.title("🌊 Deep Wave AI")
    lang = st.radio("Language", ["Singlish", "English"], index=0)
    
    # --- UPDATED: SEARCH ANY SYMBOL ---
    st.subheader("Asset Selection")
    sym_input = st.text_input("Symbol (e.g. BTC/USDT)", value="BTC/USDT")
    sym = sym_input.upper().strip() # Convert to uppercase
    
    st.subheader("Timeframe")
    tf = st.selectbox(
        "Select Micro Chart", 
        ["1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d"], 
        index=0 
    )
    
    run = st.button("🚀 Analyze Structure", type="primary")
    
    st.divider()
    
    with st.expander("💬 Chat about Structure", expanded=True):
        if st.session_state.ai_data:
            for m in st.session_state.chat_history:
                st.chat_message(m['role']).write(m['content'])
            
            q = st.chat_input("Ask about the wave count...")
            if q:
                st.session_state.chat_history.append({"role": "user", "content": q})
                st.chat_message("user").write(q)
                
                with st.spinner("AI thinking..."):
                    try:
                        mod = genai.GenerativeModel(MODEL_NAME)
                        context_summary = {
                            "macro": st.session_state.ai_data.get('macro_analysis'),
                            "micro": st.session_state.ai_data.get('micro_analysis'),
                            "scenarios": st.session_state.ai_data.get('trade_scenarios')
                        }
                        context_str = json.dumps(context_summary)
                        res = mod.generate_content(f"You are an Elliott Wave expert. Context: {context_str}. User Question: {q}. Language: {lang}")
                        
                        st.session_state.chat_history.append({"role": "assistant", "content": res.text})
                        st.chat_message("assistant").write(res.text)
                        
                    except Exception as e:
                        st.error(f"Chat Error: {e}")
        else:
            st.info("⚠️ Please run an analysis first to enable chat.")
        
        if st.button("Clear Chat"): 
            st.session_state.chat_history = []
            st.rerun()

# ==========================================
# 4. MAIN PAGE
# ==========================================

st.title(f"🌊 {sym} Deep Analysis ({tf})")

if run:
    with st.spinner(f"📡 Fetching Data for {sym}..."):
        d1w = get_crypto_data(sym, "1w", 200)
        d1d = get_crypto_data(sym, "1d", 300)
        dm = get_crypto_data(sym, tf, 1000) 
        
        if not dm.empty and not d1w.empty:
            st.session_state.df_micro = dm
            st.session_state.df_1w = d1w
            st.session_state.df_1d = d1d
            
            with st.spinner("🧠 AI Analyzing Elliott Waves (Last 300 Candles)..."):
                ai = analyze_deep_wave(sym, tf, d1w, d1d, dm, lang)
                if ai:
                    st.session_state.ai_data = ai
                    st.session_state.last_update = datetime.now().strftime("%H:%M:%S")
                    st.session_state.chat_history = [] 
                    st.rerun()
                else:
                    st.error("❌ AI Analysis Failed. Please try again.")
        else:
            st.error(f"❌ Failed to fetch data for {sym}.")
            st.warning("⚠️ All exchanges failed. Please check the symbol and try again.")

# --- DISPLAY RESULTS ---
if st.session_state.ai_data:
    data = st.session_state.ai_data
    macro = data.get('macro_analysis', {})
    micro = data.get('micro_analysis', {})
    scenarios = data.get('trade_scenarios', [])
    
    # Update Button
    c1, c2 = st.columns([5,1])
    c1.caption(f"Last Update: {st.session_state.last_update}")
    
    # --- PDF DOWNLOAD BUTTON ---
    try:
        pdf_bytes = create_pdf(macro, micro, scenarios, sym, tf)
        c2.download_button(
            label="📥 Download PDF",
            data=pdf_bytes,
            file_name=f"{sym}_{tf}_Analysis.pdf",
            mime="application/pdf"
        )
    except Exception as e:
        c2.error(f"PDF Error: {e}")
    
    # --- 1. MACRO BREAKDOWN ---
    st.markdown("### 🌍 Macro Structure (1W / 1D)")
    st.markdown(f"""
    <div class='metric-card'>
        <h3>{macro.get('trend')} - {macro.get('current_structure')}</h3>
        <p style='color:#bbb;'>{macro.get('detailed_breakdown')}</p>
        <small>🔑 Key Levels: {macro.get('key_levels')}</small>
    </div>
    """, unsafe_allow_html=True)
    
    # --- 2. MICRO DEEP DIVE ---
    st.markdown(f"### 🔬 Micro Wave Analysis ({tf})")
    
    m1, m2 = st.columns(2)
    with m1:
        st.markdown(f"""
        <div class='deep-analysis-box'>
            <b>🌊 Current Wave Degree:</b> {micro.get('current_wave_degree')}<br>
            <b>📍 Status:</b> {micro.get('wave_count_status')}
        </div>
        """, unsafe_allow_html=True)
    with m2:
        st.markdown(f"""
        <div class='deep-analysis-box'>
            <b>📐 Fib Confluence:</b> {micro.get('fib_confluence')}<br>
            <b>📊 Volume:</b> {micro.get('volume_validation')}
        </div>
        """, unsafe_allow_html=True)

    st.info(f"**📝 Sub-Wave Structure:** {micro.get('sub_wave_structure')}")

    # --- 3. CHART & SCENARIOS ---
    tab1, tab2 = st.tabs(["📈 Chart (Price + Volume)", "🛠 Trade Setup"])
    
    with tab1:
        if st.session_state.df_micro is not None:
            df = st.session_state.df_micro.tail(300)
            
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.7, 0.3])
            
            fig.add_trace(go.Candlestick(x=df['time'], open=df['open'], high=df['high'], low=df['low'], close=df['close'], name="Price"), row=1, col=1)
            colors = ['#00E676' if r['close'] >= r['open'] else '#FF5252' for i, r in df.iterrows()]
            fig.add_trace(go.Bar(x=df['time'], y=df['volume'], marker_color=colors, name="Volume"), row=2, col=1)
            
            if scenarios:
                prim = scenarios[0]
                try:
                    fig.add_hline(y=float(prim['target']), line_dash="dash", line_color="#00E676", row=1, col=1, annotation_text="TP")
                    fig.add_hline(y=float(prim['invalidation']), line_dash="dot", line_color="#FF5252", row=1, col=1, annotation_text="Invalidation")
                except: pass
            
            fig.update_layout(height=600, template="plotly_dark", title=f"{sym} {tf} Analysis", xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True)
        
    with tab2:
        if scenarios:
            for s in scenarios:
                st.markdown(f"""
                <div class='scenario-card' style='border-left: 5px solid {s.get('color')};'>
                    <h3>{s.get('name')} ({s.get('trade_type')})</h3>
                    <p>{s.get('summary')}</p>
                    <div style='background:#111; padding:10px; border-radius:5px;'>
                        🔵 Entry: {s.get('entry_zone')}<br>
                        🟢 Target: {s.get('target')}<br>
                        🔴 Invalidation: {s.get('invalidation')}
                    </div>
                </div>
                """, unsafe_allow_html=True)

# --- DISCLAIMER ---
st.markdown("---")
st.caption("⚠️ **Disclaimer:** This application provides technical analysis based on Elliott Wave Theory using AI. Cryptocurrency trading involves high risk. This is not financial advice. Please do your own research before trading.")


