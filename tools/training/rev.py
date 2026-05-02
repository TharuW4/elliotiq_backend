import streamlit as st
import json
import pandas as pd
import plotly.graph_objects as go
import os

DATA_FILE = 'elliott_impulse_dataset.json'
st.set_page_config(layout="wide", page_title="Elliott Wave Strict Labeler")

if 'data' not in st.session_state:
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, 'r') as f:
            st.session_state.data = json.load(f)
    else:
        st.session_state.data = []
    st.session_state.current_index = 0

def save_data():
    with open(DATA_FILE, 'w') as f:
        json.dump(st.session_state.data, f)

if not st.session_state.data:
    st.error("No data found! Delete the old json and run data_gen.py again.")
    st.stop()

idx = st.session_state.current_index
if idx >= len(st.session_state.data): idx = len(st.session_state.data) - 1
sample = st.session_state.data[idx]

# --- PLOTTING ---
df = pd.DataFrame(sample['data'], columns=['Open', 'High', 'Low', 'Close', 'Volume'])
df.index.name = 'Date'

wave_indices = sample['wave_indices'] 
wave_prices = sample['wave_prices']

fig = go.Figure()

# Market Data
fig.add_trace(go.Candlestick(
    x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
    name='Price'
))

# Wave Lines
fig.add_trace(go.Scatter(
    x=wave_indices, 
    y=wave_prices,
    mode='lines+markers+text',
    line=dict(color='#2962FF', width=2),
    marker=dict(size=10, color='#FFD600'),
    text=['0', '1', '2', '3', '4', '5'],
    textposition="top center",
    name='Elliott Wave'
))

fig.update_layout(
    title=f"{sample['symbol']} - Valid Impulse? ({idx+1}/{len(st.session_state.data)})",
    xaxis_rangeslider_visible=False,
    height=600,
    template="plotly_dark",
    margin=dict(l=20, r=20, t=40, b=20)
)

st.plotly_chart(fig, use_container_width=True)

# --- CONTROLS ---
c1, c2, c3 = st.columns(3)
if c1.button("⬅️ Prev"):
    if idx > 0: 
        st.session_state.current_index -= 1
        st.rerun()

if c2.button("🗑️ Delete (Incorrect)", type="primary"):
    del st.session_state.data[idx]
    save_data()
    st.rerun()

if c3.button("✅ Keep (Correct)", type="secondary"):
    if idx < len(st.session_state.data) - 1:
        st.session_state.current_index += 1
        st.rerun()