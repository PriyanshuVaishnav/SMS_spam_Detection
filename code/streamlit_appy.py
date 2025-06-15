import streamlit as st
import pandas as pd
import plotly.express as px
from main import predict
from pathlib import Path

st.set_page_config(page_title="SMS Spam Detector", layout="centered")

# ------------------ Header ------------------
st.markdown(
    """
    # 📩 SMS Spam Detection Application

    This tool helps you detect whether an SMS message is **spam** 🚫 or **ham** ✅ (not spam).  
    Just paste your message below and click **"Classify"**.
    """
)

# ------------------ Text Input ------------------
message = st.text_area("✉️ Enter an SMS message", height=150)

# ------------------ Prediction Button ------------------
if st.button("🔍 Classify Message"):
    if not message.strip():
        st.warning("Please en
