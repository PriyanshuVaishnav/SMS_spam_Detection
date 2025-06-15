import streamlit as st
import pandas as pd
import plotly.express as px
from main import predict



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
        st.warning("Please enter a valid message.")
    else:
        with st.spinner("Analyzing..."):
            try:
                result = predict(message)
                if result == 1:
                    st.error("🚫 The message is **SPAM**!")
                else:
                    st.success("✅ The message is **HAM** (not spam).")
            except FileNotFoundError:
                st.error("❗ Model not found. Please run training first (step 2).")
            except Exception as e:
                st.exception(e)

# ------------------ Optional Data Preview ------------------
with st.expander("📊 Want to see example data or how it works?"):
    st.markdown("""
    You can explore how messages are processed and what features are extracted before classification.  
    This area can later be expanded with real-time EDA visualizations.
    """)
    if Path("data/SMSSpamCollection").exists():
        df = pd.read_csv("data/SMSSpamCollection", sep="\t", header=None, names=["label", "sms_message"])
        st.dataframe(df.sample(5))
    else:
        st.warning("Sample data not found. Please upload or download `SMSSpamCollection` in `data/` folder.")
