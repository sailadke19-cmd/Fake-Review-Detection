import streamlit as st
import torch

# Import the prediction function and necessary components from model.py
from model import clean_text, encode_text, model, predict_email_sentiment

# --- Streamlit Application ---
st.title("🧠 Fake Review Detection (BERT AI Model)")

review_input = st.text_area("Enter Review")

if st.button("Check"):
    if review_input:
        # Use the predict_email_sentiment function from model.py
        result = predict_email_sentiment(review_input)
        st.success(result)
    else:
        st.warning("Please enter a review to check.")
