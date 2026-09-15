"""Minimal clinician-facing review dashboard starter."""
import os
import requests
import streamlit as st

st.title("Digital Liver World Model")
st.caption("Decision support prototype — predictions require clinician review.")
api_url = st.text_input("API URL", os.getenv("LIVER_API_URL", "http://localhost:8000"))
trajectory = st.text_area("Observation window (JSON list of 8-D states)", "[[0,0,0,0,0,0,0,0]]")
if st.button("Predict"):
    response = requests.post(f"{api_url.rstrip('/')}/predict", json={"observation_window": __import__('json').loads(trajectory)}, timeout=60)
    st.json(response.json())
