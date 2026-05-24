import streamlit as st
import requests

st.set_page_config(page_title="Travel AI", page_icon="✈️")

st.title("✈️ AI Travel Agent")
st.write("Checking system status...")

# This is the bridge! We ask the backend for data.
try:
    # FastAPI runs on port 8000 by default on your local machine
    response = requests.get("http://127.0.0.1:8000/")
    
    if response.status_code == 200:
        data = response.json()
        st.success(f"✅ Connection Successful: {data['status']}")
        st.info(f"Backend says: {data['message']}")
    else:
        st.warning("Backend found, but returned an unexpected response.")

except requests.exceptions.ConnectionError:
    st.error("🚨 Failed to connect. Is the FastAPI backend running?")