import streamlit as st
import pickle
import numpy as np

# Title
st.title("🏏 Cricket Win Predictor")

# Load model
try:
    model = pickle.load(open('model (2).pkl', 'rb'))
except:
    st.error("Model file not found or not loading!")
    st.stop()

# Inputs
runs_left = st.number_input("Runs Left", min_value=0)
balls_left = st.number_input("Balls Left", min_value=0)
wickets_left = st.number_input("Wickets Left", min_value=0, max_value=10)
crr = st.number_input("Current Run Rate", min_value=0.0)

# Button
if st.button("Predict"):
    input_data = np.array([[runs_left, balls_left, wickets_left, crr]])
    
    try:
        prob = model.predict_proba(input_data)[0][1] * 100
        st.success(f"Win Probability: {prob:.2f}%")
        st.progress(int(prob))
    except Exception as e:
        st.error(f"Error: {e}")