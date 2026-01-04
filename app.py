import streamlit as st
import tensorflow as tf
import joblib
import numpy as np

# 1. Load the Scaler and Model Structure
scaler = joblib.load('mobile_scaler.pkl')

def load_mobi_ai():
    # This architecture must be IDENTICAL to your training code
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(8, activation='relu', input_shape=(scaler.n_features_in_,)),
        tf.keras.layers.Dense(4, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.load_weights('mobile_model.weights.h5')
    return model

# 2. Front-End Design
st.set_page_config(page_title="Nimal's Mobile AI", page_icon="📱")
st.title("📱 Nimal's Mobile Price Estimator")
st.markdown("Enter your new phone's specifications to see the market tier.")

# 3. User Inputs (Adjust these to match your CSV columns!)
col1, col2 = st.columns(2)
with col1:
    ram = st.number_input("RAM (MB)", value=2048)
    battery = st.number_input("Battery (mAh)", value=4000)
    int_mem = st.number_input("Internal Memory (GB)", value=64)
with col2:
    clock = st.number_input("Clock Speed (GHz)", value=2.0)
    fc = st.number_input("Front Camera (MP)", value=8)

# 4. Prediction Logic
if st.button("Predict Price Tier"):
    model = load_mobi_ai()

    # IMPORTANT: The list below must be in the EXACT order of your CSV columns
    # Example order: [Battery, Clock, FC, Int_Mem, RAM]
    input_data = np.array([[battery, clock, fc, int_mem, ram]])

    # Scale and Predict
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0][0]

    if prediction > 0.5:
        st.success(f"💎 **Result: HIGH PRICE TIER** (Confidence: {prediction*100:.1f}%)")
    else:
        st.info(f"💰 **Result: BUDGET PRICE TIER** (Confidence: {(1-prediction)*100:.1f}%)")
