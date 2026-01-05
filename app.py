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
# 4. Prediction Logic
if st.button("Predict Price Tier"):
    model = load_mobi_ai()
 
    data_input = [0.0] * 20 

    data_input[0] = battery    # 'battery_power' is index 0
    data_input[2] = clock      # 'clock_speed' is index 2
    data_input[4] = fc         # 'fc' is index 4
    data_input[6] = int_mem    # 'int_memory' is index 6
    data_input[13] = ram       # 'ram' is index 13

    
    data_input[8] = 150        # 'mobile_wt' (Average weight)
    data_input[9] = 4          # 'n_cores' (Average 4 cores)
    data_input[11] = 1000      # 'px_height' (Average screen height)
    data_input[12] = 1000      # 'px_width' (Average screen width)


    input_data = np.array([data_input])
    scaled_input = scaler.transform(input_data)

    predict = model.predict( scaled_input)[0][0]

    if predict > 0.5:
        st.success(f"HIGH PRICE TIER")
    else:
        st.info(f"BUDGET PRICE TIER")

    
