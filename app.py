import streamlit as st
import tensorflow as tf
import joblib
import numpy as np

scaler = joblib.load('mobile_scaler.pkl')

def load_mobi_ai():
    
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(8, activation='relu', input_shape=(scaler.n_features_in_,)),
        tf.keras.layers.Dense(4, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.load_weights('mobile_model.weights.h5')
    return model


st.set_page_config(page_title="Mobile AI", page_icon="🤖")
st.title("🤖  Mobile Price tier Estimator")
st.markdown("Enter your new phone's specifications to see the market tier.")

col1, col2 = st.columns(2)
with col1:
    ram = st.number_input("RAM (MB)", value=2048)
    battery = st.number_input("Battery (mAh)", value=4000)
    int_mem = st.number_input("Internal Memory (GB)", value=64)
with col2:
    clock = st.number_input("Clock Speed (GHz)", value=2.0)
    fc = st.number_input("Front Camera (MP)", value=8)

#prediction
if st.button("Predict Price Tier"):
    model = load_mobi_ai()
 
    data_input = [0.0] * 20 

    data_input[0] = battery    
    data_input[2] = clock     
    data_input[4] = fc        
    data_input[6] = int_mem    
    data_input[13] = ram       

    
    data_input[8] = 150        
    data_input[9] = 4          
    data_input[11] = 1000      
    data_input[12] = 1000      


    input_data = np.array([data_input])
    scaled_input = scaler.transform(input_data)

    predict = model.predict( scaled_input)[0][0]

    if predict > 0.5:
        st.success(f"HIGH PRICE TIER")
    else:
        st.info(f"BUDGET PRICE TIER")

    
