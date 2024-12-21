import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Path to the saved model
MODEL_PATH = "RandomForestClassifier_final_model.joblib"

# Function to load the model
@st.cache_resource
def load_model():
    try:
        # Load the full pipeline, including preprocessing and the model
        model = joblib.load(MODEL_PATH)
        st.success("Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None

# Load the model
model = load_model()

# Page title
st.title("CO2 Emissions Classification App")

# Input fields
st.header("Input Vehicle Specifications")
engine_size = st.number_input("Engine Size (L)", min_value=0.0, step=0.1, format="%.1f")
cylinders = st.number_input("Number of Cylinders", min_value=1, step=1)
fuel_consumption = st.number_input("Fuel Consumption Comb (L/100 km)", min_value=0.0, step=0.1, format="%.1f")
vehicle_class = st.selectbox("Vehicle Class", ["Compact", "SUV", "Sedan", "Truck", "Van", "Crossover", "Other"])
fuel_type = st.selectbox("Fuel Type", ["Gasoline", "Diesel", "Electricity", "Hybrid", "Other"])
transmission = st.selectbox("Transmission Type", ["Automatic", "Manual", "CVT", "Other"])

# Predict button
if st.button("Predict"):
    if model:
        try:
            # Prepare input data as a pandas DataFrame
            input_data = pd.DataFrame({
                "Engine Size(L)": [engine_size],
                "Cylinders": [cylinders],
                "Fuel Consumption Comb (L/100 km)": [fuel_consumption],
                "Vehicle Class": [vehicle_class],
                "Fuel Type": [fuel_type],
                "Transmission": [transmission]
            })
            
            # Compute engineered features
            input_data["Engine_Cylinders_Ratio"] = input_data["Engine Size(L)"] / (input_data["Cylinders"] + 1)
            input_data["Fuel_Consumption_per_Cylinder"] = input_data["Fuel Consumption Comb (L/100 km)"] / (input_data["Cylinders"] + 1)
            
            # Make prediction using the model
            prediction = model.predict(input_data)
            st.success(f"Predicted CO2 Emissions Category: {prediction[0]}")
        except Exception as e:
            st.error(f"Failed to make a prediction: {e}")
    else:
        st.error("Model not loaded. Please check the model path.")

# Footer
st.write("Developed by Varshini")
