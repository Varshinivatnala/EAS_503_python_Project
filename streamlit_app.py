
import streamlit as st
import joblib
import numpy as np

# Path to the saved model
MODEL_PATH = ""

# Function to load the model
@st.cache_resource
def load_model():
    try:
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

# Map categorical inputs to numerical representations
vehicle_class_map = {"Compact": 0, "SUV": 1, "Sedan": 2, "Truck": 3, "Van": 4, "Crossover": 5, "Other": 6}
fuel_type_map = {"Gasoline": 0, "Diesel": 1, "Electricity": 2, "Hybrid": 3, "Other": 4}
transmission_map = {"Automatic": 0, "Manual": 1, "CVT": 2, "Other": 3}

# Predict button
if st.button("Predict"):
    if model:
        # Prepare features for the model
        features = np.array([
            [
                engine_size,
                cylinders,
                fuel_consumption,
                vehicle_class_map[vehicle_class],
                fuel_type_map[fuel_type],
                transmission_map[transmission],
            ]
        ])

        try:
            # Get the prediction
            prediction = model.predict(features)
            st.success(f"Predicted CO2 Emissions Category: {prediction[0]}")
        except Exception as e:
            st.error(f"Failed to make a prediction: {e}")
    else:
        st.error("Model not loaded. Please check the model path.")

# Footer
st.write("Developed by Varshini")
