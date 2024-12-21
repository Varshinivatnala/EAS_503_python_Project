import streamlit as st
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer, StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.pipeline import Pipeline

# Path to the saved model
MODEL_PATH = "RandomForestClassifier_final_model.joblib"

@st.cache_resource
def load_model():
    try:
        model = joblib.load(MODEL_PATH)
        st.success("Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None

# Define preprocessing steps (as used during training)
def create_preprocessor():
    numeric_features = ['Engine Size', 'Cylinders', 'Fuel Consumption']
    categorical_features = ['Vehicle Class', 'Fuel Type', 'Transmission']
    
    numeric_transformer = Pipeline(steps=[
        ('log', FunctionTransformer(np.log1p, validate=True)),
        ('scaler', StandardScaler()),
        ('minmax', MinMaxScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )
    return preprocessor

# Load the model and preprocessor
model = load_model()
preprocessor = create_preprocessor()

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

# Map categorical inputs to expected column names
vehicle_class_map = {"Compact": "Compact", "SUV": "SUV", "Sedan": "Sedan", "Truck": "Truck", "Van": "Van", "Crossover": "Crossover", "Other": "Other"}
fuel_type_map = {"Gasoline": "Gasoline", "Diesel": "Diesel", "Electricity": "Electricity", "Hybrid": "Hybrid", "Other": "Other"}
transmission_map = {"Automatic": "Automatic", "Manual": "Manual", "CVT": "CVT", "Other": "Other"}

# Predict button
if st.button("Predict"):
    if model:
        # Prepare input features as a DataFrame
        input_data = pd.DataFrame({
            'Engine Size': [engine_size],
            'Cylinders': [cylinders],
            'Fuel Consumption': [fuel_consumption],
            'Vehicle Class': [vehicle_class_map[vehicle_class]],
            'Fuel Type': [fuel_type_map[fuel_type]],
            'Transmission': [transmission_map[transmission]],
        })

        try:
            # Apply preprocessing
            features_preprocessed = preprocessor.fit_transform(input_data)
            
            # Get prediction
            prediction = model.predict(features_preprocessed)
            st.success(f"Predicted CO2 Emissions Category: {prediction[0]}")
        except Exception as e:
            st.error(f"Failed to make a prediction: {e}")
    else:
        st.error("Model not loaded. Please check the model path.")

# Footer
st.write("Developed by Varshini")
