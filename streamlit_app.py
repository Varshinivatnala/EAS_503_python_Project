import streamlit as st
import joblib
import numpy as np
from sklearn.preprocessing import FunctionTransformer, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Path to the saved model
MODEL_PATH = "./RandomForestClassifier_final_model.joblib"

# Define the preprocessing steps
numeric_features = ['Engine_Size', 'Cylinders', 'Fuel_Consumption_Comb']
categorical_features = ['Vehicle_Class', 'Fuel_Type', 'Transmission']

# Define preprocessing for numerical features
numeric_transformer = Pipeline(steps=[
    ('log', FunctionTransformer(np.log1p, validate=True)),
    ('scaler', StandardScaler())
])

# Define preprocessing for categorical features
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine preprocessing for numeric and categorical features
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

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
vehicle_class_map = {"Compact": "Compact", "SUV": "SUV", "Sedan": "Sedan", "Truck": "Truck", "Van": "Van", "Crossover": "Crossover", "Other": "Other"}
fuel_type_map = {"Gasoline": "Gasoline", "Diesel": "Diesel", "Electricity": "Electricity", "Hybrid": "Hybrid", "Other": "Other"}
transmission_map = {"Automatic": "Automatic", "Manual": "Manual", "CVT": "CVT", "Other": "Other"}

# Predict button
if st.button("Predict"):
    if model:
        # Prepare raw input as a DataFrame-like structure for preprocessing
        raw_features = np.array([
            [
                engine_size,
                cylinders,
                fuel_consumption,
                vehicle_class_map[vehicle_class],
                fuel_type_map[fuel_type],
                transmission_map[transmission]
            ]
        ], dtype=object)

        try:
            # Apply preprocessing
            preprocessed_features = preprocessor.transform(raw_features)

            # Get the prediction
            prediction = model.predict(preprocessed_features)
            st.success(f"Predicted CO2 Emissions Category: {prediction[0]}")
        except Exception as e:
            st.error(f"Failed to make a prediction: {e}")
    else:
        st.error("Model not loaded. Please check the model path.")

# Footer
st.write("Developed by Varshini")
