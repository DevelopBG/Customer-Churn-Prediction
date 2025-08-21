import streamlit as st
import pickle
import numpy as np

# Load the scaler and model
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("bestmodel.pkl", "rb") as f1:
    model = pickle.load(f1)

st.title("Churn Prediction App")
st.divider()

st.write("Please enter the values and click the 'Predict' button to get the prediction.")

st.divider()

# User inputs
age = st.number_input("Enter age", min_value=10, max_value=100, value=30)
tenure = st.number_input("Enter tenure (months)", min_value=0, max_value=130, value=30)
monthly_charge = st.number_input("Enter monthly charge", min_value=0, max_value=150, value=50)
gender = st.selectbox("Select gender", ["Male", "Female"])

st.divider()

if st.button("Predict"):
    # Convert gender to numeric
    gender_num = 1 if gender == "Female" else 0

    # Combine features and scale
    features = np.array([[age, gender_num, tenure, monthly_charge]])
    scaled_features = scaler.transform(features)

    # Make prediction
    prediction = model.predict(scaled_features)[0]
    result = "Churn" if prediction == 1 else "Not Churn"

    st.success(f"Prediction: {result}")
else:
    st.info("Fill in the details and click 'Predict' to see the result.")
