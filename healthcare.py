import streamlit as st
import pandas as pd
import joblib

model = joblib.load("healthcare_model.pkl")
encoders = joblib.load("encoders.pkl")
scaler = joblib.load("scaler.pkl")
selector = joblib.load("selector.pkl")
model_columns = joblib.load("model_columns.pkl")

st.title("Healthcare Test Result Prediction App")
st.write("This app predicts the patient's test result based on healthcare information.")

age = st.slider("Age", 0, 100, 30)

gender = st.selectbox("Gender", ["Male", "Female"])

blood_type = st.selectbox(
    "Blood Type",
    ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"]
)

medical_condition = st.selectbox(
    "Medical Condition",
    ["Cancer", "Obesity", "Diabetes", "Asthma", "Hypertension", "Arthritis"]
)

billing_amount = st.number_input(
    "Billing Amount",
    min_value=0.0,
    max_value=40000.0,
    value=10000.0
)

admission_type = st.selectbox(
    "Admission Type",
    ["Emergency", "Elective", "Urgent"]
)

medication = st.selectbox(
    "Medication",
    ["Aspirin", "Ibuprofen", "Lipitor", "Paracetamol", "Penicillin"]
)

stay_duration = st.slider("Stay Duration (days)", 1, 30, 5)

input_data = pd.DataFrame({
    "Age": [age],
    "Gender": [gender],
    "Blood_Type": [blood_type],
    "Medical_Condition": [medical_condition],
    "Billing_Amount": [billing_amount],
    "Admission_Type": [admission_type],
    "Medication": [medication],
    "Stay_Duration": [stay_duration]
})


for col in input_data.select_dtypes(include='object').columns:
    input_data[col] = encoders[col].transform(input_data[col])


input_data = input_data[model_columns]

input_scaled = scaler.transform(input_data)
input_selected = selector.transform(input_scaled)

if st.button("Predict Test Result"):
    prediction = model.predict(input_selected)
    result = encoders["Test_Results"].inverse_transform(prediction)
    st.success(f"Predicted Test Result: {result[0]}")
