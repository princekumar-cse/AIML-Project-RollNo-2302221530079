import streamlit as st
import pandas as pd
import joblib

# Load the trained model and the exact column order it expects
model = joblib.load('churn_model.pkl')
model_columns = joblib.load('model_columns.pkl')

st.title("Customer Churn Prediction")
st.write("Fill in the customer's details below to predict whether they are likely to churn.")

# --- Collect raw inputs from the user, matching the original dataset columns ---
gender = st.selectbox("Gender", ["Male", "Female"])
SeniorCitizen = st.selectbox("Senior Citizen", [0, 1])
Partner = st.selectbox("Has Partner", ["Yes", "No"])
Dependents = st.selectbox("Has Dependents", ["Yes", "No"])
tenure = st.slider("Tenure (months)", 0, 72, 12)
PhoneService = st.selectbox("Phone Service", ["Yes", "No"])
MultipleLines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
OnlineSecurity = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
OnlineBackup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
DeviceProtection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
TechSupport = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
StreamingTV = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
StreamingMovies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
PaperlessBilling = st.selectbox("Paperless Billing", ["Yes", "No"])
PaymentMethod = st.selectbox("Payment Method", [
    "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
])
MonthlyCharges = st.number_input("Monthly Charges", min_value=0.0, max_value=200.0, value=70.0)
TotalCharges = st.number_input("Total Charges", min_value=0.0, max_value=10000.0, value=1000.0)

if st.button("Predict"):
    # Build a single-row dataframe exactly like the original raw data
    input_dict = {
        'gender': gender, 'SeniorCitizen': SeniorCitizen, 'Partner': Partner,
        'Dependents': Dependents, 'tenure': tenure, 'PhoneService': PhoneService,
        'MultipleLines': MultipleLines, 'InternetService': InternetService,
        'OnlineSecurity': OnlineSecurity, 'OnlineBackup': OnlineBackup,
        'DeviceProtection': DeviceProtection, 'TechSupport': TechSupport,
        'StreamingTV': StreamingTV, 'StreamingMovies': StreamingMovies,
        'Contract': Contract, 'PaperlessBilling': PaperlessBilling,
        'PaymentMethod': PaymentMethod, 'MonthlyCharges': MonthlyCharges,
        'TotalCharges': TotalCharges
    }
    input_df = pd.DataFrame([input_dict])

    # Recreate the same tenure_group feature used in training
    input_df['tenure_group'] = pd.cut(
        input_df['tenure'], bins=[0, 12, 24, 48, 100],
        labels=['0-12', '13-24', '25-48', '49+']
    )

    # One-hot encode the same way as training
    input_encoded = pd.get_dummies(input_df, drop_first=True)

    # Add any missing columns (the model expects a fixed set of columns)
    # and put them in the exact same order the model was trained on.
    input_encoded = input_encoded.reindex(columns=model_columns, fill_value=0)

    prediction = model.predict(input_encoded)[0]
    probability = model.predict_proba(input_encoded)[0][1]

    if prediction == 1:
        st.error(f"This customer is likely to CHURN. (Probability: {probability:.2%})")
    else:
        st.success(f"This customer is likely to STAY. (Probability of churn: {probability:.2%})")
