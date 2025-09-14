import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
import joblib
import os

st.set_page_config(page_title="Health Insurance Claim Prediction", layout="wide")
st.title("🏥 Health Insurance Claim Prediction")
st.write("Predict the approximate health insurance claim amount based on individual characteristics.")

DATA_FILE = "health_insurance.csv"
MODEL_FILE = "model.pkl"

if not os.path.exists(DATA_FILE):
    st.error(f"Dataset {DATA_FILE} not found!")
else:
    df = pd.read_csv(DATA_FILE)
    st.subheader("Sample Data")
    st.dataframe(df.head())

if os.path.exists(MODEL_FILE):
    model = joblib.load(MODEL_FILE)
    st.success("✅ Model loaded successfully!")
else:
    st.warning("⚡ Training model... this may take a few seconds.")
    X = pd.get_dummies(df.drop("claim", axis=1), drop_first=True)
    y = df["claim"]
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    joblib.dump(model, MODEL_FILE)
    st.success("✅ Model trained and saved!")

X = pd.get_dummies(df.drop("claim", axis=1), drop_first=True)
if hasattr(model, "feature_names_in_"):
    for c in set(model.feature_names_in_) - set(X.columns):
        X[c] = 0
    X = X[model.feature_names_in_]

st.subheader("Predictions for Sample Data")
predictions = model.predict(X)
st.dataframe(pd.DataFrame(predictions[:10], columns=["Predicted Claim"]))

st.subheader("Predict Your Own Claim")
with st.form("claim_form"):
    age = st.number_input("Age", 0, 120, 30)
    sex = st.selectbox("Sex", ["male", "female"])
    weight = st.number_input("Weight (kg)", 0, 300, 70)
    bmi = st.number_input("BMI", 0.0, 60.0, 22.0)
    hereditary = st.selectbox("Hereditary Diseases", df["hereditary_diseases"].unique())
    dependents = st.number_input("Number of Dependents", 0, 10, 1)
    smoker = st.selectbox("Smoker", [0, 1])
    city = st.selectbox("City", df["city"].unique())
    bp = st.number_input("Blood Pressure", 0, 200, 75)
    diabetes = st.selectbox("Diabetes", [0, 1])
    exercise = st.selectbox("Regular Exercise", [0, 1])
    job = st.selectbox("Job Title", df["job_title"].unique())
    
    submit_button = st.form_submit_button("Predict Claim")

if submit_button:
    input_df = pd.DataFrame(
        [[age, sex, weight, bmi, hereditary, dependents, smoker, city, bp, diabetes, exercise, job]],
        columns=df.columns[:-1]
    )
    input_df = pd.get_dummies(input_df, drop_first=True)
    for c in set(model.feature_names_in_) - set(input_df.columns):
        input_df[c] = 0
    input_df = input_df[model.feature_names_in_]

    predicted_claim = model.predict(input_df)[0]
    st.success(f"💰 Predicted Health Insurance Claim: ${predicted_claim:,.2f}")
