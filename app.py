import streamlit as st
import pandas as pd
import joblib
import numpy as np

# PAGE CONFIGURATION
st.set_page_config(
    page_title="Bank Subscription Predictor",
    page_icon="💰",
    layout="wide"
)

# The custom feature engineering function must be defined in the script
def feature_engineer(data):
    """Creates interaction features. Must be identical to the training script version."""
    df_eng = data.copy()
    df_eng['balance_per_age'] = df_eng['balance'] / (df_eng['age'] + 1)
    df_eng['duration_x_campaign'] = df_eng['duration'] * df_eng['campaign']
    return df_eng

# MODEL LOADING
@st.cache_resource
def load_model():
    """Loads the single, final end-to-end pipeline object."""
    try:
        pipeline = joblib.load('final_model_pipeline.joblib')
        return pipeline
    except FileNotFoundError:
        st.error("Model file not found. Please ensure 'final_model_pipeline.joblib' is in the same directory.")
        return None

pipeline = load_model()

# OPTIMAL THRESHOLD
OPTIMAL_THRESHOLD = 0.52

# UI & LAYOUT
st.title("💰 Bank Term Deposit Subscription Predictor")
st.markdown("Enter the client's details to predict their subscription likelihood.")

if pipeline is None:
    st.warning("Model is not loaded. Cannot proceed.")
else:
    # Input Fields
    col1, col2 = st.columns(2)
    with col1:
        st.header("Client Demographics")
        age = st.slider("Age", 18, 100, 41)
        job = st.selectbox("Job", ['management', 'technician', 'blue-collar', 'admin.', 'services', 'retired', 'self-employed', 'student', 'unemployed', 'entrepreneur', 'housemaid', 'unknown'])
        marital = st.selectbox("Marital Status", ['married', 'single', 'divorced'])
        education = st.selectbox("Education Level", ['secondary', 'tertiary', 'primary', 'unknown'])

    with col2:
        st.header("Financial & Loan Info")
        default = st.radio("Has Credit in Default?", ('no', 'yes'), horizontal=True)
        balance = st.number_input("Average Yearly Balance (€)", -8019, 102127, 1500)
        housing = st.radio("Has Housing Loan?", ('yes', 'no'), horizontal=True, index=0)
        loan = st.radio("Has Personal Loan?", ('no', 'yes'), horizontal=True, index=1)
    
    duration = st.number_input("Last Contact Duration (seconds)", 0, 4918, 200)
    campaign = st.number_input("Contacts during this campaign", 1, 63, 1)
    pdays = st.number_input("Days since last contact (pdays)", -1, 871, -1)
    previous = st.number_input("Contacts before this campaign", 0, 275, 0)
    poutcome = st.selectbox("Outcome of Previous Campaign", ['unknown', 'failure', 'other', 'success'])
    contact = st.selectbox("Contact Communication Type", ['cellular', 'unknown', 'telephone'])
    day = st.slider("Last Contact Day of Month", 1, 31, 15)
    month = st.selectbox("Last Contact Month", ['may', 'jul', 'aug', 'jun', 'nov', 'apr', 'feb', 'jan', 'oct', 'sep', 'mar', 'dec'])

    if st.button("🚀 Predict Subscription", use_container_width=True, type="primary"):
        input_data = pd.DataFrame({
            'age': [age], 'job': [job], 'marital': [marital], 'education': [education],
            'default': [default], 'balance': [balance], 'housing': [housing], 'loan': [loan],
            'contact': [contact], 'day': [day], 'month': [month], 'duration': [duration],
            'campaign': [campaign], 'pdays': [pdays], 'previous': [previous], 'poutcome': [poutcome]
        })

        # Make a prediction.
        prediction_proba = pipeline.predict_proba(input_data)[0]
        probability_subscribe = prediction_proba[1]
        
        # Display the result
        st.subheader("Prediction Result")
        if probability_subscribe >= OPTIMAL_THRESHOLD:
            st.success(f"**YES**, the client is likely to subscribe.")
            st.metric(label="Subscription Probability", value=f"{probability_subscribe:.2%}")
            st.balloons()
        else:
            st.error(f"**NO**, the client is unlikely to subscribe.")
            st.metric(label="Subscription Probability", value=f"{probability_subscribe:.2%}")