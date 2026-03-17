import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. SET UP THE PAGE ---
st.set_page_config(page_title="Dashboard", layout="wide")


# --- 2. CACHING DATA & MODEL ---
@st.cache_data
def load_data():
    df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df.dropna(inplace=True)
    return df


@st.cache_resource
def load_model():
    with open('churn_model.pkl', 'rb') as f:
        return pickle.load(f)


df = load_data()
model = load_model()

# --- 3. SIDEBAR NAVIGATION & TECH STACK ---
st.sidebar.title("Navigation")
st.sidebar.markdown("---")

page = st.sidebar.radio("Select Module:", ["Data & Analytics", "Inference Engine"])

st.sidebar.markdown("---")
st.sidebar.markdown("### Architecture Stack")
st.sidebar.code("""
• Algorithm: XGBoost
• Preprocessing: Pipeline
  - StandardScaler
  - OneHotEncoder (Sparse)
• Tuning: RandomizedSearchCV
• Backend: Python 3.12
• Frontend: Streamlit
""", language="markdown")


# ==========================================
# PAGE 1: DATA ANALYTICS
# ==========================================
if page == "Data & Analytics":
    st.title("Training Data Overview")
    st.markdown("Exploratory Data Analysis (EDA) of the historical dataset.")

    col1, col2, col3 = st.columns(3)
    col1.metric("Training Samples", f"{len(df):,}")
    col2.metric("Features Analyzed", "19")
    churn_rate = (df[df['Churn'] == 'Yes'].shape[0] / len(df)) * 100
    col3.metric("Global Churn Rate", f"{churn_rate:.1f}%")

    st.markdown("---")

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.subheader("Target Variable Distribution")
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        sns.countplot(data=df, x='Churn', palette=["#2ecc71", "#e74c3c"], ax=ax1)
        ax1.set_ylabel("Volume")
        st.pyplot(fig1)

    with chart_col2:
        st.subheader("Feature Multicollinearity (Pearson)")
        df_numeric = df.copy()
        df_numeric['Churn'] = df_numeric['Churn'].map({'Yes': 1, 'No': 0})
        num_cols = df_numeric.select_dtypes(include=['float64', 'int64'])

        fig2, ax2 = plt.subplots(figsize=(6, 4))
        sns.heatmap(num_cols.corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=ax2, cbar=False)
        st.pyplot(fig2)


# ==========================================
# PAGE 2: CHURN PREDICTION (INFERENCE)
# ==========================================
elif page == "Inference Engine":
    st.title("Live Inference Engine")
    st.markdown("Inject customer parameters into the predictive pipeline to calculate probability.")

    with st.container():
        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("Demographics")
            gender = st.selectbox("Gender", ["Male", "Female"])
            senior_citizen = st.selectbox("Senior Citizen", [0, 1])
            partner = st.selectbox("Partner", ["Yes", "No"])
            dependents = st.selectbox("Dependents", ["Yes", "No"])

        with col2:
            st.subheader("Active Services")
            internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
            phone_service = st.selectbox("Phone Service", ["Yes", "No"])
            multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
            online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
            online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
            device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
            tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
            streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
            streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])

        with col3:
            st.subheader("Financials & Contract")
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
            paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
            payment_method = st.selectbox("Payment Method",
                                          ["Electronic check", "Mailed check", "Bank transfer (automatic)",
                                           "Credit card (automatic)"])
            tenure = st.slider("Tenure (Months)", min_value=0, max_value=72, value=12)
            monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=150.0, value=50.0)
            total_charges = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=500.0)

    st.markdown("---")

    if st.button("Execute Model Inference", type="primary", use_container_width=True):

        input_dict = {
            'gender': gender, 'SeniorCitizen': senior_citizen, 'Partner': partner, 'Dependents': dependents,
            'tenure': tenure, 'PhoneService': phone_service, 'MultipleLines': multiple_lines,
            'InternetService': internet_service, 'OnlineSecurity': online_security,
            'OnlineBackup': online_backup, 'DeviceProtection': device_protection, 'TechSupport': tech_support,
            'StreamingTV': streaming_tv, 'StreamingMovies': streaming_movies, 'Contract': contract,
            'PaperlessBilling': paperless_billing, 'PaymentMethod': payment_method,
            'MonthlyCharges': monthly_charges, 'TotalCharges': total_charges
        }

        input_df = pd.DataFrame([input_dict])

prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]

        st.markdown("### Inference Result:")
        if prediction == 1:
            st.error(f"**HIGH RISK DETECTED** \n\n Predicted Churn Risk: {probability:.1%}")
            st.progress(float(probability))
        else:
            st.success(f"**RETENTION LIKELY** \n\n Predicted Churn Risk: {probability:.1%}")
            st.progress(float(probability))
