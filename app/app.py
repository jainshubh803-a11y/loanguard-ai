#python -m streamlit run app/app.py
import streamlit as st
import pandas as pd
from xgboost import XGBClassifier
from groq import Groq
groq_client = Groq(api_key=st.secrets["GROQ_API_KEY"])

st.set_page_config(page_title="LoanGuard AI", page_icon="🛡️", layout="centered")
st.title("🛡️ LoanGuard AI")
st.subheader("Check your loan default risk instantly")

# Load trained model
@st.cache_resource
def load_model():
    xgb = XGBClassifier()
    xgb.load_model('models/xgboost_model.json')
    return xgb

xgb_model = load_model()

st.markdown("### Enter your details")

left, right = st.columns(2)

with left:
    age = st.number_input("Age", min_value=18, max_value=80, value=25)
    annual_income = st.number_input("Annual Income (₹)", min_value=10000, max_value=2000000, value=50000)
    emp_years = st.number_input("Employment Length (years)", min_value=0, max_value=41, value=3)
    loan_amount = st.number_input("Loan Amount (₹)", min_value=500, max_value=350000, value=10000)
    interest_rate = st.number_input("Interest Rate (%)", min_value=5.0, max_value=20.0, value=10.0)
    loan_pct_income = st.number_input("Loan % of Income", min_value=0.0, max_value=0.83, value=0.2)

with right:
    home_own = st.selectbox("Home Ownership", ["RENT", "OWN", "MORTGAGE", "OTHER"])
    loan_purpose = st.selectbox("Loan Purpose", ["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "HOME IMPROVEMENT", "DEBT CONSOLIDATION"])
    grade = st.selectbox("Loan Grade", ["A", "B", "C", "D", "E", "F", "G"])
    prev_default = st.selectbox("Previous Default?", ["N", "Y"])
    credit_years = st.number_input("Credit History Length (years)", min_value=2, max_value=30, value=5)

# Encoding mappings
home_enc = {"RENT": 3, "OWN": 2, "MORTGAGE": 0, "OTHER": 1}
purpose_enc = {"PERSONAL": 4, "EDUCATION": 1, "MEDICAL": 3, "VENTURE": 5, "HOME IMPROVEMENT": 2, "DEBT CONSOLIDATION": 0}
grade_enc = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5, "G": 6}
default_enc = {"N": 0, "Y": 1}

# Feature engineering
loan_to_income = loan_amount / annual_income
interest_burden = (interest_rate / 100) * loan_amount / annual_income
age_income_ratio = age / (annual_income / 10000)

# Build input
user_data = pd.DataFrame({
    'person_age': [age],
    'person_income': [annual_income],
    'person_home_ownership': [home_enc[home_own]],
    'person_emp_length': [emp_years],
    'loan_intent': [purpose_enc[loan_purpose]],
    'loan_grade': [grade_enc[grade]],
    'loan_amnt': [loan_amount],
    'loan_int_rate': [interest_rate],
    'loan_percent_income': [loan_pct_income],
    'cb_person_default_on_file': [default_enc[prev_default]],
    'cb_person_cred_hist_length': [credit_years],
    'loan_to_income': [loan_to_income],
    'interest_burden': [interest_burden],
    'age_income_ratio': [age_income_ratio]
})

if st.button("Check My Risk 🔍"):
    prob = xgb_model.predict_proba(user_data)[0][1]
    risk_pct = float(f"{prob * 100:.2f}")

    st.markdown("---")
    st.markdown("### Result")

    if risk_pct >= 50:
        st.error(f"⚠️ High Risk — {risk_pct:.2f}% chance of default")
    else:
        st.success(f"✅ Low Risk — {risk_pct:.2f}% chance of default")

    st.progress(int(risk_pct))

    st.markdown("### Top factors affecting your risk")
    feat_imp = pd.DataFrame({
        'Factor': user_data.columns,
        'Importance': xgb_model.feature_importances_
    }).sort_values('Importance', ascending=False).head(5)

    st.bar_chart(feat_imp.set_index('Factor')['Importance'])

    st.markdown("### AI Explanation")
    with st.spinner("Analyzing your profile..."):
        prompt = f"""
        You are a friendly financial advisor. A person has the following profile:
        - Age: {age}
        - Annual Income: ₹{annual_income}
        - Employment Length: {emp_years} years
        - Home Ownership: {home_own}
        - Loan Amount: ₹{loan_amount}
        - Loan Purpose: {loan_purpose}
        - Loan Grade: {grade}
        - Interest Rate: {interest_rate}%
        - Previous Default: {prev_default}
        - Credit History: {credit_years} years
        - Loan to Income Ratio: {loan_to_income:.2f}

        Our ML model predicted their loan default risk as {risk_pct:.2f}%.

        In 3-4 simple sentences explain:
        1. Why this risk score makes sense
        2. Which 2-3 factors are most responsible
        3. One practical tip to improve their financial health

        Be friendly, simple, and helpful. No technical jargon.
        """

        reply = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}]
        )
        st.info(reply.choices[0].message.content)