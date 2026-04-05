import streamlit as st
import pandas as pd
from xgboost import XGBClassifier
from groq import Groq
import plotly.graph_objects as go

st.set_page_config(page_title="LoanGuard AI", page_icon="🛡️", layout="wide")

# Custom CSS
st.markdown("""
<style>
    .main { background-color: #0f1117; }
    .stApp { background: linear-gradient(135deg, #0f1117 0%, #1a1f2e 100%); }
    .metric-card {
        background: #1e2130;
        border: 1px solid #2d3250;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }
    .high-risk {
        background: linear-gradient(135deg, #ff4b4b22, #ff4b4b11);
        border: 1px solid #ff4b4b;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }
    .low-risk {
        background: linear-gradient(135deg, #00c85322, #00c85311);
        border: 1px solid #00c853;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }
    .title-text {
        font-size: 48px;
        font-weight: 800;
        background: linear-gradient(90deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
</style>
""", unsafe_allow_html=True)

# Header
col_title, col_info = st.columns([3, 1])
with col_title:
    st.markdown('<p class="title-text">🛡️ LoanGuard AI</p>', unsafe_allow_html=True)
    st.markdown("##### AI-powered loan default risk prediction with explainable insights")

with col_info:
    st.markdown("""
    <div class="metric-card">
        <h3 style="color:#667eea">94.02%</h3>
        <p style="color:gray">Model Accuracy</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# Groq client
groq_client = Groq(api_key=st.secrets["GROQ_API_KEY"])

# Model load
@st.cache_resource
def load_model():
    xgb = XGBClassifier()
    xgb.load_model('models/xgboost_model.json')
    return xgb

xgb_model = load_model()

# Sidebar
with st.sidebar:
    st.markdown("### 📋 About LoanGuard AI")
    st.markdown("""
    LoanGuard AI uses **XGBoost** trained on 31,000+ real loan records to predict default risk.
    
    **How it works:**
    - Enter your financial details
    - ML model predicts risk score
    - AI explains the result
    
    **Model Performance:**
    - Accuracy: 94.02%
    - AUC-ROC: 0.952
    - F1 Score: 0.85
    """)
    st.markdown("---")
    st.markdown("Built with XGBoost + LLaMA 3")

# Input section
st.markdown("### 📝 Enter Your Financial Details")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Personal Info**")
    age = st.number_input("Age", min_value=18, max_value=80, value=25)
    annual_income = st.number_input("Annual Income (₹)", min_value=10000, max_value=2000000, value=50000)
    emp_years = st.number_input("Employment Length (years)", min_value=0, max_value=41, value=3)
    home_own = st.selectbox("Home Ownership", ["RENT", "OWN", "MORTGAGE", "OTHER"])

with col2:
    st.markdown("**Loan Details**")
    loan_amount = st.number_input("Loan Amount (₹)", min_value=500, max_value=1100000, value=10000)
    interest_rate = st.number_input("Interest Rate (%)", min_value=5.0, max_value=20.0, value=10.0)
    loan_pct_income = st.number_input("Loan % of Income", min_value=0.0, max_value=0.83, value=0.2)
    loan_purpose = st.selectbox("Loan Purpose", ["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "HOME IMPROVEMENT", "DEBT CONSOLIDATION"])

with col3:
    st.markdown("**Credit History**")
    grade = st.selectbox("Loan Grade", ["A", "B", "C", "D", "E", "F", "G"])
    prev_default = st.selectbox("Previous Default?", ["N", "Y"])
    credit_years = st.number_input("Credit History (years)", min_value=2, max_value=30, value=5)

st.markdown("---")

# Encoding
home_enc = {"RENT": 3, "OWN": 2, "MORTGAGE": 0, "OTHER": 1}
purpose_enc = {"PERSONAL": 4, "EDUCATION": 1, "MEDICAL": 3, "VENTURE": 5, "HOME IMPROVEMENT": 2, "DEBT CONSOLIDATION": 0}
grade_enc = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5, "G": 6}
default_enc = {"N": 0, "Y": 1}

loan_to_income = loan_amount / annual_income
interest_burden = (interest_rate / 100) * loan_amount / annual_income
age_income_ratio = age / (annual_income / 10000)

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

# Predict button
col_btn, col_empty = st.columns([1, 3])
with col_btn:
    predict_btn = st.button("🔍 Analyze My Risk", use_container_width=True)

if predict_btn:
    prob = xgb_model.predict_proba(user_data)[0][1]
    risk_pct = float(f"{prob * 100:.2f}")

    st.markdown("---")
    st.markdown("## 📊 Your Risk Analysis")

    col_gauge, col_result, col_metrics = st.columns(3)

    with col_gauge:
        # Gauge chart
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=risk_pct,
            title={'text': "Risk Score", 'font': {'color': 'white'}},
            number={'suffix': "%", 'font': {'color': 'white', 'size': 36}},
            gauge={
                'axis': {'range': [0, 100], 'tickcolor': "white"},
                'bar': {'color': "#ff4b4b" if risk_pct >= 50 else "#00c853"},
                'bgcolor': "#1e2130",
                
                'steps': [
    {'range': [0, 30], 'color': 'rgba(0, 200, 83, 0.2)'},
    {'range': [30, 60], 'color': 'rgba(255, 165, 0, 0.2)'},
    {'range': [60, 100], 'color': 'rgba(255, 75, 75, 0.2)'}
],
                'threshold': {
                    'line': {'color': "white", 'width': 2},
                    'thickness': 0.75,
                    'value': risk_pct
                }
            }
        ))
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font={'color': 'white'},
            height=280
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_result:
        if risk_pct >= 50:
            st.markdown(f"""
            <div class="high-risk">
                <h1 style="color:#ff4b4b">⚠️</h1>
                <h2 style="color:#ff4b4b">High Risk</h2>
                <h3 style="color:white">{risk_pct:.2f}%</h3>
                <p style="color:gray">chance of default</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="low-risk">
                <h1 style="color:#00c853">✅</h1>
                <h2 style="color:#00c853">Low Risk</h2>
                <h3 style="color:white">{risk_pct:.2f}%</h3>
                <p style="color:gray">chance of default</p>
            </div>
            """, unsafe_allow_html=True)

    with col_metrics:
        st.markdown(f"""
        <div class="metric-card">
            <p style="color:gray">Loan to Income</p>
            <h3 style="color:#667eea">{loan_to_income:.2f}</h3>
            <p style="color:gray">Interest Burden</p>
            <h3 style="color:#667eea">{interest_burden:.4f}</h3>
            <p style="color:gray">Loan Grade</p>
            <h3 style="color:#667eea">{grade}</h3>
        </div>
        """, unsafe_allow_html=True)

    # Feature importance
    st.markdown("### 🔑 Top Risk Factors")
    feat_imp = pd.DataFrame({
        'Factor': user_data.columns,
        'Importance': xgb_model.feature_importances_
    }).sort_values('Importance', ascending=True).tail(5)

    fig2 = go.Figure(go.Bar(
        x=feat_imp['Importance'],
        y=feat_imp['Factor'],
        orientation='h',
        marker_color='#667eea'
    ))
    fig2.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': 'white'},
        xaxis={'gridcolor': '#2d3250'},
        yaxis={'gridcolor': '#2d3250'},
        height=300
    )
    st.plotly_chart(fig2, use_container_width=True)

    # AI Explanation
    st.markdown("### 🤖 AI Explanation")
    with st.spinner("Analyzing your profile..."):
        prompt = f"""
You are a friendly credit analyst who explains loan decisions in simple terms that anyone can understand.

Loan Application Details:
- Applicant Age: {age} years
- Annual Income: ₹{annual_income}
- Employment: {emp_years} years
- Home Ownership: {home_own}
- Loan Requested: ₹{loan_amount} for {loan_purpose}
- Loan Grade: {grade} | Interest Rate: {interest_rate}%
- Previous Default: {prev_default}
- Credit History: {credit_years} years
- Loan to Income Ratio: {loan_to_income:.2f}

ML Model predicted default risk: {risk_pct:.2f}%

Provide analysis in this format:

1. VERDICT: Should this loan be approved? One clear sentence.

2. WHY THIS SCORE: Explain in simple words why the risk is high or low. Mention 2 specific numbers from their profile.

3. WHAT'S WORKING: Mention 1-2 positive things about their application.

4. HOW TO IMPROVE: Give 2 specific actionable steps to reduce their risk score. Be practical — mention exact numbers like "reduce loan amount to ₹X" or "build 6 months emergency fund".

5. FINAL ADVICE: One warm, encouraging sentence to end.

Write like a helpful friend who works at a bank.
Simple English, no jargon, under 150 words.
"""

        reply = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}]
        )
        st.info(reply.choices[0].message.content)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:red; font-size:12px">
    Built by Shubh Jain | XGBoost + LLaMA 3.1 | 
    <a href="https://github.com/jainshubh803-a11y/loanguard-ai" style="color:#667eea">GitHub</a>
</div>
""", unsafe_allow_html=True)