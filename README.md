# 🛡️ LoanGuard AI

An AI-powered loan default prediction system that combines Machine Learning with Generative AI to predict financial risk and explain it in plain English.

## 🎯 Problem Statement
Banks lose crores every year due to loan defaults. Traditional systems only predict risk — they don't explain why. LoanGuard AI solves both problems.

## 🚀 Live Demo
Run locally using Streamlit — see setup instructions below.

## 🧠 How It Works
1. User enters their financial details
2. XGBoost model predicts default risk (0-100%)
3. Feature importance shows top risk factors
4. LLaMA 3 (via Groq) explains the result in plain English

## 📊 Model Performance
| Metric | Score |
|--------|-------|
| Accuracy | 94.02% |
| AUC-ROC | 0.952 |
| F1 Score | 0.85 |

## 🛠️ Tech Stack
- **ML Model:** XGBoost
- **Explainability:** Feature Importance
- **LLM:** LLaMA 3.1 via Groq API
- **Frontend:** Streamlit
- **Data Processing:** Pandas, NumPy, Scikit-learn

## 📁 Project Structure
```
loanguard-ai/
├── app/
│   └── app.py
├── notebooks/
│   ├── data_cleaning.py
│   ├── data_encoding.py
│   ├── model.py
│   └── feature_importance.py
└── .gitignore
```

## ⚙️ Setup Instructions

1. Clone the repository
```bash
git clone https://github.com/jainshubh803-a11y/financial-health-advisor.git
cd financial-health-advisor
```

2. Install dependencies
```bash
pip install pandas numpy scikit-learn xgboost streamlit groq python-dotenv
```

3. Add your Groq API key — create `.env` file in `app/` folder:
```
GROQ_API_KEY=your-groq-api-key-here
```

4. Run the app
```bash
python -m streamlit run app/app.py
```

## 📈 Key Features
- **94% accurate** XGBoost model trained on 31,000+ real loan records
- **Feature Engineering** — 3 custom features created (loan_to_income, interest_burden, age_income_ratio)
- **Explainable AI** — LLaMA 3 converts ML output to plain English advice
- **Live Web App** — Interactive Streamlit interface

## 🔍 Dataset
[Credit Risk Dataset](https://www.kaggle.com/datasets/laotse/credit-risk-dataset) from Kaggle — 32,000+ real loan records.

## 👨‍💻 Author
Shubh Jain — BTech Student