import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
df = pd.read_csv(r'F:\financial-health-advisor\data\credit_risk_cleaned.csv')
# new features
df['loan_to_income'] = df['loan_amnt'] / df['person_income']
df['interest_burden'] = (df['loan_int_rate'] / 100) * df['loan_amnt'] / df['person_income']
df['age_income_ratio'] = df['person_age'] / (df['person_income'] / 10000)
print(df[['loan_to_income', 'interest_burden', 'age_income_ratio']].describe().round(3))


categorical_cols = df.select_dtypes(include=['object']).columns
print("Categorical columns:", categorical_cols)

la = LabelEncoder()
for col in categorical_cols:
    df[col] = la.fit_transform(df[col])
    print(f"{col} encoded successfully")

x=df.drop(['loan_status'], axis=1)
y=df['loan_status']

print("\nFinal columns:")
print(x.columns.tolist())
df.to_csv(r'F:\financial-health-advisor\data\credit_risk_features.csv', index=False)