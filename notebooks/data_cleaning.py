import pandas as pd
import numpy as np
df = pd.read_csv(r'F:\financial-health-advisor\data\credit_risk_dataset.csv')
print(df['loan_status'].value_counts(normalize=True).round(3))
df=df[df['person_age']<90]
print("Age range:", df['person_age'].min(), "to", df['person_age'].max())
df=df[df['person_emp_length']<60]
print("Employment length range:", df['person_emp_length'].min(), "to", df['person_emp_length'].max())
df['person_emp_length']=df['person_emp_length'].fillna(df['person_emp_length'].median())
df['loan_int_rate']=df['loan_int_rate'].fillna(df['loan_int_rate'].median())
print("Shape after cleaning:", df.shape)
print("\nMissing values remaining:")
print(df.isnull().sum())
df.to_csv(r'F:\financial-health-advisor\data\credit_risk_cleaned.csv', index=False)