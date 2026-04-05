import pandas as pd
df = pd.read_csv(r'F:\financial-health-advisor\data\credit_risk_features.csv')
numeric_cols = ['person_income','person_emp_length','loan_amnt','loan_int_rate','loan_percent_income','cb_person_cred_hist_length']
for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3-Q1
    lower = Q1-1.5*IQR
    upper = Q3+1.5*IQR
    outliers = df[(df[col]<lower) | (df[col]>upper)].shape[0]
    print(f"{col}:")
    print(f" Outliers  : {outliers} rows")
    print()

#removing outliers
Q1_int = df['loan_int_rate'].quantile(0.25)
Q3_int = df['loan_int_rate'].quantile(0.75)
IQR_int = Q3_int - Q1_int
upper_int = Q3_int + 1.5 * IQR_int
print(df.shape)
df = df[df['loan_int_rate'] <= upper_int]
print( df.shape)
df.drop_duplicates(inplace=True)
print( df.shape)
df.to_csv(r'F:\financial-health-advisor\data\credit_risk_outliers.csv', index=False)
