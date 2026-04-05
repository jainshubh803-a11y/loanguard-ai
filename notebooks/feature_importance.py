import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
model = XGBClassifier()
model.load_model(r'F:\financial-health-advisor\models\xgboost_model.json')
df = pd.read_csv(r'F:\financial-health-advisor\data\credit_risk_outliers.csv')
x = df.drop('loan_status', axis=1)
importance = pd.DataFrame({'Feature': x.columns,'Importance': model.feature_importances_}).sort_values('Importance', ascending=False)
print(importance)
#Chart
plt.figure(figsize=(10, 6))
plt.barh(importance['Feature'], importance['Importance'], color='steelblue')
plt.xlabel('Importance')
plt.title('Feature Importance')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig(r'F:\financial-health-advisor\data\feature_importance.png', dpi=150)