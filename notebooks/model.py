import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
df = pd.read_csv(r'F:\financial-health-advisor\data\credit_risk_outliers.csv')
x=df.drop(['loan_status'],axis=1)
y=df['loan_status']
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42,stratify=y)
print("Train size:", x_train.shape)
print("Test size:", x_test.shape)
model=XGBClassifier(n_estimators=500,max_depth=5,learning_rate=0.1,random_state=42,eval_metric='logloss',scale_pos_weight=2)
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
y_prob = model.predict_proba(x_test)[:,1]
print("Accuracy:", model.score(x_test,y_test)*100)
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)
print(classification_report(y_test, y_pred))
print("AUC-ROC Score:", round(roc_auc_score(y_test, y_prob),3))
model.save_model(r'F:\financial-health-advisor\models\xgboost_model.json')
model.load_model(r'F:\financial-health-advisor\models\xgboost_model.json')
