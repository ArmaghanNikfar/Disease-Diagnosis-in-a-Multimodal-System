import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, mean_squared_error
import matplotlib.pyplot as plt

data_path = 'Patient_Data_Diagnosis.xlsx'
df = pd.read_excel(data_path)
df_cleaned = df.drop(columns=['Patient Name'])

label_encoders = {}
for column in df_cleaned.columns:
    if df_cleaned[column].dtype == 'object':
        le = LabelEncoder()
        df_cleaned[column] = le.fit_transform(df_cleaned[column].astype(str))
        label_encoders[column] = le

X = df_cleaned.drop(columns=['Diagnosis']).values
y = df_cleaned['Diagnosis'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# نرمال‌سازی داده‌ها
X_train = (X_train - np.min(X_train, axis=0)) / (np.max(X_train, axis=0) - np.min(X_train, axis=0) + 1e-6)
X_test = (X_test - np.min(X_test, axis=0)) / (np.max(X_test, axis=0) - np.min(X_test, axis=0) + 1e-6)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

y_pred = rf_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
_, _, f2, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted', beta=2)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print(f"Test Accuracy: {accuracy:.4f}")
print(f"F1-score: {f1:.4f}")
print(f"F2-score: {f2:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MSE: {mse:.4f}")


importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1]

print("Feature ranking:")
for i in range(X_train.shape[1]):
    print(f"{i+1}. Feature {indices[i]} - Importance: {importances[indices[i]]:.4f}")

plt.figure(figsize=(10, 6))
plt.title("Feature Importances")
plt.bar(range(X_train.shape[1]), importances[indices], align="center")
plt.xticks(range(X_train.shape[1]), indices)
plt.xlim([-1, X_train.shape[1]])
plt.show()
