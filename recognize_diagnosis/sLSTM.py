import matplotlib.pyplot as plt
from keras._tf_keras.keras.models import load_model
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.layers import Dense, LSTM, Dropout
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, mean_squared_error


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

X_train = X_train.reshape((X_train.shape[0], 1, 27))  
X_test = X_test.reshape((X_test.shape[0], 1, 27))  

# ساخت مدل LSTM مجبور شدیم بخاطر از اور فیتینگ، از دو لایه Dropout استفاده کنیم
model = Sequential([
    LSTM(128, activation='relu', input_shape=(1, 27), return_sequences=True),
    Dropout(0.7),
    LSTM(64, activation='relu', return_sequences=True),
    LSTM(32, activation='relu', return_sequences=False),
    Dropout(0.7),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(len(np.unique(y)), activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=25, batch_size=2, validation_data=(X_test, y_test))
model.save('lstm_Adam_model.h5')

loss, accuracy = model.evaluate(X_test, y_test)

print(f"Test Accuracy: {accuracy:.4f}")

y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)

accuracy = accuracy_score(y_test, y_pred)
precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
_, _, f2, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted', beta=2)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print(f"Test Accuracy: {accuracy:.4f}")
print(f"Loss : {loss:.4f}")
print(f"F1-score: {f1:.4f}")
print(f"F2-score: {f2:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MSE: {mse:.4f}")

print("Number of features:", X_train.shape[2])
print("Expected input shape:", model.input_shape)
print(X_test.shape)

