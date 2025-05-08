import numpy as np
import tensorflow as tf
from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.layers import Dense, LSTM, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, mean_squared_error
import pandas as pd

data_path = 'AI_MedData.xlsx'  
df = pd.read_excel(data_path)

df_cleaned = df.drop(columns=['Patient Name'], errors='ignore')

label_encoders = {}
for column in df_cleaned.columns:
    if df_cleaned[column].dtype == 'object':
        le = LabelEncoder()
        df_cleaned[column] = le.fit_transform(df_cleaned[column].astype(str))
        label_encoders[column] = le

X = df_cleaned.drop(columns=['Diagnosis']).values
y = df_cleaned['Diagnosis'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)  
X_test = scaler.transform(X_test)  

model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),  
    Dropout(0.3),  
    Dense(64, activation='relu'),
    Dropout(0.3),  
    Dense(32, activation='relu'),
    Dense(len(np.unique(y)), activation='softmax')  # خروجی چندکلاسه
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=25, batch_size=16, validation_data=(X_test, y_test), verbose=1)

y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)
loss, accuracy = model.evaluate(X_test, y_test)
accuracy = accuracy_score(y_test, y_pred)
precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
_, _, f2, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted', beta=2)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print(f"Test Accuracy: {accuracy:.4f}")
print(f"Loss: {loss:.4f}")
print(f"F1-score: {f1:.4f}")
print(f"F2-score: {f2:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MSE: {mse:.4f}")
