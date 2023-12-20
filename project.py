import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Încarcare date
file = 'REMS_Mars_Dataset.csv'
data = pd.read_csv(file)

# Asigurare că coloanele numerice au tipul corect
columns = ['sol_number', 'max_ground_temp(°C)', 'min_ground_temp(°C)', 'max_air_temp(°C)', 'min_air_temp(°C)', 'mean_pressure(Pa)', 'wind_speed(m/h)', 'humidity(%)']
data[columns] = data[columns].apply(pd.to_numeric, errors='coerce')

# Eliminare rânduri cu valori lipsă în variabilele țintă
data = data.dropna(subset=['min_ground_temp(°C)', 'min_air_temp(°C)'])

# Selecție date necesare
features = data[['sol_number', 'max_ground_temp(°C)', 'min_ground_temp(°C)', 'max_air_temp(°C)', 'min_air_temp(°C)', 'mean_pressure(Pa)', 'wind_speed(m/h)', 'humidity(%)']]
targets = data[['min_ground_temp(°C)', 'min_air_temp(°C)']]

# Split pentru setul de antrenare și testare
X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=42)

# Imputare date lipsă
imputer = SimpleImputer()
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

# Normalizare
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train_imputed)
X_test_scaled = scaler.transform(X_test_imputed)

# Construire model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X_train_scaled.shape[1],)),
    tf.keras.layers.Dense(units=2, activation='linear')  # Ajustează numărul de unități în funcție de nevoi
])

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae', 'mse'])

# Antrenare model
history = model.fit(X_train_scaled, y_train, epochs=50, verbose=1, validation_data=(X_test_scaled, y_test))

# Evaluare model
y_pred = model.predict(X_test_scaled)

# Analiza rezultatelor
for i in range(len(y_test.columns)):
    mse = mean_squared_error(y_test.iloc[:, i], y_pred[:, i])
    r2 = r2_score(y_test.iloc[:, i], y_pred[:, i])
    print(f'Mean Squared Error for {y_test.columns[i]}: {mse}')
    print(f'R2 Score for {y_test.columns[i]}: {r2}')

# Plotarea evoluției erorii în timpul antrenării
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
