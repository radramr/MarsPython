import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from pyearth import Earth  # Import Earth from pyearth package
import matplotlib.pyplot as plt


# Load data
file_path = 'REMS_Mars_Dataset.csv'
df = pd.read_csv(file_path)

# Ensure numeric columns have the correct data type
numeric_columns = ['sol_number', 'max_ground_temp(°C)', 'min_ground_temp(°C)', 'max_air_temp(°C)',
                    'min_air_temp(°C)', 'mean_pressure(Pa)', 'wind_speed(m/h)', 'humidity(%)']
df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')

# Drop rows with missing values in the target variables
df = df.dropna(subset=['min_ground_temp(°C)', 'min_air_temp(°C)'])

# Select necessary data
features = df[['sol_number', 'max_ground_temp(°C)', 'min_ground_temp(°C)', 'max_air_temp(°C)',
                'min_air_temp(°C)', 'mean_pressure(Pa)', 'wind_speed(m/h)', 'humidity(%)']]
targets = df[['min_ground_temp(°C)', 'min_air_temp(°C)']]

# Split for training and testing
X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=42)

# Impute missing values
imputer = SimpleImputer()
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

# Normalize
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train_imputed)
X_test_scaled = scaler.transform(X_test_imputed)

# Build and train MARS model
mars_model = Earth()
mars_model.fit(X_train_scaled, y_train['min_ground_temp(°C)'])

# Predictions
y_pred_min_ground_temp = mars_model.predict(X_test_scaled)
mse_min_ground_temp = mean_squared_error(y_test['min_ground_temp(°C)'], y_pred_min_ground_temp)
r2_min_ground_temp = r2_score(y_test['min_ground_temp(°C)'], y_pred_min_ground_temp)
print(f'Mean Squared Error for min_ground_temp(°C): {mse_min_ground_temp}')
print(f'R2 Score for min_ground_temp(°C): {r2_min_ground_temp}')

# Build and train another MARS model for min_air_temp(°C)
mars_model_air_temp = Earth()
mars_model_air_temp.fit(X_train_scaled, y_train['min_air_temp(°C)'])

# Predictions for min_air_temp(°C)
y_pred_min_air_temp = mars_model_air_temp.predict(X_test_scaled)
mse_min_air_temp = mean_squared_error(y_test['min_air_temp(°C)'], y_pred_min_air_temp)
r2_min_air_temp = r2_score(y_test['min_air_temp(°C)'], y_pred_min_air_temp)
print(f'Mean Squared Error for min_air_temp(°C): {mse_min_air_temp}')
print(f'R2 Score for min_air_temp(°C): {r2_min_air_temp}')
