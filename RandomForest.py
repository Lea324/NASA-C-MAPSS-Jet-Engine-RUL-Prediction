import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import time
# Load the training data
df = pd.read_csv('C:\\Engine URL Prediction\\Aircraft Engine Dataset\\FD001\\train.csv')
# Alreaady preprocessed the data in Linear_Regression_baseline.py, so we can directly use it here
X = df.drop(['max_cycle', 'RUL','health_stage'], axis=1)
y_train = df['RUL']
# Load the test data
df_test = pd.read_csv('C:\\Engine URL Prediction\\Aircraft Engine Dataset\\FD001\\test.csv')
df_test_last = df_test.loc[df_test.groupby('unit_number')['time_in_cycles'].idxmax()]  # only keep the last cycle of each unit_number
X_test = df_test_last[X.columns]
y_test = pd.read_csv('C:\\Engine URL Prediction\\Aircraft Engine Dataset\\FD001\\RUL_targets.csv')['true_RUL'].astype(float).to_numpy()
# Train the RandomForest model
model = RandomForestRegressor(n_estimators=100, random_state=42)
start_time = time.time()
model.fit(X, y_train)
end_time = time.time()
# Make predictions
y_pred = model.predict(X_test)
# Calculate errors
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
# Print the results
print(f"MAE: {mae:.4g}")
print(f"RMSE: {rmse:.4g}")
print(f"R²: {r2:.4g}")
print(f"Computational Time: {format(end_time - start_time, '.4g')} seconds")
