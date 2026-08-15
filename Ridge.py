import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
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

alphas = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
for alpha in alphas:
    model = Ridge(alpha=alpha)  # You can adjust the alpha parameter for regularization strength
    start_time = time.time()
    model.fit(X, y_train)
    end_time = time.time()
# Make predictions
    y_pred = model.predict(X_test)
# Calculate errors
    mae = mean_absolute_error(y_test, y_pred)
    mae_record=mae_record.append(mae)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    rmse_record=rmse_record.append(rmse)
    r2 = r2_score(y_test, y_pred)
    r2_record=r2_record.append(r2)
#plot the results
plt.figure(figsize=(10, 6))
plt.plot(mae_record, label='MAE', marker='o')
plt.plot(rmse_record, label='RMSE', marker='o')
plt.plot(r2_record, label='R²', marker='o')
plt.xticks(range(len(alphas)), alphas)
plt.xlabel("Alpha")
plt.ylabel("Error")
plt.title("Ridge Regression Results")
plt.legend()
plt.show()

#plot the error distribution
error = y_test - y_pred
plt.figure(figsize=(10, 6))
plt.plot(error, alpha=0.5)
plt.xlabel("Unit Number")
plt.ylabel("Prediction Error")
plt.title("Prediction Error Distribution")
plt.show()
