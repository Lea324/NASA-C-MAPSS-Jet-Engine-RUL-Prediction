import time
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load the training data
df = pd.read_csv('C:\\Engine URL Prediction\\Aircraft Engine Dataset\\FD001\\train.csv')
# Preprocess the data
# Checking the missing values
#missing_values = df.isnull().sum()
#print(missing_values[missing_values > 0])
# Remove rows with missing values
df = df.dropna()# Remove rows that contain at least one missing value
#----------Actually, there are no missing values in the dataset-----------------
# Remove duplicates
#duplicates = df.duplicated().sum()
#print(f'Number of duplicate rows: {duplicates}')
df = df.drop_duplicates()
#------------------Actually, there are no duplicate rows in the dataset--------
#print(df["unit_number"].nunique()) # it tell me it have 100 unit_number
#===============================================================================
#Input features
X = df.drop(['max_cycle', 'RUL','health_stage'], axis=1)
Y = df['RUL']
start_time = time.time()
# Implementing the linear regression model
model = LinearRegression()
model.fit(X, Y) #the training step
end_time = time.time()
df_test = pd.read_csv('C:\\Engine URL Prediction\\Aircraft Engine Dataset\\FD001\\test.csv')
df_test_last = df_test.loc[
    df_test.groupby('unit_number')['time_in_cycles'].idxmax()
] #only keep the last cycle of each unit_number
X_test = df_test_last[X.columns]
Y_pred = model.predict(X_test)

#obtain the true RUL values from the RUL_targets.csv file
df_target = pd.read_csv('C:\\Engine URL Prediction\\Aircraft Engine Dataset\\FD001\\RUL_targets.csv')
Y_test = df_target['true_RUL'].astype(float).to_numpy()
#print("X_test shape:", X_test.shape)
#print("Y_pred shape:", Y_pred.shape)
#print("Y_test shape:", Y_test.shape)

#==================== Calculate errors==========================
mae = mean_absolute_error(Y_test, Y_pred)
rmse = np.sqrt(mean_squared_error(Y_test, Y_pred))
r2 = r2_score(Y_test, Y_pred)
print("MAE:", format(mae, ".4g"))
print("RMSE:", format(rmse, ".4g"))
print("R²:", format(r2, ".4g"))
print(f"Computational Time: {format(end_time - start_time, '.4g')} seconds")
#MAE: 26.12
#RMSE: 31.74
#R²: 0.4166
#Computational Time: 0.198 seconds
#=====================Visualization of the results=========================
plt.figure(figsize=(10, 6))
plt.plot(range(len(Y_test)), Y_test, color='blue', label='True RUL')
plt.plot(range(len(Y_pred)), Y_pred, color='red', label='Predicted RUL')
plt.title('True vs Predicted RUL')
plt.xlabel('Unit Number')
plt.ylabel('RUL')
plt.legend()
plt.show()
#new plot to visualize the error distribution
errors = Y_test - Y_pred
plt.figure(figsize=(10, 6))
plt.plot(range(len(errors)), errors, color='purple', alpha=0.7)
plt.title('Error Distribution')
plt.xlabel('Unit Number')
plt.ylabel('Error')
plt.show()