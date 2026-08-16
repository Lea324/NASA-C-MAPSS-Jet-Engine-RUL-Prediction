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

alphas = np.linspace(0.001, 10, 1000)  
mae_record = []
rmse_record = []
r2_record = []
for alpha in alphas:
    model = Ridge(alpha=alpha)  # You can adjust the alpha parameter for regularization strength
    start_time = time.time()
    model.fit(X, y_train)
    end_time = time.time()
# Make predictions
    y_pred = model.predict(X_test)
# Calculate errors
    mae = mean_absolute_error(y_test, y_pred)
    mae_record.append(mae)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    rmse_record.append(rmse)
    r2 = r2_score(y_test, y_pred)
    r2_record.append(r2)
#find the best alpha based on MAE
best_alpha_index = np.argmin(mae_record)
best_alpha = alphas[best_alpha_index]

#plot the results
plt.figure(figsize=(10, 6))
#mark the best alpha on the plot
plt.axvline(x=best_alpha_index, color='red', linestyle='--', label=f'Best Alpha: {best_alpha:.3f}')
plt.plot(mae_record, label='MAE')
plt.xticks(100 * np.arange(len(alphas)) / len(alphas), [f'{alpha:.2f}' for alpha in alphas], rotation=45)
plt.xlabel("Regularization Strength ($\alpha$) [Log Scale]")
plt.xscale('log')
plt.ylabel("Mean Absolute Error (MAE)")
plt.legend(loc='best', fontsize=10)

plt.show()
print(f"Best Alpha: {best_alpha:.3f}")
print(f"MAE: {mae_record[best_alpha_index]:.4g}")
print(f"RMSE: {rmse_record[best_alpha_index]:.4g}")
print(f"R²: {r2_record[best_alpha_index]:.4g}")
print(f"Computational Time: {format(end_time - start_time, '.4g')} seconds")
