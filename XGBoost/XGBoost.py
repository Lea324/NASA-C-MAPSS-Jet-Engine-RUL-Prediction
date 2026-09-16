import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import GroupKFold
from sklearn.model_selection import RandomizedSearchCV #this is for hyperparameter tuning
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import time
# self-defined function to conservatively predict the RUL
def conservative_predict(y_true,y_predict):
    error= y_predict-y_true
    over_penalty = 5 #range 
    under_penalty = 1

    grad= np.where(error>0,
                   2*over_penalty*error,
                     2*under_penalty*error)

    #Hessian
    hess = np.where(error>0,
                    np.full_like(error,2*over_penalty),
                    np.full_like(error,2*under_penalty))
    return grad,hess
#import the dataset
df = pd.read_csv("C:\\Engine URL Prediction\\Aircraft Engine Dataset\\FD001\\train.csv")
# Alreaady preprocessed the data in Linear_Regression_baseline.py, so we can directly use it here
X = df.drop(['max_cycle', 'RUL','health_stage'], axis=1)
y_train = df['RUL']
# Load the test data
df_test = pd.read_csv("C:\\Engine URL Prediction\\Aircraft Engine Dataset\\FD001\\test.csv")
df_test_last = df_test.loc[df_test.groupby('unit_number')['time_in_cycles'].idxmax()]  # only keep the last cycle of each unit_number
X_test = df_test_last[X.columns]
y_test = pd.read_csv("C:\\Engine URL Prediction\\Aircraft Engine Dataset\\FD001\\RUL_targets.csv")['true_RUL'].astype(float).to_numpy()
# Train the XGBoost model
# Train the XGBoost model
train_begin_time = time.time()

model = XGBRegressor(
    random_state=42,
    objective= conservative_predict,  # Use the custom objective function
    eval_metric="mae"
)

param_dist = {
    "n_estimators": [100, 200, 300, 500, 800, 1000],
    "max_depth": [2, 3, 4, 5, 6],
    "learning_rate": [0.01, 0.03, 0.05, 0.1],
    "subsample": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    "min_child_weight": [1, 3, 5, 7, 10, 15],
    "gamma": [0.1, 0.2, 0.5, 1, 2],
    "reg_alpha": [0.1, 0.5, 1, 5, 10],
    "reg_lambda": [1, 5, 10, 20, 50],
    "max_delta_step": [0, 1, 5, 10]
}

cv = GroupKFold(n_splits=10)

random_search = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_dist,
    n_iter=200, 
    scoring="neg_mean_absolute_error",
    cv=cv,
    verbose=2,
    random_state=42,
    n_jobs=-1
)

random_search.fit(
    X,
    y_train,
    groups=df["unit_number"]
)

# Get the best model
model = random_search.best_estimator_


train_end_time = time.time()
training_time = train_end_time - train_begin_time
# Prediction
predict_begin_time = time.time()

y_pred = model.predict(X_test)

predict_end_time = time.time()
prediction_time = predict_end_time - predict_begin_time
y_pred_capped = np.clip(y_pred, 0, 125)  # Cap predictions at 0 to avoid negative RUL values

# Calculate metrics
mae = mean_absolute_error(y_test, y_pred)
mae_train = mean_absolute_error(y_train, model.predict(X))
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)


# Print results
print("\n===== Model Performance =====")
print(f"Predict MAE: {mae:.4f}")
print(f"Training MAE: {mae_train:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R²: {r2:.4f}")

print("\n===== Time =====")
print(f"Training/Tuning Time: {training_time:.4f} seconds")
print(f"Prediction Time: {prediction_time:.4f} seconds")

print("\n===== Best Hyperparameters =====")
for parameter, value in random_search.best_params_.items():
    print(f"{parameter}: {value}")

print("\n===== Best CV Performance =====")
print(f"Best CV MAE: {-random_search.best_score_:.4f}")# double check if the value same to the training mae above? 

# Safty check for overestimation and underestimation
overestimation_mask = y_pred > y_test   
underestimation_mask = y_pred < y_test  

over_count = np.sum(overestimation_mask)
under_count = np.sum(underestimation_mask)
over_ratio = over_count / len(y_test) * 100
under_ratio = under_count / len(y_test) * 100

mean_overestimation = np.mean(y_pred[overestimation_mask] - y_test[overestimation_mask]) if over_count > 0 else 0
mean_underestimation = np.mean(y_test[underestimation_mask] - y_pred[underestimation_mask]) if under_count > 0 else 0

print("\n===== Overestimation and Underestimation Analysis =====")
print(f"Overestimation Count: {over_count}/{len(y_test)} ({over_ratio:.2f}%)")
print(f"Underestimation Count: {under_count}/{len(y_test)} ({under_ratio:.2f}%)")
print(f"Mean Overestimation: {mean_overestimation:.4f}")
print(f"Mean Underestimation: {mean_underestimation:.4f}")