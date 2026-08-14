import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
# Load the training data
df = pd.read_csv('C:\\Jet machine learning\\Aircraft Engine Dataset\\FD001\\train.csv')
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
X = df.drop(['unit_number','max_cycle', 'RUL','health_stage'], axis=1)
Y = df['RUL']

# Implementing the linear regression model
model = LinearRegression()
model.fit(X, Y) #the training step
print("Model trained successfully!")