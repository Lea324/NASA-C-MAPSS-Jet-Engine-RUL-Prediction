import os
main_csv_local_path = 'RUL_targets.csv'
for dirname, _, filenames in os.walk(r'C:\Jet machine learning'):
    for filename in filenames:
        if filename == main_csv_local_path:
            DATA_DIR = os.path.join(dirname, filename)
            print(DATA_DIR)

import numpy as np #NumPy (numerical computing)
import pandas as pd #Pandas (data processing)
import matplotlib.pyplot as plt #Matplotlib (graphs and visualization)
import scipy #SciPy (engineering calculations)
import seaborn as sns #Seaborn (statistical data visualization)
import warnings #Warnings (to manage warning messages)
# ==========================================
# A) Imports + Global Config
# ==========================================
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['font.size'] = 10
np.random.seed(42) #set random seed for reproducibility, the number 42 is often used as a default seed..

# Configuration from Task
DATASET_NAME  = "C:\\Jet machine learning\\Aircraft Engine Dataset\\FD001\\RUL_targets.csv"
#unit_number: Unique identifier for each engine unit in the dataset.
#true_RUL: The actual Remaining Useful Life (RUL) of the engine unit, which is the target variable we want to predict.
#health_stage: Categorical variable indicating the health stage of the engine unit:RUL > 125	Healthy; 80 < RUL <= 125	Early degradation; 40 < RUL <= 80	Moderate degradation; 10 < RUL <= 40	Critical degradation; RUL <= 10	Near failure
EXACT_COLUMNS = ['unit_number', 'true_RUL', 'health_stage']
TARGET_COL = 'true_RUL' # Identifying 'true_RUL' as the primary target for this RUL dataset
# ==========================================
# B) Load + Validate
# ==========================================
def safe_read_csv(path):
    try:
        return pd.read_csv(path)
    
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to read dataset at {path}. Error: {e}")
        return None

df = safe_read_csv(DATA_DIR)

if df is not None:
    actual_cols = df.columns.tolist()
    missing_cols = [c for c in EXACT_COLUMNS if c not in actual_cols]
    extra_cols = [c for c in actual_cols if c not in EXACT_COLUMNS]
    
    print(f"--- Data Load Report: {DATASET_NAME} ---")
    if not missing_cols and not extra_cols:
        print("Column Validation: SUCCESS (Perfect match)")
    else:
        print(f"Column Validation: MISMATCH\n - Missing: {missing_cols}\n - Extra: {extra_cols}")

# ==========================================
# # C) Data Audit
# ==========================================
def perform_audit(df):
    if df is None: return
    print("\n--- Dataset Overview ---")
    print(f"Shape: {df.shape}")
    print(f"Duplicates: {df.duplicated().sum()}")
    print("\n--- Null & Type Audit ---")
    audit_df = pd.DataFrame({
        'Dtype': df.dtypes,
        'Nulls': df.isnull().sum(),
        'Null%': (df.isnull().sum() / len(df) * 100).round(2),
        'Unique': df.nunique()
    })
    print(audit_df)
    
    # Numeric Stability
    num_cols = df.select_dtypes(include=[np.number]).columns
    if not num_cols.empty:
        print("\n--- Numeric Stability ---")
        for col in num_cols:
            inf_count = np.isinf(df[col]).sum()
            if inf_count > 0: print(f"Warning: {col} contains {inf_count} inf values.")
            if df[col].std() < 1e-6: print(f"Warning: {col} has near-zero variance.")

def detect_date_cols(df):
    candidates = []
    for col in df.select_dtypes(include=['object']).columns:
        try:
            # Sample check to avoid heavy processing
            sample = df[col].dropna().head(5).astype(str)
            if any(sample.str.contains(r'\d{2,4}[-/]\d{1,2}[-/]\d{1,2}')):
                candidates.append(col)
        except: continue
    return candidates