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
warnings.filterwarnings('ignore')
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
# ==========================================
# D) ETL (Safe + Reversible)
# ==========================================
df_clean = df.copy() if df is not None else None

def clean_pipeline(df_in):
    if df_in is None: return None
    df_c = df_in.copy()
    
    # 1. Strip and Unify Missing
    missing_tokens = ["", " ", "NA", "N/A", "null", "None", "nan", "NaN"]
    for col in df_c.columns:
        if df_c[col].dtype == 'object':
            df_c[col] = df_c[col].astype(str).str.strip().replace(missing_tokens, np.nan)
    
    # 2. Convert Numeric-looking strings
    for col in df_c.select_dtypes(include=['object']).columns:
        try:
            converted = pd.to_numeric(df_c[col], errors='coerce')
            if converted.notnull().sum() > (0.8 * len(df_c)):
                df_c[col] = converted
        except: pass
    
    # 3. Handle Duplicates
    dup_count = df_c.duplicated().sum()
    df_c = df_c.drop_duplicates(keep='first')
    
    # 4. Imputation Strategy
    for col in df_c.columns:
        null_count = df_c[col].isnull().sum()
        if null_count > 0:
            df_c[f"{col}__was_missing"] = df_c[col].isnull().astype(int)
            if pd.api.types.is_numeric_dtype(df_c[col]):
                df_c[col] = df_c[col].fillna(df_c[col].median())
            else:
                df_c[col] = df_c[col].fillna("Missing")
    
    # 5. Outlier Detection (Winsorization)
    num_cols = df_c.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        if "__was_missing" in col: continue
        q1 = df_c[col].quantile(0.25)
        q3 = df_c[col].quantile(0.75)
        iqr = q3 - q1
        outlier_rate = ((df_c[col] < (q1 - 1.5 * iqr)) | (df_c[col] > (q3 + 1.5 * iqr))).mean()
        if outlier_rate > 0:
            lower = df_c[col].quantile(0.01)
            upper = df_c[col].quantile(0.99)
            df_c[f"{col}__winsor"] = df_c[col].clip(lower, upper)
            
    return df_c, dup_count

if df is not None:
    perform_audit(df)
    date_cols = detect_date_cols(df)
    df_clean, dups_removed = clean_pipeline(df)

# ==========================================
# E) EDA & G) Feature Engineering
# ==========================================
def derive_features(df_proc):
    if df_proc is None: return None
    # String length features
    for col in df_proc.select_dtypes(include=['object']).columns:
        if col == 'health_stage': # Specific to this dataset
            df_proc[f"{col}__len"] = df_proc[col].astype(str).apply(len)
    return df_proc

df_clean = derive_features(df_clean)

if df_clean is not None:
    print("\n--- Descriptive Stats (Numeric) ---")
    numeric_df = df_clean.select_dtypes(include=[np.number])
    if not numeric_df.empty:
        stats = numeric_df.describe().T
        stats['skew'] = numeric_df.skew()
        stats['kurtosis'] = numeric_df.kurtosis()
        print(stats)

# ==========================================
# F) Visualization
# ==========================================
def plot_pipeline(df_orig, df_proc):
    if df_orig is None: return
    
    # 1. Missingness
    plt.figure(figsize=(10, 4))
    null_pct = df_orig.isnull().mean() * 100
    if null_pct.any():
        null_pct = null_pct[null_pct > 0].sort_values(ascending=False).head(30)
        sns.barplot(x=null_pct.index, y=null_pct.values, hue=null_pct.index, palette='viridis', legend=False)
        plt.title("Missing Data Percentage by Column")
        plt.xticks(rotation=45)
        plt.show()

    # 2. Distributions
    num_cols = df_proc.select_dtypes(include=[np.number]).columns[:12]
    if len(num_cols) > 0:
        fig, axes = plt.subplots(nrows=(len(num_cols)+3)//4, ncols=4, figsize=(16, 4 * ((len(num_cols)+3)//4)))
        axes = axes.flatten()
        for i, col in enumerate(num_cols):
            sns.histplot(df_proc[col], kde=True, ax=axes[i], color='skyblue')
            axes[i].set_title(f"Dist: {col}")
        plt.tight_layout()
        plt.show()

    # 3. Boxplots (Outliers)
    if len(num_cols) > 0:
        fig, axes = plt.subplots(nrows=(len(num_cols)+3)//4, ncols=4, figsize=(16, 4 * ((len(num_cols)+3)//4)))
        axes = axes.flatten()
        for i, col in enumerate(num_cols):
            sns.boxplot(y=df_proc[col], ax=axes[i], color='salmon')
            axes[i].set_title(f"Box: {col}")
        plt.tight_layout()
        plt.show()

    # 4. Categorical Counts
    cat_cols = df_orig.select_dtypes(include=['object', 'category']).columns[:6]
    if len(cat_cols) > 0:
        fig, axes = plt.subplots(nrows=(len(cat_cols)+1)//2, ncols=2, figsize=(14, 5 * ((len(cat_cols)+1)//2)))
        axes = axes.flatten()
        for i, col in enumerate(cat_cols):
            top_vals = df_orig[col].value_counts().head(10)
            sns.barplot(x=top_vals.values, y=top_vals.index, hue=top_vals.index, palette='magma', ax=axes[i], legend=False)
            axes[i].set_title(f"Top Values: {col}")
        plt.tight_layout()
        plt.show()

    # 5. Correlation Heatmap
    corr_cols = df_proc.select_dtypes(include=[np.number]).columns
    if len(corr_cols) >= 2:
        plt.figure(figsize=(10, 8))
        if len(corr_cols) > 20: 
            print("Note: Capping correlation matrix to top 20 numeric features.")
            corr_cols = corr_cols[:20]
        sns.heatmap(df_proc[corr_cols].corr(), annot=True, fmt=".2f", cmap='coolwarm', center=0)
        plt.title("Correlation Matrix (Pearson)")
        plt.show()

    # 6. Target Analysis
    if TARGET_COL in df_proc.columns:
        print(f"\n--- Target-Aware Analysis: {TARGET_COL} ---")
        if pd.api.types.is_numeric_dtype(df_proc[TARGET_COL]):
            high_corr = df_proc.select_dtypes(include=[np.number]).corr()[TARGET_COL].sort_values(ascending=False)
            print(f"Top Correlations with {TARGET_COL}:")
            print(high_corr.head(10))
            
            # Bivariate scatter with top feature
            top_feat = high_corr.index[1] if len(high_corr) > 1 else None
            if top_feat:
                plt.figure(figsize=(8, 5))
                sns.scatterplot(data=df_proc, x=top_feat, y=TARGET_COL, alpha=0.5)
                plt.title(f"{TARGET_COL} vs {top_feat}")
                plt.show()
        
        if 'health_stage' in df_proc.columns:
            plt.figure(figsize=(10, 5))
            sns.boxplot(data=df_proc, x='health_stage', y=TARGET_COL, hue='health_stage', palette='Set2', legend=False)
            plt.title(f"{TARGET_COL} Distribution by Health Stage")
            plt.show()

if df is not None:
    plot_pipeline(df, df_clean)

# ==========================================
# H) Final Artifact Output
# ==========================================
if df is not None and df_clean is not None:
    print("\n--- Final Data Quality Summary ---")
    print(f"Original Shape: {df.shape}")
    print(f"Cleaned Shape:  {df_clean.shape}")
    print(f"Duplicates Removed: {dups_removed}")
    print(f"Datetime Columns Detected: {len(date_cols)}")
    print(f"Total Missing Post-Imputation: {df_clean.isnull().sum().sum()}")
    print("\n--- Processed Data Head ---")
    print(df_clean.head())