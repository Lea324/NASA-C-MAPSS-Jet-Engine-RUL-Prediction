import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ------------------------------------------------------------
# Data
# ------------------------------------------------------------
data = {
    'Ratio': [1, 2, 3, 4, 5],
    'Train_MAE': [3.8788, 3.6193, 3.6411, 3.5812, 3.6743],
    'Predict_MAE': [14.8872, 14.6575, 14.4098, 14.9349, 14.6091],
    'OOF_MAE': [14.0082, 14.4584, 14.0043, 14.7919, 14.8088],
    'Train_Time': [566.0462, 609.2696, 590.1608, 597.8432, 635.1659],
    'Over': [56, 56, 51, 54, 49],
    'Under': [44, 44, 49, 46, 51],
}
df = pd.DataFrame(data)

# Extra point for ratio 3 with batch size 128
extra = {'Ratio': 3, 'Train_Time': 583.5219, 'OOF_MAE': 14.0043}

# ------------------------------------------------------------
# Global style
# ------------------------------------------------------------
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
})
# ============================================================
# PLOT 1: Training Time vs OOF MAE (scatter, kept as-is)
# ============================================================
fig1, ax1 = plt.subplots(figsize=(8, 6))
ax1.scatter(df['Train_Time'], df['OOF_MAE'],
            s=150, c='cornflowerblue', edgecolors='white', zorder=5,
            label='Over/Under Estimation \n parameter ratio')

for i, row in df.iterrows():
    ax1.annotate(f"{int(row['Ratio'])}:{1}",# i:1
                 (row['Train_Time'], row['OOF_MAE']),
                 textcoords="offset points", xytext=(8, 6),
                 ha='left', fontsize=12, fontweight='bold')


ax1.set_xlabel('Training Time (seconds)')
ax1.set_ylabel('OOF MAE')
ax1.set_title('Training Time vs OOF MAE')
ax1.legend(loc='lower right', fontsize=12)
ax1.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

# ============================================================
# PLOT 2: Over/Under Deviation from 50/50 (column chart)
# ============================================================
fig2, ax2 = plt.subplots(figsize=(8, 6))
deviation = np.abs(df['Over'] - 50)

bars2 = ax2.bar(df['Ratio'], deviation,
                color='grey', edgecolor='grey', width=0.6)

# Highlight best (lowest deviation)
best_idx = deviation.idxmin()
bars2[best_idx].set_color('green')

# Value labels
for bar in bars2:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width() / 2, height + 0.1,
             f'{height:.0f}', ha='center', va='bottom',
             fontsize=12, fontweight='bold')

ax2.set_xlabel('Ratio')
ax2.set_ylabel('|Over% - 50|')
ax2.set_title('Over/Under Estimation Deviation from 50/50')
ax2.set_xticks(df['Ratio'])
ax2.grid(True, axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

# ============================================================
# PLOT 3: Overfitting Gap (OOF MAE - Train MAE) (column chart)
# ============================================================
fig3, ax3 = plt.subplots(figsize=(8, 6))
gap = df['OOF_MAE'] - df['Train_MAE']

bars3 = ax3.bar(df['Ratio'], gap,
                color='grey', edgecolor='grey', width=0.6)

# Highlight smallest gap
min_gap_idx = gap.idxmin()
bars3[min_gap_idx].set_color('green')

# Value labels
for bar in bars3:
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width() / 2, height + 0.05,
             f'{height:.2f}', ha='center', va='bottom',
             fontsize=12, fontweight='bold')

ax3.set_xlabel('Ratio')
ax3.set_ylabel('OOF MAE - Train MAE')
ax3.set_title('Overfitting Gap (OOF MAE - Train MAE)')
ax3.grid(True, axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()