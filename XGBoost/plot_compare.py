import matplotlib.pyplot as plt
import numpy as np

# Data from experiments
configs = ['(1,1)', '(3,1)', '(5,1)', '(7,1)', '(5,2)', '(50,1)', '(100,1)']
predict_mae = [19.01, 16.88, 17.28, 20.73, 16.92, 22.80, 23.12]
train_mae = [21.98, 22.53, 20.93, 17.73, 22.14, 17.12, 17.95]
train_time = [514.87, 674.83, 692.3693, 686.66, 670.84, 686.33, 679.57]

# Create figure with two y-axes
fig, ax1 = plt.subplots(figsize=(10, 6))

# Plot MAE on left axis
ax1.plot(configs, predict_mae, 'o-', color='blue', linewidth=2, markersize=10, label='Predict MAE')
ax1.plot(configs, train_mae, '^-', color='black', linewidth=2, markersize=10, label='Train MAE')
ax1.set_xlabel('Penalty (Over, Under)')
ax1.set_ylabel('MAE', color='black')
ax1.tick_params(axis='y', labelcolor='black')
ax1.grid(True, alpha=0.3)

# Plot training time on right axis
ax2 = ax1.twinx()
ax2.plot(configs, train_time, 's--', color='red', linewidth=2, markersize=10, label='Train Time')
ax2.set_ylabel('Training Time (seconds)', color='black')
ax2.tick_params(axis='y', labelcolor='black')

# Highlight best config (5,1)
ax1.axvline(x=2, color='green', linestyle=':', alpha=0.5, linewidth=2)
ax1.text(2, max(predict_mae), ' Best', color='green', fontsize=12, fontweight='bold')


# Combine legends
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

plt.tight_layout()
plt.show()