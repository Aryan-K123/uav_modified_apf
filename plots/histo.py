import matplotlib.pyplot as plt
import numpy as np

# === Labels for X axis groups ===
obstacle_types = [
    'Light Map\n(2.0, 2.0, 2.0)',
    'Heavy Map\n(2.0, 2.0, 2.0)',
    'Heavy Map\n(1.7, 1.7, 2.0)'
]

# === Your data ===
standard_apf = [0.6974431446	, 0.0, 0.5635	]
our_method   = [0.6887523407	, 0.4645142857	, 0.4986142857	]

# === Plotting ===
bar_width = 0.35
x = np.arange(len(obstacle_types))

fig, ax = plt.subplots(figsize=(10, 6))

bars1 = ax.bar(x - bar_width/2, standard_apf, bar_width, label='Standard APF', color='purple')
bars2 = ax.bar(x + bar_width/2, our_method, bar_width, label='Our Method', color='lightgreen')

# Axis labels and ticks
ax.set_xlabel('Obstacle Density Type (with APF parameters)', fontsize=12)
ax.set_ylabel('Average Obstacle Distance', fontsize=12)
ax.set_title('Standard APF vs Our Method', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(obstacle_types, fontsize=10)
ax.legend()

# === Manually zoom Y-axis for clarity ===
ax.set_ylim(0.45, 0.75)

# Add values on top of bars
# Add values on top of bars, showing "Algorithm Failed" for zero
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        label = f'{height:.2f}' if height > 0.0 else 'Algorithm Failed'
        offset = 0.01 if height > 0 else 0.02  # slightly higher for "Algorithm Failed"
        ax.text(bar.get_x() + bar.get_width()/2., height + offset, label,
                ha='center', va='bottom', fontsize=9, color='red' if height == 0 else 'black')


plt.tight_layout()
plt.show()
