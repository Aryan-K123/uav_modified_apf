import matplotlib.pyplot as plt
import numpy as np
# Data
map_densities = [0.0702, 0.1103, 0.1264, 0.152]
success_apf = [1.098950605, 1.144638515, 1.099025299, 1.13445657]
success_modified_apf = [1.105241021, 1.113108595, 1.082935745, 1.1250119]
print(np.mean(success_apf))
print(np.mean(success_modified_apf))
print((np.mean(success_apf)-np.mean(success_modified_apf))/np.mean(success_apf))
map_labels = ['Map 1', 'Map 2', 'Map 3', 'Map 4']
combined_labels = [f"{label}\n{density:.4f}" for label, density in zip(map_labels, map_densities)]

# Set global font size
plt.rcParams.update({'font.size': 25})

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(map_densities, success_apf, marker='o', markersize=12, color='blue', label='APF')
plt.plot(map_densities, success_modified_apf, marker='s', markersize=12, color='green', label='Modified APF')

# Dotted vertical lines
baseline = min(success_apf + success_modified_apf) - 0.005
for x, y in zip(map_densities, success_apf):
    plt.plot([x, x], [baseline, y], linestyle=':', color='orange', linewidth=1.5)
for x, y in zip(map_densities, success_modified_apf):
    plt.plot([x, x], [baseline, y], linestyle=':', color='orange', linewidth=1.5)

# X-axis ticks
plt.xticks(map_densities, combined_labels)

# Labels and Title
plt.xlabel('Obstacle Density')
plt.ylabel('Average Trajectory Length')
# plt.title('Average Trajectory Length vs Obstacle Density')
plt.ylim(1.075, 1.15)

plt.grid(False)
plt.legend()
plt.tight_layout()
plt.show()
