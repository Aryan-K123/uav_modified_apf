import matplotlib.pyplot as plt

# Data
map_densities = [0.0702, 0.1103, 0.1264, 0.152]
success_apf = [71.42, 100, 28.571, 14.28]
success_modified_apf = [100, 100, 100, 42.85]
map_labels = ['Map 1', 'Map 2', 'Map 3', 'Map 4']

# Combine labels
combined_labels = [f"{label}\n{density:.4f}" for label, density in zip(map_labels, map_densities)]

# Set global font size
plt.rcParams.update({'font.size': 20})

# Plotting
plt.figure(figsize=(8, 5))
plt.plot(map_densities, success_apf, marker='o', markersize=12, color='blue', label='APF')
plt.plot(map_densities, success_modified_apf, marker='s', markersize=12, color='green', label='Modified APF')

# Dotted vertical lines
for x, y in zip(map_densities, success_apf):
    plt.plot([x, x], [0, y], linestyle=':', color='orange', linewidth=2.0)
for x, y in zip(map_densities, success_modified_apf):
    plt.plot([x, x], [0, y], linestyle=':', color='orange', linewidth=2.0)

# X-axis ticks and labels
plt.xticks(map_densities, combined_labels)

# Labels and Title
plt.xlabel('Obstacle Density')
plt.ylabel('Success Rate (%)')
# plt.title('Success Rate vs Obstacle Density')
plt.ylim(0, 120)
plt.legend()
plt.grid(False)
plt.tight_layout()
plt.show()
