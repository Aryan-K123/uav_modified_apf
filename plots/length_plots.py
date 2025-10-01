import matplotlib.pyplot as plt

# Data
map_densities = [0.0702, 0.1103, 0.1264, 0.152]
success_apf = [1.35639329	, 1.144638515	, 1.742578657	, 1.876350939	]
success_modified_apf = [1.134687362	, 1.113108595	, 1.334418778	, 1.683846368	]

# Plotting
plt.figure(figsize=(8, 5))
plt.plot(map_densities, success_apf, marker='o', color='blue', label='APF')
plt.plot(map_densities, success_modified_apf, marker='s', color='green', label='Modified APF')

# Dotted vertical lines only up to the respective data points
for x, y in zip(map_densities, success_apf):
    plt.plot([x, x], [0, y], linestyle=':', color='orange', linewidth=1.5)
for x, y in zip(map_densities, success_modified_apf):
    plt.plot([x, x], [0, y], linestyle=':', color='orange', linewidth=1.5)

# Keep full grid (default vertical and horizontal lines)
plt.grid(True)

# X-axis ticks
plt.xticks(map_densities, [f"{x:.4f}" for x in map_densities])

# Labels and Title
plt.xlabel('Map Density')
plt.ylabel('Average Trajectory Length with penalty')
plt.title('Average Trajectory Length vs Map Density')
plt.ylim(0, 2)
plt.legend()
plt.tight_layout()

plt.show()
