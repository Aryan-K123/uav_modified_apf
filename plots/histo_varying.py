import matplotlib.pyplot as plt
import numpy as np

# Example success rates for two algorithms
algorithms = ['Standard APF', 'Modified APF']
success_rates = [0, 57.143]  # out of 100

# Positions for the bars on y-axis (horizontal bars)
y_pos = np.arange(len(algorithms))

plt.figure(figsize=(8, 3))
plt.barh(y_pos, success_rates, color=['skyblue', 'salmon'])

# Remove y-axis labels and ticks
plt.yticks([])

# Label x-axis
plt.xlabel('Success Rate (%)')

# Set x-axis limit to 0-100
plt.xlim(0, 100)

# Optionally add success rate text on bars
for i, v in enumerate(success_rates):
    plt.text(v + 1, i, str(v), va='center')

plt.title('Success Rate Comparison for Two Algorithms in Varying map')
plt.show()
