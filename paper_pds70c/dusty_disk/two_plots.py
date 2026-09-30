import numpy as np
import matplotlib.pyplot as plt
import corner
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

# Generate data
data = np.random.multivariate_normal(mean=[0, 0], cov=np.eye(2), size=1000)

# Main figure and grid
fig = plt.figure(figsize=(10, 10))
gs = GridSpec(3, 3, figure=fig)

# Create nested 2x2 GridSpec for the corner plot
corner_spec = GridSpecFromSubplotSpec(2, 2, subplot_spec=gs[1:, :2])

# Allocate all 2x2 axes, even if some stay unused
corner_axes = np.empty((2, 2), dtype=object)
for i in range(2):
    for j in range(2):
        corner_axes[i, j] = fig.add_subplot(corner_spec[i, j])

# Flatten and pass to corner
corner.corner(
    data,
    color='blue',
    bins=20,
    labels=['x', 'y'],
    fig=fig,
    axes=corner_axes.flatten().tolist(),  # Must be a flat list
    plot_datapoints=False
)

# Another plot in top-right 2 rows of last column
ax_function = fig.add_subplot(gs[:2, 1:])
x = np.linspace(0, 10, 100)
y = np.sin(x)
ax_function.plot(x, y, label='sin(x)')
ax_function.set_title('Plot of sin(x)')
ax_function.legend()
ax_function.xaxis.set_label_position('top') 
ax_function.yaxis.set_label_position('right')
# Hide or use bottom-right slot
ax_unused = fig.add_subplot(gs[2, 2])
ax_unused.axis('off')

plt.tight_layout()
plt.show()