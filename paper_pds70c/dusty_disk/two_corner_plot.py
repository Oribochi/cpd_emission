import numpy as np
import matplotlib.pyplot as plt
import corner
import smplotlib
import matplotlib
matplotlib.rcParams['axes.linewidth'] = 1. # the default is too thin

# the following should avoid having the silly-looking "10^0, 10^1, 10^2",
#  the unnecessarily complicated and "noisy" versions of "1, 10, 100"!
matplotlib.rcParams['axes.formatter.min_exponent'] = 5
matplotlib.rcParams['axes.formatter.limits'] = (-4,4)
matplotlib.rcParams['axes.formatter.useoffset'] = False  # also usually better
# Generate some random data for demonstration
np.random.seed(42)
data1 = np.random.multivariate_normal([0, 0], [[1, 0.5], [0.5, 1]], 1000)
data2 = np.random.multivariate_normal([2, 2], [[1, -0.5], [-0.5, 1]], 1000)

# Create a figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(10, 10))

# Plot the first corner plot
corner.corner(data1, fig=fig, color='b', alpha=0.5, labels=['x1', 'y1'])

# Plot the second corner plot
corner.corner(data2, fig=fig, color='r', alpha=0.5, labels=['x2', 'y2'])

axes[0,1].plot([0,1], [0,1], color='k', linestyle='--')  # Example line for reference
axes[0,1].tick_params(axis='both', direction='in', pad=-15)

# Optional: Add LaTeX in ticks or labels
axes[0, 1].set_xlabel(r"$x$", labelpad=-20)
axes[0, 1].set_ylabel(r"$y$", labelpad=-30)


# Adjust layout to prevent overlap
plt.tight_layout()
plt.show()