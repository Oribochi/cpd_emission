# plot the rosseland mean opacity due to H-, H2-, He- and metals in a grid of density and temperature
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['axes.linewidth'] = 1. # the default is too thin

# now in a grid from rho=1e-12 to 1e-8 g/cm^3 and from T=500 to 5000 K of 100x100 points and a grid plot 
# of the rosseland mean opacity (save fig and data in a file)
rho_array=np.logspace(-12, -8, 100) # g/cm^3
T_array=np.logspace(3, 3.7, 100) # K
# we have the data in a txt file
kappa_rosseland_array=np.loadtxt("rosseland_opacity_grid.txt")
# nan values to 0
kappa_rosseland_array=np.nan_to_num(kappa_rosseland_array, nan=1/1000000)
# all numbers less than 1e-12 to 1e-12
kappa_rosseland_array[kappa_rosseland_array<1/1000000]=1/1000000
# plot the data
R, T = np.meshgrid(rho_array, T_array)

# now with a continuous colorbar from 1e-6 to 1e-2 viridis colormap
plt.figure(figsize=(4.7,3.7))
# continuous colormap with pcolormesh
cp = plt.pcolormesh(
    R, T, kappa_rosseland_array.T,
    norm=matplotlib.colors.LogNorm(vmin=1e-6, vmax=1e-2),
    cmap='GnBu',
    shading='auto'  # avoids gridline artifacts
)

plt.colorbar(cp, label='Rosseland Mean Opacity (cm$^2$ g$^{-1}$)')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Mass Density (g cm$^{-3}$)')
plt.ylabel('Temperature (K)')
plt.savefig("rosseland_opacity_grid_GnBu.pdf", dpi=300, bbox_inches='tight')
plt.show()
