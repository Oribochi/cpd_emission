# example of a SED up to 3 um (3.5 um is L band)
import numpy as np
import matplotlib.pyplot as plt
from units_astro import *
from flux_magnetic_disk import F_metals_Hminus
from numpy import log10
import smplotlib
import matplotlib
matplotlib.rcParams['axes.linewidth'] = 1. # the default is too thin

# the following should avoid having the silly-looking "10^0, 10^1, 10^2",
#  the unnecessarily complicated and "noisy" versions of "1, 10, 100"!
matplotlib.rcParams['axes.formatter.min_exponent'] = 5
matplotlib.rcParams['axes.formatter.limits'] = (-4,4)
matplotlib.rcParams['axes.formatter.useoffset'] = False  # also usually better
plt.style.use('tableau-colorblind10')

# the best fit parameters for the SED of the magnetic disk of PDS 70c
Mpdot=10**(-5.54)*Mj/yr
Bps=613 # in Gauss

nu_array=np.logspace(log10(1E9), 15, 200)

# calculate the flux density
flux_density = np.zeros(len(nu_array))
spectral_index = np.zeros(len(nu_array)-1)
for i in range(len(nu_array)):
    flux_density[i] = F_metals_Hminus(nu_array[i], Mpdot, Bps)
    if i>0:
        spectral_index[i-1] = log10(flux_density[i]/flux_density[i-1]) / log10(nu_array[i]/nu_array[i-1])

np.savetxt('SED_disk.txt', np.column_stack((nu_array, flux_density)), header='Frequency (Hz) Flux Density (mJy)', fmt='%e %e')
# plot the results
plt.figure(figsize=(4.8, 3.8))
plt.plot(nu_array/1e9, flux_density, label='SED of PDS 70c', color='C0', lw=2)
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'$\nu$ [GHz]')
plt.ylabel(r'$F_{\nu}$ [mJy]')
plt.ylim(1e-6, 1e4)
plt.savefig('SED_disk.png', dpi=300, bbox_inches='tight')
plt.show()


plt.figure(figsize=(4.8, 3.8))
plt.plot(nu_array[:-1]/1e9, spectral_index, label='Spectral Index', color='C1', lw=2)
plt.xscale('log')
plt.xlabel(r'$\nu$ [GHz]')
plt.ylabel(r'$\alpha$')
plt.ylim(-1, 2.2)
plt.savefig('Spectral_Index_disk.png', dpi=300, bbox_inches='tight')
plt.show()

nu_array=np.logspace(log10(1E9*60000), log10(1E9*300000), 200)
# calculate the flux density
flux_density = np.zeros(len(nu_array))
spectral_index = np.zeros(len(nu_array)-1)
for i in range(len(nu_array)):
    flux_density[i] = F_metals_Hminus(nu_array[i], Mpdot, Bps)
    if i>0:
        spectral_index[i-1] = log10(flux_density[i]/flux_density[i-1]) / log10(nu_array[i]/nu_array[i-1])


# Also now a zoom in the maximum
plt.figure(figsize=(4.8, 3.8))
plt.plot(nu_array/1e9, flux_density, label='SED of PDS 70c', color='C0', lw=2)
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'$\nu$ [GHz]')
plt.ylabel(r'$F_{\nu}$ [mJy]')
plt.savefig('SED_disk_zoom.png', dpi=300, bbox_inches='tight')
plt.show()

plt.figure(figsize=(4.8, 3.8))
plt.plot(nu_array[:-1]/1e9, spectral_index, label='Spectral Index', color='C1', lw=2)
plt.xscale('log')
plt.xlabel(r'$\nu$ [GHz]')
plt.ylabel(r'$\alpha$')
plt.savefig('Spectral_Index_disk_zoom.png', dpi=300, bbox_inches='tight')
plt.show()

# print the maximum flux density and the frequency at which it occurs
max_flux_density = np.max(flux_density)
max_freq = nu_array[np.argmax(flux_density)]
print(f'Maximum flux density: {max_flux_density:.2e} mJy at {max_freq/1e9:.2f} GHz')

