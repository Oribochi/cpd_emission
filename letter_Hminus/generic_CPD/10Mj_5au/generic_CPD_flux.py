from flux_gas_dust_mag_disk import F_nu
from dustyCPD import *
from units_astro import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['axes.linewidth'] = 1. # the default is too thin

# the following should avoid having the silly-looking "10^0, 10^1, 10^2",
#  the unnecessarily complicated and "noisy" versions of "1, 10, 100"!
matplotlib.rcParams['axes.formatter.min_exponent'] = 5
matplotlib.rcParams['axes.formatter.limits'] = (-4,4)
matplotlib.rcParams['axes.formatter.useoffset'] = False  # also usually better
plt.style.use('tableau-colorblind10')

# Calculate for this specific case:

print('Planet mass =', Mp/Mj, 'Mjup')
print('Orbital distance =', a/au, 'AU')

Mpdot=1e-6*Mj/yr # 1 Jupiter mass per Myr
Bps=600 # Gauss
zeta1=1e-2
zeta2=1e-5
zeta3=1e-8
zeta4=0

# calculate the flux from 50 to 1000 GHz
nu_arr=np.logspace(log10(50e9), log10(1000e9), 10)
flux_arr1=np.array([F_nu(nu, Mpdot, Bps, zeta1) for nu in nu_arr])
flux_arr2=np.array([F_nu(nu, Mpdot, Bps, zeta2) for nu in nu_arr])
flux_arr3=np.array([F_nu(nu, Mpdot, Bps, zeta3) for nu in nu_arr])
flux_arr4=np.array([F_nu(nu, Mpdot, Bps, zeta4) for nu in nu_arr])

# save all fluxes in a text file
np.savetxt(f'fluxes_{int(Mp/Mj)}Mj_{int(a/au)}au.txt', np.column_stack((nu_arr/1e9, flux_arr1, flux_arr2, flux_arr3, flux_arr4)), header='nu(GHz) flux_zeta1(mJy) flux_zeta2(mJy) flux_zeta3(mJy) flux_zeta4(mJy)')