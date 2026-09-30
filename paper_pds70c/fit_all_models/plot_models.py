# plot all the models
# plot MHD disk

import numpy as np
import matplotlib.pyplot as plt
from numpy import pi, log, log10
import smplotlib
import matplotlib
matplotlib.rcParams['axes.linewidth'] = 1. # the default is too thin

# the following should avoid having the silly-looking "10^0, 10^1, 10^2",
#  the unnecessarily complicated and "noisy" versions of "1, 10, 100"!
matplotlib.rcParams['axes.formatter.min_exponent'] = 5
matplotlib.rcParams['axes.formatter.limits'] = (-4,4)
matplotlib.rcParams['axes.formatter.useoffset'] = False  # also usually better
plt.style.use('tableau-colorblind10')

# First our data with the error bars
nus = [97.5E9, 145005402197.7, 343.5E9, 671E9] # [nu1, nu2] are the x-axis values , 671E9
Fnus = [12e-06, 21.4e-06, 0.000121, 0.000143] # [Fnu1, Fnu2] are the y-axis values in Jy , 0.000122
sFnus = [4.7e-06, 4.1e-06, 0.000013, 0.000115] # [sigma_Fnu1, sigma_Fnu2] are the y-axis errors in Jy , 0.000090

# to mJy
nu = np.array(nus)
Fnus = np.array(Fnus)*1E3
sFnus = np.array(sFnus)*1E3
# blackbody emission
nu_arr=np.logspace(log10(nus[0]), log10(nus[-1]), 100)
# array of fluxes
flux0=0.010
nu0=nu[0]
fluxes=(flux0*(nu_arr/nu0)**2)

# obtaining our txt
fig = plt.figure(figsize=(6, 4), dpi=300)
nu_array, flux_zhu = np.loadtxt('nus_fluxes_zhu.txt', unpack=True)
nu_array2, flux_Hminus = np.loadtxt('nus_fluxes_Hminus.txt', unpack=True)
nu_array3, flux_ff1, flux_ff2 = np.loadtxt('best_fit_HII_fluxes.txt', unpack=True)
nu_array4, flux_uniform1, flux_uniform2 = np.loadtxt('best_fit_uniform_fluxes.txt', unpack=True)
plt.errorbar(np.array(nus)/10**9, Fnus, yerr=sFnus, fmt='o', label='Data', color='C0')
plt.plot(nu_array/10**9, flux_zhu, label='Dusty disk', color='C1')
plt.plot(nu_array3/10**9, flux_ff2, label='Free-free (HII)',color='black', linestyle=':')
plt.plot(nu_array4/10**9, flux_uniform1, label='Uniform slab', color='C3')
plt.plot(nu_array2/10**9, flux_Hminus, label='Magnetic disk', color='C4', linestyle='--')
plt.plot(nu_array/10**9, fluxes, label='Blackbody', color='C5', linewidth=5, alpha=0.3)
plt.loglog()
plt.xlabel(r'$\nu$ (GHz)')
plt.ylabel(r'$F_{\nu}$ (mJy)')
plt.legend(loc='upper left')
plt.legend()
plt.savefig('all_models_comparison.png', dpi=300, bbox_inches='tight')
plt.show()
