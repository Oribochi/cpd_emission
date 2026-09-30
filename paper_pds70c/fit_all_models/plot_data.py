# upper limit for Mpdt using a fixed Bps

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

# To mJy
nu = np.array(nus)
Fnus = np.array(Fnus)*1E3
sFnus = np.array(sFnus)*1E3

# plot the obtained SED
fig = plt.figure(figsize=(4.8, 3.8), dpi=300)

plt.errorbar(np.array(nus)/10**9, Fnus, yerr=sFnus, fmt='o', label='Data', color='C0')
plt.plot(np.array(nus)/10**9, Fnus, color='C1')
plt.loglog()
plt.xlabel(r'$\nu$ (GHz)')
plt.ylabel(r'$F_{\nu}$ (mJy)')
plt.legend(loc='upper left', frameon=True)
plt.savefig('SED.png', dpi=300, bbox_inches='tight')
plt.show()

# plot the obtained SED
fig = plt.figure(figsize=(4.8, 3.8), dpi=300)
plt.errorbar(np.array(nus)/10**9, Fnus, yerr=sFnus, fmt='o', label='Data', color='C0')
plt.loglog()
plt.xlabel(r'$\nu$ (GHz)')
plt.ylabel(r'$F_{\nu}$ (mJy)')
plt.legend(loc='upper left', frameon=True)
plt.savefig('SED_data.png', dpi=300, bbox_inches='tight')
plt.show()
