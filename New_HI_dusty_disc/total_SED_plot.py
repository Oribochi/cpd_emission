# Plot all the SED from 10 GHz to 100000 GHz
# plot MHD disk

import numpy as np
import matplotlib.pyplot as plt
from flux_gas_dust_simple_disk import F_nu as F
from units_astro import *
import corner
from numpy import log10
import matplotlib.pyplot as plt
# now plot everything together
plt.style.use('tableau-colorblind10')
# now changing the stile not 10^n but 100, 1000, etc
from matplotlib.ticker import ScalarFormatter
import matplotlib
matplotlib.rcParams['axes.linewidth'] = 1. # the default is too thin
# the following should avoid having the silly-looking "10^0, 10^1, 10^2",
#  the unnecessarily complicated and "noisy" versions of "1, 10,
# 100"!
matplotlib.rcParams['axes.formatter.min_exponent'] = 5
matplotlib.rcParams['axes.formatter.limits'] = (-4,4)
matplotlib.rcParams['axes.formatter.useoffset'] = False 


# First our data with the error bars
nus = [97.5E9, 145005402197.7, 343.5E9, 671E9] # [nu1, nu2] are the x-axis values , 671E9
Fnus = [12e-06, 21.4e-06, 0.000121, 0.000143] # [Fnu1, Fnu2] are the y-axis values in Jy , 0.000122
sFnus = [4.7e-06, 4.1e-06, 0.000013, 0.000115] # [sigma_Fnu1, sigma_Fnu2] are the y-axis errors in Jy , 0.000090

# To mJy
nu = np.array(nus)
Fnus = np.array(Fnus)*1E3
sFnus = np.array(sFnus)*1E3
# array of frequencies
nu_arr=np.logspace(10, 16.5, 500) # from 10 GHz to 100000 GHz


best_fit1=np.loadtxt('best_fit_params_magnetic_disk_with_zeta.txt', dtype=str)
Mpdot1=10**(float(best_fit1[0][1]))*Mj/yr
dust_to_gas1=10**(-10)
alpha1=0.01

best_fit2=np.loadtxt('best_fit_params_magnetic_disk_with_zeta_constant_Mpdot.txt', dtype=str)
alpha2=10**(float(best_fit2[0][1]))
dust_to_gas2=10**(-10)
Mpdot2=10**(-6)*Mj/yr


# array of fluxes
F_arr_ff1=np.zeros(len(nu_arr))
F_arr_ff2=np.zeros(len(nu_arr))

for i in range(len(nu_arr)):
    F_arr_ff1[i]=F(nu_arr[i], Mpdot1, alpha1, zeta=dust_to_gas1)
    F_arr_ff2[i]=F(nu_arr[i], Mpdot2, alpha2, zeta=dust_to_gas2)

# save nu_array and fluxes
np.savetxt('totalSED_const_Mpdot.txt', np.column_stack((nu_arr, F_arr_ff1)), header='nu [GHz] F_nu [mJy]', fmt='%f %f')
np.savetxt('totalSED_const_alpha.txt', np.column_stack((nu_arr, F_arr_ff2)), header='nu [GHz] F_nu [mJy]', fmt='%f %f')

plt.figure(figsize=(4.8,3.8))
plt.plot(nu_arr/1E9, F_arr_ff1, label=r'Fixed $\alpha$', color='C1')
plt.plot(nu_arr/1E9, F_arr_ff2, label=r'Fixed $\dot{M}_{\rm p}$', color='C2')
plt.errorbar(nu/1E9, Fnus, yerr=sFnus, fmt='o', color='C0', label='Data')
plt.xscale('log')
plt.yscale('log')
plt.ylim(0.001, 30000)
plt.xlabel(r'$\nu$ [GHz]', fontsize=14)
plt.ylabel(r'$F_\nu$ [mJy]', fontsize=14)
plt.legend(fontsize=12)
# make the ticks bigger
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.savefig('total_SED.pdf', bbox_inches='tight', dpi=300)
plt.show()