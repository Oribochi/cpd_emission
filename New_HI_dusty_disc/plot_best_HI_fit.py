# plot the best fits
import numpy as np
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

# First our data with the error bars BAND 7 on 2019: 118.5 ± 16.6
nu = [97.5E9, 145005402197.7, 343.5E9, 671E9] # [nu1, nu2] are the x-axis values , 671E9
Fnus = [12e-06, 21.4e-06, 0.000121, 0.000143] # [Fnu1, Fnu2] are the y-axis values in Jy , 0.000122
sFnus = [4.7e-06, 4.1e-06, 0.000013, 0.000115] # [sigma_Fnu1, sigma_Fnu2] are the y-axis errors in Jy , 0.000090

# To mJy
nu = np.array(nu)
Fnus = np.array(Fnus)*1E3
sFnus = np.array(sFnus)*1E3

nu_arr1, F_arr_ff1 = np.loadtxt('nus_fluxes_const_alpha.txt', unpack=True) # this is the singleparameter dusty model
nu_arr2, F_arr_ff2 = np.loadtxt('nus_fluxes_const_Mpdot.txt', unpack=True) # this is the singleparameter dusty model

plt.figure(figsize=(4.8,3.8))
plt.plot(nu_arr1/1E9, F_arr_ff1, label=r'Fixed $\alpha$', color='C1')
plt.plot(nu_arr2/1E9, F_arr_ff2, label=r'Fixed $\dot{M}_{\rm p}$', color='C2')
plt.errorbar(nu/1E9, Fnus, yerr=sFnus, fmt='o', color='C0', label='Data')
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'$\nu$ [GHz]', fontsize=14)
plt.ylabel(r'$F_\nu$ [mJy]', fontsize=14)
plt.legend(fontsize=12)
# make the ticks bigger
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.savefig('best_fit_HI.pdf', bbox_inches='tight', dpi=300)
plt.show()

# print the maximun log likelihood and spectral index
log_likelihood_const_alpha=np.load('logZ1.npy')
log_likelihood_const_Mpdot=np.load('logZ2.npy')

print('Max log likelihood for constant alpha:', np.max(log_likelihood_const_alpha))
print('Max log likelihood for constant Mpdot:', np.max(log_likelihood_const_Mpdot))

# and the flux at 671 GHz for both models
print('Flux at 671 GHz for constant alpha:', F_arr_ff1[-1])
print('Flux at 671 GHz for constant Mpdot:', F_arr_ff2[-1])