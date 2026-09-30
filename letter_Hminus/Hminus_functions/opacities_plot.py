# Here we study the model of H- emission in the disk of PDS 70c.
# we plot the opacity of H- and metals at 0.05 au for the best fit
import numpy as np
import matplotlib.pyplot as plt
from units_astro import *
from magneticCPD import alpha_calculator, T_z
from k_ff_metals import k_ff_metals
from k_ff_bf_Hminus_2021 import k_ff_Hminus, k_bf_Hminus
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
Mpdot=10**(-5.586141861325059)*Mj/yr
Bps=10**(2.397439497803699)

nu_array=np.logspace(log10(1E9), 15, 100)
# we calculate the temperature at 0.05 au
R=0.05*au
alpha=alpha_calculator(R, Mpdot, Bps)
Temp_z = T_z(0, R, Mpdot, alpha)
print('Temperature at 0.05 au:', Temp_z, 'K')
print('alpha at 0.05 au:', alpha)
# we calculate the opacity at 0.05 au
k_ff_metals_arr = np.array([k_ff_metals(Temp_z, nu_i) for nu_i in nu_array])
k_ff_Hminus_arr = np.array([k_ff_Hminus(Temp_z, nu_i) for nu_i in nu_array])
k_bf_Hminus_arr = np.array([k_bf_Hminus(Temp_z, nu_i) for nu_i in nu_array])
k_ff_bf_Hminus_arr = k_ff_Hminus_arr + k_bf_Hminus_arr
# we plot the opacities
plt.figure(figsize=(4.8, 3.8))
plt.plot(nu_array/1e9, k_ff_metals_arr, label='Metals', color='C0', lw=2)
plt.plot(nu_array/1e9, k_ff_Hminus_arr, label='H-', color='C1', lw=2)
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'$\nu$ [GHz]', fontsize=14)
plt.ylabel(r'$\kappa_{\nu}$ [$P_{\rm e}\, \rm cm^{-4}$]', fontsize=14)
plt.legend()
plt.savefig('Opacities_0.05au.png', dpi=300, bbox_inches='tight')
plt.show()

# now a zoomed version
nu_array=np.logspace(log10(1E9*60000), log10(1E9*300000), 100)
# we calculate the opacity at 0.05 au
k_ff_metals_arr = np.array([k_ff_metals(Temp_z, nu_i) for nu_i in nu_array])
k_ff_Hminus_arr = np.array([k_ff_Hminus(Temp_z, nu_i) for nu_i in nu_array])
k_bf_Hminus_arr = np.array([k_bf_Hminus(Temp_z, nu_i) for nu_i in nu_array])
k_ff_bf_Hminus_arr = k_ff_Hminus_arr + k_bf_Hminus_arr
# we plot the opacities
plt.figure(figsize=(4.8, 3.8))
plt.plot(nu_array/1e9, k_ff_metals_arr, label='Metals', color='C0', lw=2)
plt.plot(nu_array/1e9, k_ff_Hminus_arr, label='H-', color='C1', lw=2)
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'$\nu$ [GHz]', fontsize=14)
plt.ylabel(r'$\kappa_{\nu}$ [$P_{\rm e}\, \rm cm^{-4}$]', fontsize=14)
plt.legend()
plt.savefig('Opacities_0.05au_zoom.png', dpi=300, bbox_inches='tight')
plt.show()
