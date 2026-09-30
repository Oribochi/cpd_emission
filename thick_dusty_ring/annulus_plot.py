# making interesting plots

import matplotlib.pyplot as plt
import numpy as np
from numpy import pi, log10
from units_astro import *
from dustyCPD import T_c_approx, Rin, Rout, dis
from flux_dusty_disk import I_nu_dust

plt.style.use('tableau-colorblind10')
import matplotlib
matplotlib.rcParams['axes.linewidth'] = 1. # the default is too thin
# the following should avoid having the silly-looking "10^0, 10^1, 10^2",
#  the unnecessarily complicated and "noisy" versions of "1, 10,
# 100"!
matplotlib.rcParams['axes.formatter.min_exponent'] = 5
matplotlib.rcParams['axes.formatter.limits'] = (-4,4)
matplotlib.rcParams['axes.formatter.useoffset'] = False 


# best fit parameters for the stratified model


alpha=10**(-5.681669159825457)
Mpdot=10**(-9.740569562462973)*Mj/yr


# # makin a plot of the flux per anulus
# R_arr=np.linspace(1.0001*Rin, Rout, 500)
nu_9=671E9 # Band 9 frequency
# delta_R=R_arr[1]-R_arr[0]
# makin a plot of the flux per anulus
R_arr=np.logspace(log10(1.0001*Rin/au), log10(Rout/au), 200)*au
delta_R=np.diff(R_arr)
delta_R=np.append(delta_R, delta_R[-1]) # to have the same length as R_arr

I_nu_arr=np.zeros(len(R_arr))
for i in range(len(R_arr)):
    I_nu_arr[i]=I_nu_dust(R_arr[i], nu_9, Mpdot, alpha)*R_arr[i]*2*pi*delta_R[i]/dis**2 # to mJy

# print the total flux
print('Total flux at', nu_9/1e9, 'GHz:', np.trapezoid(I_nu_arr/delta_R, R_arr), 'mJy')

plt.plot(R_arr/au, I_nu_arr/delta_R*au, color='C0')
plt.xlabel(r'R/au', fontsize=14)
plt.ylabel(r'$I_\nu \times 2\pi R \times$1au [mJy]', fontsize=14)
plt.xscale('log')
#plt.yscale('log')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.savefig('flux_per_anulus.pdf', dpi=300, bbox_inches='tight')
plt.show()

# making a plot of the central temperature
T_c_arr=np.zeros(len(R_arr))
for i in range(len(R_arr)):
    T_c_arr[i]=T_c_approx(R_arr[i], Mpdot, alpha)

plt.plot(R_arr/au, T_c_arr, color='C0')
plt.xlabel(r'R/au', fontsize=14)
plt.ylabel(r'$T_c$ [K]', fontsize=14)
plt.xscale('log')
#plt.yscale('log')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.savefig('T_c_strat_disc.pdf', dpi=300, bbox_inches='tight')
plt.show()
