# making interesting plots

import matplotlib.pyplot as plt
import numpy as np
from numpy import pi
from units_astro import *
from dustyCPD import T_c_approx, Rin, Rout, dis
from magneticCPD import alpha_calculator
from flux_gas_mag_disk import I_nu 

plt.style.use('tableau-colorblind10')
import matplotlib
matplotlib.rcParams['axes.linewidth'] = 1. # the default is too thin
# the following should avoid having the silly-looking "10^0, 10^1, 10^2",
#  the unnecessarily complicated and "noisy" versions of "1, 10,
# 100"!
matplotlib.rcParams['axes.formatter.min_exponent'] = 5
matplotlib.rcParams['axes.formatter.limits'] = (-4,4)
matplotlib.rcParams['axes.formatter.useoffset'] = False 


# best fit parameters for the stratified model magnetic atomic plasma disc

Mpdot=10**(-5.380344898689586)*Mj/yr
Bps=578.2092152281214

# makin a plot of the flux per anulus
R_arr=np.linspace(1.0001*Rin, Rout, 500)
nu_9=671E9 # Band 9 frequency
delta_R=R_arr[1]-R_arr[0]
I_nu_arr=np.zeros(len(R_arr))
for i in range(len(R_arr)):
    alpha_R=alpha_calculator(R_arr[i], Mpdot, Bps)
    I_nu_arr[i]=I_nu(R_arr[i], nu_9, Mpdot, alpha_R)*R_arr[i]*2*pi*delta_R/dis**2 # to mJy

# print the total flux
print('Total flux at', nu_9/1e9, 'GHz:', np.trapezoid(I_nu_arr, R_arr/delta_R), 'mJy')

plt.plot(R_arr/delta_R, I_nu_arr, color='C0')
plt.xlabel(r'R/$\Delta R$', fontsize=14)
plt.ylabel(r'$I_\nu \times 2\pi R \Delta R$ [mJy]', fontsize=14)
#plt.xscale('log')
#plt.yscale('log')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.savefig('flux_per_anulus_mag.pdf', dpi=300, bbox_inches='tight')
plt.show()

# making a plot of the central temperature
T_c_arr=np.zeros(len(R_arr))
for i in range(len(R_arr)):
    alpha_R=alpha_calculator(R_arr[i], Mpdot, Bps)
    T_c_arr[i]=T_c_approx(R_arr[i], Mpdot, alpha_R)

plt.plot(R_arr/delta_R, T_c_arr, color='C0')
plt.xlabel(r'R/$\Delta R$', fontsize=14)
plt.ylabel(r'$T_c$ [K]', fontsize=14)
#plt.xscale('log')
#plt.yscale('log')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.savefig('T_c_strat_disc_mag.pdf', dpi=300, bbox_inches='tight')
plt.show()