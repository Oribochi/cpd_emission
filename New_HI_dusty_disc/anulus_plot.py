# making interesting plots

import matplotlib.pyplot as plt
import numpy as np
from numpy import pi, log10
from units_astro import *
from dustyCPD import T_c_approx, Rin, Rout, dis
from flux_gas_dust_simple_disk import I_nu

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

best_fit=np.loadtxt('best_fit_params_magnetic_disk_with_zeta.txt', dtype=str)
Mpdot=10**(float(best_fit[0][1]))*Mj/yr
zeta=10**(float(best_fit[3][1]))
print('zeta: '+str(zeta))    
alpha=0.01

# makin a plot of the flux per anulus
R_arr=np.logspace(log10(1.0001*Rin/au), log10(Rout/au), 100)*au
delta_R=np.diff(R_arr)
delta_R=np.append(delta_R, delta_R[-1]) # to have the same length as R_arr
nu_9=671E9 # Band 9 frequency
I_nu_arr=np.zeros(len(R_arr))
for i in range(len(R_arr)-1):
    I_nu_arr[i]=I_nu(R_arr[i], nu_9, Mpdot, alpha, dust_to_gas=zeta)*R_arr[i]*2*pi/dis**2 # to mJy

I_nu_arr2=I_nu(R_arr, nu_9, Mpdot, alpha, dust_to_gas=zeta)*2*pi*R_arr/dis**2

print('Total flux at', nu_9/1e9, 'GHz:', np.trapezoid(I_nu_arr, R_arr), 'mJy')

print('Total flux at', nu_9/1e9, 'GHz (using I_nu_arr2):', np.trapezoid(I_nu_arr2, R_arr), 'mJy')

plt.plot(R_arr/au, I_nu_arr*au, color='C0')
plt.plot(R_arr/au, I_nu_arr2*au, color='C1', linestyle='--')
plt.xlabel(r'$R$/au', fontsize=14)
plt.ylabel(r'$I_\nu \times 2\pi R \Delta R \times$1au [mJy $\rm cm^{-1}$]', fontsize=14)
plt.xscale('log')
#plt.yscale('log')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.savefig('flux_per_anulus_HII.pdf', dpi=300, bbox_inches='tight')
plt.show()

# making a plot of the central temperature
T_c_arr=np.zeros(len(R_arr))
for i in range(len(R_arr)):
    T_c_arr[i]=T_c_approx(R_arr[i], Mpdot, alpha)

plt.plot(R_arr/au, T_c_arr, color='C0')
plt.xlabel(r'$R$/au', fontsize=14)
plt.ylabel(r'$T_c$ [K]', fontsize=14)
plt.xscale('log')
#plt.yscale('log')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.savefig('T_c_strat_disc_HII.pdf', dpi=300, bbox_inches='tight')
plt.show()