# plot different MHD disk fluxes and compare with data
import numpy as np
import matplotlib.pyplot as plt
from flux_magnetic_disk import F_nu_tot, F_nu_mag, F_nu_dusty
from dustyCPD import Tfloor
from units_astro import *
import corner
from numpy import pi, log, log10
plt.style.use('tableau-colorblind10')

# # First our data with the error bars
nus = [97.5E9, 145005402197.7, 343.5E9, 671E9] # [nu1, nu2] are the x-axis values , 671E9
Fnus = [12e-06, 21.4e-06, 0.000121, 0.000143] # [Fnu1, Fnu2] are the y-axis values in Jy , 0.000122
sFnus = [4.7e-06, 4.1e-06, 0.000013, 0.000115] # [sigma_Fnu1, sigma_Fnu2] are the y-axis errors in Jy , 0.000090

# To mJy
nu = np.array(nus)
Fnus = np.array(Fnus)*1E3
sFnus = np.array(sFnus)*1E3

# Using the following accretion rate, magnetic field force and dust to gas ratio
Mpdot=10**(-5)*Mj/yr
Bps=600
dust_to_gas_ratio=1e-2

fluxes = np.zeros(len(nu))
for i in range(len(nu)):
    fluxes[i]=F_nu_tot(nu[i], Mpdot, Bps, TISM=Tfloor, zeta=dust_to_gas_ratio)
    # print also each component of the flux
    print('Magnetic flux :',F_nu_mag(nu[i], Mpdot, Bps), '; Dusty flux :', F_nu_dusty(nu[i], Mpdot, Bps, TISM=Tfloor, zeta=dust_to_gas_ratio))
    print('Flux at', round(nu[i]/1E9, 1), 'GHz =', fluxes[i], 'mJy')
    if i>0:
        print('Spectral index is ', (log10(fluxes[i])-log10(fluxes[i-1]))/(log10(nu[i])-log10(nu[i-1])))

plt.errorbar(np.array(nus)/10**9, Fnus, yerr=sFnus, fmt='o', label='Data', color='blue')
plt.plot(np.array(nus)/10**9, fluxes, label='model', alpha=0.5, color='red')
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'$\nu$ [GHz]', fontsize=12)
plt.ylabel(r'$F_{\nu}$ [mJy]', fontsize=12)
plt.legend(loc='upper left')
plt.show()