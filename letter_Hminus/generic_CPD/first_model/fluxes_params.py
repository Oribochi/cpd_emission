# Making different fluxes arrays for diferent parameters
# plot different MHD disk fluxes and compare with data
import numpy as np
import matplotlib.pyplot as plt
from flux_magnetic_disk import F_nu_mag, F_nu_dusty
from flux_dust_gas_disk import F_nu_tot
from dustyCPD import Tfloor
from units_astro import *
import corner
from numpy import pi, log, log10
plt.style.use('tableau-colorblind10')

# # First our data with the error bars
nus = [97.5E9, 145.0E9, 343.5E9, 671E9] # [nu1, nu2] are the x-axis values , 671E9

nu = np.array(nus)

# Defining a function that for a given set of parameters returns the free-free gas and dusty fluxes at the given frequencies
def fluxes_model(Mpdot, Bps, dust_to_gas_ratio):
    tau_free_free = np.zeros(len(nu)) # we use the global nu array
    tau_dusty = np.zeros(len(nu))
    flux_tot = np.zeros(len(nu))
    for i in range(len(nu)):
        tau_free_free[i] = F_nu_mag(nu[i], Mpdot, Bps)
        tau_dusty[i] = F_nu_dusty(nu[i], Mpdot, Bps, TISM=Tfloor, zeta=dust_to_gas_ratio)
        flux_tot[i] = F_nu_tot(nu[i], Mpdot, Bps, zeta=dust_to_gas_ratio)
    
    return tau_free_free, tau_dusty, flux_tot