# Plot best fit magnetic disk
from flux_dust_gas_disk import F_nu_tot as F
import matplotlib.pyplot as plt
import numpy as np
from units_astro import *

# First our data with the error bars
nus = [97.5E9, 145005402197.7, 343.5E9, 671E9] # [nu1, nu2] are the x-axis values , 671E9
Fnus = [12e-06, 21.4e-06, 0.000121, 0.000143] # [Fnu1, Fnu2] are the y-axis values in Jy , 0.000122
sFnus = [4.7e-06, 4.1e-06, 0.000013, 0.000115] # [sigma_Fnu1, sigma_Fnu2] are the y-axis errors in Jy , 0.000090

# To mJy
nu = np.array(nus)
Fnus = np.array(Fnus)*1E3
sFnus = np.array(sFnus)*1E3

Mpdot=10**(-5.65)*Mj/yr
Bps=600

F_model = F(nu, Mpdot, Bps, zeta=1e-7)
plt.errorbar(nu/1E9, Fnus, yerr=sFnus, fmt='o', label='Observed Data', color='blue')
plt.plot(nu/1E9, F_model, label='Model Prediction', color='red')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Frequency (GHz)')
plt.ylabel('Flux Density (mJy)')
plt.title('Flux Density vs Frequency for CPD around PDS 70 c')
plt.legend()
plt.grid(True, which="both", ls="--")
plt.show()