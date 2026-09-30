# plot MHD disk
import numpy as np
import matplotlib.pyplot as plt
from flux_simple_disk import F_nu_tot
from units_astro import *
import corner
from numpy import pi, log, log10
plt.style.use('tableau-colorblind10')

# First our data with the error bars
nus = [97.5E9, 145005402197.7, 343.5E9, 671E9] # [nu1, nu2] are the x-axis values , 671E9
Fnus = [12e-06, 21.4e-06, 0.000121, 0.000143] # [Fnu1, Fnu2] are the y-axis values in Jy , 0.000122
sFnus = [4.7e-06, 4.1e-06, 0.000013, 0.000115] # [sigma_Fnu1, sigma_Fnu2] are the y-axis errors in Jy , 0.000090

# To mJy
nu = np.array(nus)
Fnus = np.array(Fnus)*1E3
sFnus = np.array(sFnus)*1E3
# array of frequencies
nu_arr=np.logspace(log10(nus[0]), log10(nus[-1])+1, 100)


Mpdot=10**(-5.596172)*Mj/yr
Bps=10**(2.81155774) # G
print('Using Mdot =', Mpdot/(Mj/yr), 'Mj/yr and Bps =', Bps, 'G')

# array of fluxes
F_arr=np.zeros(len(nu_arr))
F_arr_ff=np.zeros(len(nu_arr))
F_arr_zhu=np.zeros(len(nu_arr))

for i in range(len(nu_arr)):
    F_arr_ff[i]=F_nu_tot(nu_arr[i], Mpdot, Bps)

# save nu_array and fluxes
np.savetxt('nus_fluxes_Hminus_extended.txt', np.column_stack((nu_arr, F_arr_ff)), header='nu [GHz] F_nu [mJy]', fmt='%f %f')
