# plot MHD disk
import numpy as np
import matplotlib.pyplot as plt
from flux_gas_mag_disk import F_nu
from magneticCPD import R_T1000, R_truncation
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
nu_arr=np.logspace(log10(nus[0]), log10(nus[-1])+0.3, 100)

# Now the same but with Ṁp = 10−5 MJup yr−1 and α = 10−3
Mpdot=10**(-5.380344898689586)*Mj/yr
Bps=578.2092152281214

print('Using Mdot =', Mpdot/(Mj/yr), 'Mj/yr and Bps =', Bps, 'G')

# array of fluxes
F_arr=np.zeros(len(nu_arr))
F_arr_ff=np.zeros(len(nu_arr))
F_arr_zhu=np.zeros(len(nu_arr))

for i in range(len(nu_arr)):
    F_arr_ff[i]=F_nu(nu_arr[i], Mpdot, Bps)

# plot
plt.errorbar(nu, Fnus, yerr=sFnus, fmt='o', label='Data', color='black', capsize=5)
plt.plot(nu_arr, F_arr_ff, label='MHD Disk Model', color='C1')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Frequency (Hz)')
plt.ylabel(r'Flux Density (mJy)')
plt.title('MHD Disk Model Fit to Data')     
plt.legend()
plt.grid(True, which="both", ls="--", linewidth=0.5)
plt.tight_layout()
plt.show()

# prin the log likelihood
def log_likelihood_F(param_dict):
    Mpdot = 10**param_dict['log_Mpdot']*Mj/yr
    Bps = 10**param_dict['log_Bps']
    dust_to_gas = 0

    R1k = R_T1000(Mpdot, Bps)/au
    Rtrunc = R_truncation(Mpdot, Bps)/au

    diff = abs(Rtrunc-R1k)/R1k
    R_diff = (Rtrunc-R1k)
    # we want to make sure that the truncation radius is bigger than the 1000 radius so the ionized gas can couple the magnetic field
    ymodels = np.zeros(len(Fnus))
    Xis = np.zeros(len(Fnus))
    for i in range(len(Fnus)):
        ymodels[i] = F_nu(nu[i], Mpdot, Bps)
        Xis[i] = -0.5*((Fnus[i]-ymodels[i])/sFnus[i])**2
    # The more negative R_diff I want a bigger penalty, else just the sum of the likelihoods
    if R_diff < 0:
        return np.sum(Xis)-10*diff
    else:
        return np.sum(Xis)
    
print('Log Likelihood:', log_likelihood_F({'log_Mpdot': -5.378675601725678, 'log_Bps': 2.7565721690699867}))