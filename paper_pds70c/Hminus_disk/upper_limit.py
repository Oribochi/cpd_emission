# upper limit for Mpdt using a fixed Bps

import numpy as np
import matplotlib.pyplot as plt
from flux_magnetic_disk import F_metals_Hminus
from magneticCPD import R_T1000, R_truncation, profiles
from errors import error_param
from units_astro import *
from numpy import pi, log, log10
import smplotlib
import matplotlib
matplotlib.rcParams['axes.linewidth'] = 1. # the default is too thin

# the following should avoid having the silly-looking "10^0, 10^1, 10^2",
#  the unnecessarily complicated and "noisy" versions of "1, 10, 100"!
matplotlib.rcParams['axes.formatter.min_exponent'] = 5
matplotlib.rcParams['axes.formatter.limits'] = (-4,4)
matplotlib.rcParams['axes.formatter.useoffset'] = False  # also usually better
plt.style.use('tableau-colorblind10')

# First our data with the error bars
nus = [97.5E9, 145005402197.7, 343.5E9, 671E9] # [nu1, nu2] are the x-axis values , 671E9
Fnus = [12e-06, 21.4e-06, 0.000121, 0.000143] # [Fnu1, Fnu2] are the y-axis values in Jy , 0.000122
sFnus = [4.7e-06, 4.1e-06, 0.000013, 0.000115] # [sigma_Fnu1, sigma_Fnu2] are the y-axis errors in Jy , 0.000090

# To mJy
nu = np.array(nus)
Fnus = np.array(Fnus)*1E3
sFnus = np.array(sFnus)*1E3

Bps=610
Mpdot=np.logspace(-5.6,-5.3,100)*Mj/yr

flux_upper=3*sFnus[2]  # 3 sigma upper limit for the flux at 343.5 GHz

for i in range(len(Mpdot)):
    flux = F_metals_Hminus(nu[2], Mpdot[i], Bps)
    if flux > flux_upper:
        print('Upper limit for Mpdot is:', Mpdot[i]/Mj*yr, 'Mj/yr')
        print(log10(Mpdot[i]/Mj*yr), 'log10(Mpdot/Mj/yr)')
        break

# F_nu1234 = np.zeros((len(nu), len(Mpdot)))
# for i in range(len(Mpdot)):
#     for j in range(len(nu)):
#         F_nu1234[j, i] = F_metals_Hminus(nu[j], Mpdot[i], Bps=Bps)


# plt.plot(Mpdot/Mj*yr, F_nu1234[0, :], label=r'$97.5$ GHz', color='C0')
# plt.plot(Mpdot/Mj*yr, F_nu1234[1, :], label=r'$145$ GHz', color='C1')
# plt.plot(Mpdot/Mj*yr, F_nu1234[2, :], label=r'$343.5$ GHz', color='C2')
# plt.plot(Mpdot/Mj*yr, F_nu1234[3, :], label=r'$671$ GHz', color='C3')
# plt.loglog()
# # and also the 3 sigma error for each frequency
# for i in range(len(nu)):
#     plt.fill_between(Mpdot/Mj*yr, 0,  3*sFnus[i], alpha=0.2, color=f'C{i}', label=f'{nu[i]/1E9:.1f} GHz 3$\sigma$')

# plt.show()
    
         