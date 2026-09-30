# plot MHD disk

import numpy as np
import matplotlib.pyplot as plt
from numpy import log10
#import smplotlib
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
# array of frequencies
nu_arr=np.logspace(log10(nus[0]), log10(nus[-1]), 100)

model_dust=np.loadtxt('nus_fluxes_strat.txt')
model_ff=np.loadtxt('nus_fluxes_Hminus.txt')


# print spectral indexes
alpha_dust = log10(model_dust[-1,1]/model_dust[0,1])/log10(model_dust[-1,0]/model_dust[0,0])
#alpha_const_alpha = log10(model_const_alpha[-1,1]/model_const_alpha[0,1])/log10(model_const_alpha[-1,0]/model_const_alpha[0,0])
alpha_ff = log10(model_ff[-1,1]/model_ff[0,1])/log10(model_ff[-1,0]/model_ff[0,0])
print(f'Spectral index for dust model: {alpha_dust:.2f}')
#print(f'Spectral index for constant alpha model: {alpha_const_alpha:.2f}')
print(f'Spectral index for magnetic disc model: {alpha_ff:.2f}')
                                                                    
# Plotting
# vs the real data and the corner in the same plot
fig = plt.figure(figsize=(4.8, 3.8), dpi=300)
ax = fig.add_subplot(1,1,1)
ax.errorbar(nu/1e9, Fnus, yerr=sFnus, fmt='o', label='Data', color='C0')
ax.plot(model_dust[:,0]/1e9, model_dust[:,1], label='Dust model', color='C1')
#ax.plot(model_const_alpha[:,0]/1e9, model_const_alpha[:,1], label='Constant $\\alpha$ model', color='C2')
ax.plot(model_ff[:,0]/1e9, model_ff[:,1], label='Ionised gas model', color='C2')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('Frequency (GHz)', fontsize=14)
ax.set_ylabel('Flux Density (mJy)', fontsize=14)
#ax.set_title('MHD Disk Emission Models', fontsize=16)
ax.legend(fontsize=12)
plt.tight_layout()
plt.savefig('2_models.png', dpi=300, bbox_inches='tight')
plt.show()