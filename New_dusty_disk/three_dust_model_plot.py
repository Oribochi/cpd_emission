# plot the three models together
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('tableau-colorblind10')
# now changing the stile not 10^n but 100, 1000, etc
from matplotlib.ticker import ScalarFormatter
import matplotlib
matplotlib.rcParams['axes.linewidth'] = 1. # the default is too thin
# the following should avoid having the silly-looking "10^0, 10^1, 10^2",
#  the unnecessarily complicated and "noisy" versions of "1, 10,
# 100"!
matplotlib.rcParams['axes.formatter.min_exponent'] = 5
matplotlib.rcParams['axes.formatter.limits'] = (-4,4)
matplotlib.rcParams['axes.formatter.useoffset'] = False 

# First our data with the error bars BAND 7 on 2019: 118.5 ± 16.6
nu = [97.5E9, 145005402197.7, 343.5E9, 671E9] # [nu1, nu2] are the x-axis values , 671E9
Fnus = [12e-06, 21.4e-06, 0.000121, 0.000143] # [Fnu1, Fnu2] are the y-axis values in Jy , 0.000122
sFnus = [4.7e-06, 4.1e-06, 0.000013, 0.000115] # [sigma_Fnu1, sigma_Fnu2] are the y-axis errors in Jy , 0.000090

# To mJy
nu = np.array(nu)
Fnus = np.array(Fnus)*1E3
sFnus = np.array(sFnus)*1E3

# the three models frequency and flux

nu_arr1, F_arr_ff1 = np.loadtxt('nus_fluxes_gamma.txt', unpack=True) # this is the single parameter dusty model
nu_arr2, F_arr_ff2 = np.loadtxt('nus_fluxes_zhu.txt', unpack=True) # this is the 2D model of Zhu et al. 2018
nu_arr3, F_arr_ff3 = np.loadtxt('nus_fluxes_strat.txt', unpack=True) # this is the 3D model with stratification from us

plt.figure(figsize=(4.8,3.8))
plt.errorbar(nu/1E9, Fnus, yerr=sFnus, fmt='o', color='C0', label='Data')
plt.plot(nu_arr1/1E9, F_arr_ff1, label='Gamma model', color='C1')
plt.plot(nu_arr2/1E9, F_arr_ff2, label='Zhu et al. 2018', color='C2')
plt.plot(nu_arr3/1E9, F_arr_ff3, label='Stratified model', color='C3')
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'$\nu$ [GHz]', fontsize=14)
plt.ylabel(r'$F_\nu$ [mJy]', fontsize=14)
plt.legend(fontsize=12)
# make the ticks bigger 
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.savefig('models_comparison.pdf', bbox_inches='tight', dpi=300)
plt.show()




plt.figure(figsize=(4.8,3.8))
plt.errorbar(nu/1E9, Fnus, yerr=sFnus, fmt='o', color='C0', label='Data')
# plt.plot(nu_arr1/1E9, F_arr_ff1, label='Gamma model', color='C1')
# plt.plot(nu_arr2/1E9, F_arr_ff2, label='Zhu et al. 2018', color='C2')
# plt.plot(nu_arr3/1E9, F_arr_ff3, label='Stratified model', color='C3')
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'$\nu$ [GHz]', fontsize=14)
plt.ylabel(r'$F_\nu$ [mJy]', fontsize=14)
plt.legend(fontsize=12, loc='lower right')
# make the ticks bigger 
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.savefig('data.pdf', bbox_inches='tight', dpi=300)
plt.show()


spectral_index_arr=np.zeros(len(nu)-1)
for i in range(len(nu)-1):
    spectral_index_arr[i]=np.log10(Fnus[i+1]/Fnus[i])/np.log10(nu[i+1]/nu[i])

plt.figure(figsize=(4.8,3.8))
plt.errorbar(nu/1E9, Fnus, yerr=sFnus, fmt='o', color='C0', label='Data')
plt.plot(nu/1E9, Fnus, color='C1') # ploting the slope of the data to see the spectral index
# putting each value of the spectral index in the middle of the two frequencies with alpha_{Fnu[i]}^{Fnu[i+1]}
for i in range(len(spectral_index_arr)):
    plt.text((nu[i]+nu[i+1])/2.1/1E9, (Fnus[i]+Fnus[i+1])/1.5, r'$\alpha={:.2f}$'.format(spectral_index_arr[i], i), fontsize=12, color='C1', ha='center', va='bottom')
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'$\nu$ [GHz]', fontsize=14)
plt.ylabel(r'$F_\nu$ [mJy]', fontsize=14)
plt.legend(fontsize=12, loc='lower right')
# make the ticks bigger 
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.savefig('data_spectral_index.pdf', bbox_inches='tight', dpi=300)
plt.show()