# plot MHD disk
import numpy as np
import matplotlib.pyplot as plt
from units_astro import *
from numpy import pi, log, log10
import smplotlib
import matplotlib
matplotlib.rcParams['axes.linewidth'] = 1. # the default is too thin
# the following should avoid having the silly-looking "10^0, 10^1, 10^2",
#  the unnecessarily complicated and "noisy" versions of "1, 10,
# 100"!
matplotlib.rcParams['axes.formatter.min_exponent'] = 5
matplotlib.rcParams['axes.formatter.limits'] = (-4,4)
matplotlib.rcParams['axes.formatter.useoffset'] = False  # also usually better
plt.style.use('tableau-colorblind10')
import corner

# First our data with the error bars
nus = [97.5E9, 145005402197.7, 343.5E9, 671E9] # [nu1, nu2] are the x-axis values , 671E9
Fnus = [12e-06, 21.4e-06, 0.000121, 0.000143] # [Fnu1, Fnu2] are the y-axis values in Jy , 0.000122
sFnus = [4.7e-06, 4.1e-06, 0.000013, 0.000115] # [sigma_Fnu1, sigma_Fnu2] are the y-axis errors in Jy , 0.000090

# To mJy
nu = np.array(nus)
Fnus = np.array(Fnus)*1E3
sFnus = np.array(sFnus)*1E3
# array of frequencies

# read data
data=np.loadtxt('nus_fluxes_Hminus.txt')
nu_arr=np.array(data[:,0])
F_arr_ff=np.array(data[:,1])

fig = plt.figure(layout='constrained', figsize=(9, 3.8), dpi=300)
subfigs = fig.subfigures(1, 2, width_ratios=[1.1, 1], wspace=0.05)

ax1 = subfigs[0].subplots(1, 1, sharey=True)
ax1.errorbar(np.array(nus)/10**9, Fnus, yerr=sFnus, fmt='o', label='Data', color='C0')
#ax1.plot(nu_arr/10**9, F_arr_ff, label='Free-free', linestyle=':', color='black')
#ax1.plot(nu_arr/10**9, F_arr_zhu, label='Dust',linestyle='--', color='green')
ax1.plot(nu_arr/10**9, F_arr_ff, label='Magnetic disc', color='C1')
#ax1.plot(nu_arr/10**9, F_arr_ff2, label=r'$T=10000$ (K)', color='C2')
ax1.loglog()
# ax1.tick_params(axis='both', which='minor', labelsize=12)
# ax1.tick_params(axis='both', which='major', labelsize=12)
# ax1.tick_params(axis='both', which='both', direction='in')
ax1.set_xlabel(r'$\nu$ (GHz)')
ax1.set_ylabel(r'$F_{\nu}$ (mJy)')
ax1.legend(loc='best')


ax2 = subfigs[1]
points=np.load('logP.npy')
log_w=np.load('logW1.npy')

corner.corner(points, weights=np.exp(log_w), bins=20, labels=[r'$\log\left(\dot{M}_{\rm p}/(M_{\rm Jup}/{\rm yr})\right)$', r'$\log\left(B_{\rm ps}/{\rm G}\right)$'],color='purple',
    plot_datapoints=False, range=np.repeat(0.999, 2), fig=ax2, labelpad=0.1) #, label_kwargs={'fontsize': 12})

# for axs in ax2.get_axes():
#     axs.tick_params(axis='both', labelsize=12)
    

plt.savefig('fig_mag_disk_big.png', bbox_inches='tight',dpi=300)
plt.show()


plt.figure(figsize=(4.8,3.8))
plt.errorbar(nu/1E9, Fnus, yerr=sFnus, fmt='o', color='k', label='Data', markersize=5, capsize=3)
plt.plot(nu_arr/1E9, F_arr_ff, label='This work', color='C1')
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'$\nu$ [GHz]', fontsize=14)
plt.ylabel(r'$F_{\nu}$ [mJy]', fontsize=14)
plt.ylim(1E-2, 1)
plt.show()


fig = plt.figure(layout='constrained', figsize=(9, 3.8), dpi=300)
subfigs = fig.subfigures(1, 2, width_ratios=[1.1, 1], wspace=0.05)

ax1 = subfigs[0].subplots(1, 1, sharey=True)
ax1.errorbar(np.array(nus)/10**9, Fnus, yerr=sFnus, fmt='o', label='Data', color='C0')
#ax1.plot(nu_arr/10**9, F_arr_ff, label='Free-free', linestyle=':', color='black')
#ax1.plot(nu_arr/10**9, F_arr_zhu, label='Dust',linestyle='--', color='green')
ax1.plot(nu_arr/10**9, F_arr_ff, label=r'$T=1000$ (K)', color='C1')
#ax1.plot(nu_arr/10**9, F_arr_ff2, label=r'$T=10000$ (K)', color='C2')
ax1.loglog()
# ax1.tick_params(axis='both', which='minor', labelsize=12)
# ax1.tick_params(axis='both', which='major', labelsize=12)
# ax1.tick_params(axis='both', which='both', direction='in')
ax1.set_xlabel(r'$\nu$ (GHz)')
ax1.set_ylabel(r'$F_{\nu}$ (mJy)')
ax1.legend(loc='upper left')


ax2 = subfigs[1]

corner.corner(
    points2, weights=np.exp(log_w2), bins=20, labels=[r'$\log \rm EM/(\rm cm^{-6} pc)$', r'$\log (R_{\rm max}/\rm au)$'], color='purple',
    plot_datapoints=False, range=np.repeat(0.999, len(prior.keys)), fig=ax2, labelpad=0.1) #, label_kwargs={'fontsize': 12})

# for axs in ax2.get_axes():
#     axs.tick_params(axis='both', labelsize=12)
    

plt.savefig('fig_uniform_slab_1000K.png', bbox_inches='tight',dpi=300)
plt.show()