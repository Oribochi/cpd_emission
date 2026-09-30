# plot MHD disk
from matplotlib.gridspec import GridSpec
import numpy as np
import matplotlib.pyplot as plt
from flux_gas_dust_mag_disk import F_nu as F
from units_astro import *
from nautilus import Sampler
import corner
from numpy import log10
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


Mpdot=10**(-5.607369895579998)*Mj/yr
Bps=10**(2.796625406817652) # G
dust_to_gas=10**(-10.067572749115545)

print('Using Mdot =', Mpdot/(Mj/yr), 'Mj/yr, Bps =', Bps, 'G', 'and zeta =', dust_to_gas)
# and Bps with errors
print(Bps, '+', 10**(2.796625406817652+0.07537459318234774)-Bps, '-', Bps-10**(2.796625406817652-0.37262540681765177), 'G')

F97=F(nu[0], Mpdot, Bps, zeta=dust_to_gas)
F145=F(nu[1], Mpdot, Bps, zeta=dust_to_gas)
F343=F(nu[2], Mpdot, Bps, zeta=dust_to_gas)
F671=F(nu[3], Mpdot, Bps, zeta=dust_to_gas)

# print the loglikelihood of the data given the model
logL=-0.5*np.sum(((Fnus-np.array([F97, F145, F343, F671]))/sFnus)**2)
print('Loglikelihood of the data given the model:', logL)
# the reduce chi-squared is chi/(4-3)
chi2=np.sum(((Fnus-np.array([F97, F145, F343, F671]))/sFnus)**2)
print('Reduced chi-squared:', chi2/(len(Fnus)-3))

# plot the spectral index of the model 
print('Spectral index between 97 and 145 GHz:', log10(F145/F97)/log10(nu[1]/nu[0]))
print('Spectral index between 145 and 343 GHz:', log10(F343/F145)/log10(nu[2]/nu[1]))
print('Spectral index between 343 and 671 GHz:', log10(F671/F343)/log10(nu[3]/nu[2]))

# and the total
print('Spectral index between 97 and 671 GHz:', log10(F671/F97)/log10(nu[3]/nu[0]))

# print estimated flux band 9

print('Estimated flux at 671 GHz:', F671, 'mJy')

# array of fluxes
F_arr=np.zeros(len(nu_arr))
F_arr_ff=np.zeros(len(nu_arr))
F_arr_zhu=np.zeros(len(nu_arr))

for i in range(len(nu_arr)):
    F_arr_ff[i]=F(nu_arr[i], Mpdot, Bps, zeta=dust_to_gas)

# save nu_array and fluxes
np.savetxt('nus_fluxes_Hminus.txt', np.column_stack((nu_arr, F_arr_ff)), header='nu [GHz] F_nu [mJy]', fmt='%f %f')

# vs the real data and the corner in the same plot

#plot the corner plot and the fluxes in the same figure

from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec


points=np.load('logP2.npy')
log_w=np.load('logW2.npy')

fig = plt.figure(figsize=(6.5, 6.5))
gs = GridSpec(4, 4, figure=fig)
labels=[r'$\log \left( \frac{dM_{\rm p}/dt}{M_{\rm Jup}/{\rm yr}} \right)$', r'$\log \left( \frac{B_{\rm ps}}{\rm G} \right)$', r'$\log \left( \zeta \right)$']

# Create nested 3x3
corner_spec = GridSpecFromSubplotSpec(3, 3, subplot_spec=gs[:3, :3])

# Allocate all 3x3 axes, even if some stay unused
corner_axes = np.empty((3, 3), dtype=object)
for i in range(3):
    for j in range(3):
        corner_axes[i, j] = fig.add_subplot(corner_spec[i, j])

# Flatten and pass to corner
corner.corner(points, weights=np.exp(log_w), fig=fig, color='C1', bins=20, range=np.repeat(0.999, 3),
                labels=labels, plot_datapoints=False, labelpad=0.05,
                axes=corner_axes.flatten().tolist(), label_kwargs={'fontsize': 14})  # Must be a flat list)


#corner.corner(points, weights=np.exp(log_w), bins=20, labels=[r'$\log\left(\dot{M}_{\rm p}/(M_{\rm Jup}/{\rm yr})\right)$', r'$\log\left(B_{\rm ps}/{\rm G}\right)$', r'$\log\left(\zeta\right)$'],color='purple',
#     plot_datapoints=False, range=np.repeat(0.999, 3), fig=fig, labelpad=0.1)

yerr_log = np.array([[log10(Fnus[i]) - log10(Fnus[i] - sFnus[i]) for i in range(len(Fnus))],[log10(Fnus[i] + sFnus[i]) - log10(Fnus[i]) for i in range(len(Fnus))]])

# Another plot in top-right 2 rows of last column
ax_function = fig.add_subplot(gs[0:2, 2:4])
ax_function.errorbar(np.array(nus)/1E9, log10(Fnus), yerr=yerr_log, fmt='o', color='C0', markersize=5, label='Data')
ax_function.plot(nu_arr/10**9, log10(F_arr_ff), label='Magnetic disc', color='C1')
ax_function.set_xscale('log')
ax_function.set_xlabel(r'$\nu/{\rm GHz}$', labelpad=5, fontsize=14)
ax_function.set_ylabel(r'$\log(F_{\nu}/ {\rm mJy})$', labelpad=5, fontsize=14)
ax_function.tick_params(bottom=False, top=True, labelbottom=False, labeltop=True, labelsize=12)
ax_function.tick_params(axis='both', which='both', labelsize=12)
ax_function.xaxis.tick_top()
ax_function.yaxis.tick_right()

ax_function.tick_params(axis='y', direction='in', pad=0, labelsize=12)
# ax_function.set_yticks([-5,-3, -1, 1])
# ax_function.get_yaxis().set_major_formatter(plt.ScalarFormatter())
# ax_function.get_yaxis().set_minor_formatter(plt.NullFormatter())
ax_function.set_xticks([100, 300, 700])
# Remove bottom and left ticks
ax_function.get_xaxis().set_major_formatter(plt.ScalarFormatter())
ax_function.get_xaxis().set_minor_formatter(plt.NullFormatter())
# Custom ticks (ensures you get what you want)
ax_function.legend(loc='best', fontsize=12)
ax_function.xaxis.set_label_position('top') 
ax_function.yaxis.set_label_position('right')
# now the bottom left corner
# corner_axes[1, 0].set_xticks([-5.8,-5.7,-5.6,-5.5])
# corner_axes[1, 0].tick_params(axis='both',labelsize=12)
# corner_axes[1, 1].tick_params(axis='x', labelsize=12)
# corner_axes[2, 0].tick_params(axis='x', labelsize=12)
# corner_axes[0, 2].tick_params(axis='y', labelsize=12)
#corner_axes[1, 0].get_xaxis().set_major_formatter(plt.ScalarFormatter())
#corner_axes[1, 0].get_xaxis().set_minor_formatter(plt.NullFormatter())
# the bottom left corner 
#corner_axes[1, 0].tick_params(axis='x', which='both',
#                              bottom=False, top=False,
#                              labelbottom=False, labeltop=False)

#corner_axes[2, 0].get_xaxis().set_major_formatter(plt.ScalarFormatter())
#corner_axes[2, 0].get_xaxis().set_minor_formatter(plt.NullFormatter())


# Hide or use bottom-right slot
ax_unused = fig.add_subplot(gs[3, 3])
ax_unused.axis('off')
plt.savefig('cornerplot_mag_disc2.pdf', bbox_inches='tight', dpi=300)
plt.show()
# fig = plt.figure(figsize=(9, 3.8), dpi=300)
# subfigs = fig.subfigures(1, 2, width_ratios=[1.1, 1], wspace=0.00)

# ax1 = subfigs[0].subplots(1, 1, sharey=True)
# ax1.errorbar(np.array(nus)/10**9, Fnus, yerr=sFnus, fmt='o', label='Data', color='C0')
# #ax1.plot(nu_arr/10**9, F_arr_ff, label='Free-free', linestyle=':', color='black')
# #ax1.plot(nu_arr/10**9, F_arr_zhu, label='Dust',linestyle='--', color='green')
# ax1.plot(nu_arr/10**9, F_arr_ff, label='Magnetic disk', color='C1')
# ax1.loglog()
# ax1.set_xlabel(r'$\nu$ (GHz)')
# ax1.set_ylabel(r'$F_{\nu}$ (mJy)')
# ax1.legend(loc='upper left')

# points=np.load('logP2.npy')
# log_w=np.load('logW2.npy')
# ax2 = subfigs[1]

# corner.corner(points, weights=np.exp(log_w), bins=20, labels=[r'$\log\left(\dot{M}_{\rm p}/(M_{\rm Jup}/{\rm yr})\right)$', r'$\log\left(B_{\rm ps}/{\rm G}\right)$', r'$\log\left(\zeta\right)$'],color='purple',
#     plot_datapoints=False, range=np.repeat(0.999, 3), fig=ax2, labelpad=0.1)
# plt.savefig('fig_dust_gas_mag_disc_with_zeta.pdf', bbox_inches='tight',dpi=300)
# plt.show()
