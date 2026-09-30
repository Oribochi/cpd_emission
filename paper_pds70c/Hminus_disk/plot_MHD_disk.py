# plot MHD disk

import numpy as np
import matplotlib.pyplot as plt
from flux_magnetic_disk import F_metals_Hminus
from units_astro import *
from nautilus import Sampler
import corner
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
# array of frequencies
nu_arr=np.logspace(log10(nus[0]), log10(nus[-1]), 100)



points=np.load('logP.npy')
log_w=np.load('logW1.npy')
log_l=np.load('logZ1.npy')

Mpdot = 10**points[np.argmax(log_l)][0]*Mj/yr
Bps = 10**points[np.argmax(log_l)][1]

# array of fluxes
F_arr=np.zeros(len(nu_arr))
F_arr_ff=np.zeros(len(nu_arr))
F_arr_zhu=np.zeros(len(nu_arr))

for i in range(len(nu_arr)):
    F_arr_ff[i]=F_metals_Hminus(nu_arr[i], Mpdot, Bps)

# save nu_array and fluxes
np.savetxt('nus_fluxes_Hminus.txt', np.column_stack((nu_arr, F_arr_ff)), header='nu [GHz] F_nu [mJy]', fmt='%f %f')

# vs the real data and the corner in the same plot
fig = plt.figure(figsize=(9, 3.8), dpi=300)
subfigs = fig.subfigures(1, 2, width_ratios=[1.1, 1], wspace=0.05)

ax1 = subfigs[0].subplots(1, 1, sharey=True)
ax1.errorbar(np.array(nus)/10**9, Fnus, yerr=sFnus, fmt='o', label='Data', color='C0')
#ax1.plot(nu_arr/10**9, F_arr_ff, label='Free-free', linestyle=':', color='black')
#ax1.plot(nu_arr/10**9, F_arr_zhu, label='Dust',linestyle='--', color='green')
ax1.plot(nu_arr/10**9, F_arr_ff, label='Magnetic disk', color='C1')
ax1.loglog()
ax1.set_xlabel(r'$\nu$ (GHz)')
ax1.set_ylabel(r'$F_{\nu}$ (mJy)')
ax1.legend(loc='upper left')


print('Best fit Mpdot:', Mpdot/Mj*yr, 'Mj/yr')
print('Best fit Bps:', Bps, 'G')

ax2 = subfigs[1]

corner.corner(points, weights=np.exp(log_w), bins=20, labels=[r'$\log\left(\dot{M}_{\rm p}/(M_{\rm Jup}/{\rm yr})\right)$', r'$\log\left(B_{\rm ps}/{\rm G}\right)$'],color='purple',
    plot_datapoints=False, range=np.repeat(0.999, 2), fig=ax2, labelpad=0.1)

plt.savefig('fig_metals_Hminus_ff_bf_big.png', bbox_inches='tight',dpi=300)
plt.show()
# for axs in ax2.get_axes():
#     axs.tick_params(axis='both', labelsize=6)
# now the new figure
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

# now making the plot in the upper right corner bigger
# Main figure and grid
fig = plt.figure(figsize=(5.5, 5.5))
gs = GridSpec(3, 3, figure=fig)

# Create nested 2x2 GridSpec for the corner plot
corner_spec = GridSpecFromSubplotSpec(2, 2, subplot_spec=gs[1:, :2])

# Allocate all 2x2 axes, even if some stay unused
corner_axes = np.empty((2, 2), dtype=object)
for i in range(2):
    for j in range(2):
        corner_axes[i, j] = fig.add_subplot(corner_spec[i, j])

# Flatten and pass to corner
corner.corner(points, weights=np.exp(log_w), fig=fig, color='C1', bins=20, range=np.repeat(0.999, 2),
              labels=[r'$\log\left(\dot{M}_{\rm p}/(M_{\rm Jup}/{\rm yr})\right)$', r'$\log\left(B_{\rm ps}/{\rm G}\right)$'],plot_datapoints=False, labelpad=0.05,
              axes=corner_axes.flatten().tolist())  # Must be a flat list)

yerr_log = np.array([[log10(Fnus[i]) - log10(Fnus[i] - sFnus[i]) for i in range(len(Fnus))],[log10(Fnus[i] + sFnus[i])- log10(Fnus[i]) for i in range(len(Fnus))]])

# Another plot in top-right 2 rows of last column
ax_function = fig.add_subplot(gs[:2, 1:])
ax_function.errorbar(np.array(nus)/1E9, log10(Fnus), yerr=yerr_log, fmt='o', color='C0', markersize=3, label='Data')
ax_function.plot(nu_arr/10**9, log10(F_arr_ff), label='Magnetic disk', color='C1')
ax_function.set_xscale('log')
ax_function.set_xlabel(r'$\nu/{\rm GHz}$', labelpad=5, fontsize=15)
ax_function.set_ylabel(r'$\log(F_{\nu}/ {\rm mJy})$', labelpad=5, fontsize=15)
ax_function.tick_params(bottom=False, top=True, labelbottom=False, labeltop=True, labelsize=15)
ax_function.tick_params(axis='both', which='both', labelsize=15)
ax_function.xaxis.tick_top()
ax_function.yaxis.tick_right()

ax_function.tick_params(axis='y', direction='in', pad=0, labelsize=15)
# ax_function.set_yticks([-5,-3, -1, 1])
# ax_function.get_yaxis().set_major_formatter(plt.ScalarFormatter())
# ax_function.get_yaxis().set_minor_formatter(plt.NullFormatter())
# Remove bottom and left ticks
ax_function.set_xticks([100, 300, 700])
ax_function.get_xaxis().set_major_formatter(plt.ScalarFormatter())
ax_function.get_xaxis().set_minor_formatter(plt.NullFormatter())
# Custom ticks (ensures you get what you want)
ax_function.legend(loc='upper left', fontsize=15)
ax_function.xaxis.set_label_position('top') 
ax_function.yaxis.set_label_position('right')
# now the bottom left corner
corner_axes[1, 0].set_xticks([-5.6,-5.5,-5.4,-5.3])
corner_axes[1, 0].get_xaxis().set_major_formatter(plt.ScalarFormatter())
corner_axes[1, 0].get_xaxis().set_minor_formatter(plt.NullFormatter())

# Hide or use bottom-right slot
ax_unused = fig.add_subplot(gs[2, 2])
ax_unused.axis('off')
plt.savefig('fig_metals_Hminus_ff_bf.png', bbox_inches='tight',dpi=300)
plt.show()