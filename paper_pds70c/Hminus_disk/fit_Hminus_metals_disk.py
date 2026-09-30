# adding the H- bf and free-free radiation

# Using Nautilus nested sampling to obtain the best fit parameters for the Zhu model given by our real data

import numpy as np
import matplotlib.pyplot as plt
from flux_magnetic_disk import F_metals_Hminus
from magneticCPD import R_T1000, R_truncation, profiles
from errors import error_param
from nautilus import Prior
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


prior = Prior()
prior.add_parameter('log_Mpdot', dist=(-10, -2))
prior.add_parameter('log_Bps', dist=(-4, 4))

# Depends on Bps and Mpdot
def likelihood(param_dict):
    Mpdot = 10**param_dict['log_Mpdot']*Mj/yr
    Bps = 10**param_dict['log_Bps']
    
    R1k = R_T1000(Mpdot, Bps)/au
    Rtrunc = R_truncation(Mpdot, Bps)/au

    diff = abs(Rtrunc-R1k)/R1k
    R_diff = (Rtrunc-R1k)
    # we want to make sure that the truncation radius is bigger than the 1000 radius so the ionized gas can couple the magnetic field
    ymodels = np.zeros(len(Fnus))
    Xis = np.zeros(len(Fnus))
    for i in range(len(Fnus)):
        ymodels[i] = F_metals_Hminus(nu[i], Mpdot, Bps)
        Xis[i] = -0.5*((Fnus[i]-ymodels[i])/sFnus[i])**2
    # The more negative R_diff I want a bigger penalty, else just the sum of the likelihoods
    if R_diff < 0:
        return np.sum(Xis)-10*diff
    else:
        return np.sum(Xis)
    
# Corro el nested sampling de nautilus
sampler = Sampler(prior, likelihood, n_live=1000, pool=16)
sampler.run(verbose=True)


points, log_w, log_l = sampler.posterior()
corner.corner(
    points, weights=np.exp(log_w), bins=20, labels=[r'$\log(\frac{dM_{\rm p}/dt}{M_{\rm j}/yr})$', r'$\log(B_{\rm ps}/\rm G)$'],color='purple',
    plot_datapoints=False, range=np.repeat(0.999, len(prior.keys)), fontsize=12)

# Save the log Z and the corner plot
np.save('logZ1.npy', log_l)
np.save('logW1.npy', log_w)
np.save('logP.npy', points)
plt.savefig('corner_plot_metals_Hminus.png')
plt.show()

print('log(Z) =', log_l)
print('Evidence =', np.exp(log_l))
print('The mean parameters =', np.mean(points, axis=0))
print('Errors =', np.std(points, axis=0))
print('The maximum likelihood parameters =', points[np.argmax(log_l), :])
print('The maximum likelihood =', np.max(log_l))

# To see equal weights
points_equalw, log_w_equalw, log_l_equalw = sampler.posterior(equal_weight=True)

# The best fit parameters with ther respective errors
Mpdot_best, Bps_best = points[np.argmax(log_l)]
Mpdot_min, Mpdot_max = error_param(points_equalw[:,0], Mpdot_best)
Bps_min, Bps_max = error_param(points_equalw[:,1], Bps_best)

print('log_Mpdot =', Mpdot_best,'+', Mpdot_max, '-', -Mpdot_min, 'Mj/yr')
print('log_Bps =', Bps_best,'+', Bps_max, '-', -Bps_min, 'Gauss')

print(10**Bps_best, 'G')
print(10**(Bps_best+Bps_max), 'G')
print(10**(Bps_best+Bps_min), 'G')

# now making our spectral index best fit for the fluxes
# Mpdot = 10**(-5.6)*Mj/yr
# Bps = 10**(-10)

Mpdot = 10**points[np.argmax(log_l)][0]*Mj/yr
Bps = 10**points[np.argmax(log_l)][1]

fluxes = np.zeros(len(nu))
for i in range(len(nu)):
    fluxes[i]=F_metals_Hminus(nu[i], Mpdot, Bps)
    print('Flux at', round(nu[i]/1E9, 1), 'GHz =', fluxes[i])
    if i>0:
        print('Spectral index is ', (log(fluxes[i])-log(fluxes[i-1]))/(log(nu[i])-log(nu[i-1])))

plt.errorbar(np.array(nus)/10**9, Fnus, yerr=sFnus, fmt='o', label='Data', color='blue')
plt.plot(np.array(nus)/10**9, fluxes, label='Best fit', alpha=0.5, color='red')
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'$\nu$ [GHz]', fontsize=12)
plt.ylabel(r'$F_{\nu}$ [mJy]', fontsize=12)
plt.legend(loc='upper left')
plt.savefig('best_fit_metals_Hminus.png')
plt.show()

# array of frequencies
nu_arr=np.logspace(log10(nus[0]), log10(nus[-1]), 100)

# array of fluxes
F_arr=np.zeros(len(nu_arr))
F_arr_ff=np.zeros(len(nu_arr))
F_arr_zhu=np.zeros(len(nu_arr))

for i in range(len(nu_arr)):
    F_arr_ff[i]=F_metals_Hminus(nu_arr[i], Mpdot, Bps)

# to inches the column and the text width
column_width=256.0748*0.0138888889
text_width=523.5307*0.0138888889

# vs the real data and the corner in the same plot
fig = plt.figure(layout='constrained', figsize=(9, 3.8), dpi=300)
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


ax2 = subfigs[1]

corner.corner(points, weights=np.exp(log_w), bins=20, labels=[r'$\log\left(\dot{M_{\rm p}}/(M_{\rm Jup}/{\rm yr})\right)$', r'$\log\left(B_{\rm ps}/{\rm G}\right)$'],color='purple',
    plot_datapoints=False, range=np.repeat(0.999, len(prior.keys)), fig=ax2, labelpad=0.1)

# for axs in ax2.get_axes():
#     axs.tick_params(axis='both', labelsize=6)

plt.savefig('fig_metals_Hminus_ff_bf.png', bbox_inches='tight',dpi=300)
plt.show()
