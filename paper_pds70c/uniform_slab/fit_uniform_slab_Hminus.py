# Using Nautilus nested sampling to obtain the best fit parameters for the uniform slab model

import numpy as np
import matplotlib.pyplot as plt
from uniform_slab import *
from nautilus import Prior
from units_astro import *
from nautilus import Sampler
import corner
from numpy import exp, log, log10, sqrt

# First our data with the error bars
nus = [97.5E9, 145005402197.7, 343.5E9, 671E9] # [nu1, nu2] are the x-axis values , 671E9
Fnus = [12e-06, 21.4e-06, 0.000121, 0.000143] # [Fnu1, Fnu2] are the y-axis values in Jy , 0.000122
sFnus = [4.7e-06, 4.1e-06, 0.000013, 0.000115] # [sigma_Fnu1, sigma_Fnu2] are the y-axis errors in Jy , 0.000090

# To mJy
nu = np.array(nus)
Fnus = np.array(Fnus)*1E3
sFnus = np.array(sFnus)*1E3


prior = Prior()
prior.add_parameter('log_T', dist=(2, 4)) # log10(K)
prior.add_parameter('log_EM', dist=(-15,20))
prior.add_parameter('log_Rmax', dist=(-15, 0))


# For log_alpha=-2
def likelihood(param_dict):
    T=10**param_dict['log_T'] # K
    EM = 10**param_dict['log_EM']*pc # cm^-6 pc
    Rmax = 10**param_dict['log_Rmax']
    
    ymodels = np.zeros(len(Fnus))
    Xis = np.zeros(len(Fnus))
    for i in range(len(Fnus)):
        ymodels[i] = F_uniform_slab(Rmax*au, T, EM, nu[i])
        Xis[i] = -0.5*((Fnus[i]-ymodels[i])/sFnus[i])**2
        
    return np.sum(Xis)

# Corro el nested sampling de nautilus
sampler = Sampler(prior, likelihood, n_live=1000, pool=16)
sampler.run(verbose=True)

# Hacemos el corner plot

points, log_w, log_l = sampler.posterior()
corner.corner(
    points, weights=np.exp(log_w), bins=20, labels=[r'$log(T)$', r'$log(EM)$', r'$log(R_{max})$'],color='purple',
    plot_datapoints=False, range=np.repeat(0.999, len(prior.keys)), fontsize=12)

# Save the log Z and the corner plot
np.save('logZ1.npy', log_l)
plt.show()

print('log(Z) =', log_l)
print('Evidence =', np.exp(log_l))
print('The mean parameters =', np.mean(points, axis=0))
print('Errors =', np.std(points, axis=0))
print('The log likelihood is', likelihood({'log_T': points[np.argmax(log_l)][0], 'log_EM': points[np.argmax(log_l)][1], 'log_Rmax': points[np.argmax(log_l)][2]}))
print('The best fit parameter are T =', 10**points[np.argmax(log_l)][0], 'K', '; EM =', 10**points[np.argmax(log_l)][1] ,'cm^-6*pc', '; Rmax=', 10**points[np.argmax(log_l)][2], 'au')
print('The best fit parameter are log(T) =', points[np.argmax(log_l)][0], 'K', '; log(EM) =', points[np.argmax(log_l)][1] ,'cm^-6*pc', '; log(Rmax)=', points[np.argmax(log_l)][2], 'au')

T=10**points[np.argmax(log_l)][0]
EM=10**points[np.argmax(log_l)][1]*pc
Rmax=10**points[np.argmax(log_l)][2]

fluxes = np.zeros(len(nu))
for i in range(len(nu)):
    fluxes[i]=F_uniform_slab(Rmax*au, T, EM, nu[i])
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
plt.show()

# to inches the column and the text width
column_width=256.0748*0.0138888889
text_width=523.5307*0.0138888889


# vs the real data and the corner in the same plot
fig = plt.figure(layout='constrained', figsize=(text_width, text_width/(2.5)), dpi=300)
subfigs = fig.subfigures(1, 2, width_ratios=[1.1, 1], wspace=0.15)

ax1 = subfigs[0].subplots(1, 1, sharey=True)
ax1.errorbar(np.array(nus)/10**9, Fnus, yerr=sFnus, fmt='o', label='Data', color='blue')
#ax1.plot(nu_arr/10**9, F_arr_ff, label='Free-free', linestyle=':', color='black')
#ax1.plot(nu_arr/10**9, F_arr_zhu, label='Dust',linestyle='--', color='green')
ax1.plot(np.array(nus)/10**9, fluxes, label='Model', alpha=0.5, color='red')
ax1.loglog()
ax1.tick_params(axis='both', which='minor', labelsize=6)
ax1.tick_params(axis='both', which='major', labelsize=6)
ax1.set_xlabel(r'$\nu$ (GHz)', fontsize=8)
ax1.set_ylabel(r'$F_{\nu}$ (mJy)', fontsize=8)
ax1.legend(loc='upper left', fontsize=8)


ax2 = subfigs[1]

corner.corner(points, weights=np.exp(log_w), bins=20, labels=[r'$log(T)$', r'$log(EM)$', r'$log(R_{max})$'],color='purple',
    plot_datapoints=False, range=np.repeat(0.999, len(prior.keys)), fig=ax2, label_kwargs={'fontsize': 8})

for axs in ax2.get_axes():
    axs.tick_params(axis='both', labelsize=6)

plt.savefig('fig_uniform_slab_Hminus.png', bbox_inches='tight',dpi=300)
plt.show()