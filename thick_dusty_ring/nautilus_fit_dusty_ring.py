# best fit dusty disk for the constant alpha case
from numpy import log10
from flux_ring import F
import matplotlib.pyplot as plt
import numpy as np
from units_astro import *
from nautilus import Prior, Sampler
from errors import error_param, upper_limit, strict_upper_limit
import corner

# First our data with the error bars BAND 7 on 2019: 118.5 ± 16.6
nu = [97.5E9, 145005402197.7, 343.5E9, 671E9] # [nu1, nu2] are the x-axis values , 671E9
Fnus = [12e-06, 21.4e-06, 0.000121, 0.000143] # [Fnu1, Fnu2] are the y-axis values in Jy , 0.000122
sFnus = [4.7e-06, 4.1e-06, 0.000013, 0.000115] # [sigma_Fnu1, sigma_Fnu2] are the y-axis errors in Jy , 0.000090

# To mJy
nu = np.array(nu)
Fnus = np.array(Fnus)*1E3
sFnus = np.array(sFnus)*1E3

prior = Prior()
prior.add_parameter('log_Sigma', dist=(-5, 5))
prior.add_parameter('log_width', dist=(-2, 0))

T_floor = 50 # K

def log_likelihood_F(param_dict):
    Sigma = 10**param_dict['log_Sigma'] # in g/cm^2
    width = 10**param_dict['log_width']*au # in au

    ymodels = np.zeros(len(Fnus))
    Xis = np.zeros(len(Fnus))
    for i in range(len(Fnus)):
        ymodels[i] = F(nu[i], Sigma, 1*au, width, T_floor, 112.3*pc)
        Xis[i] = -0.5*((Fnus[i]-ymodels[i])/sFnus[i])**2

    # ejemplo
    # the likehood is the product of the likelihoods for each data point
    # we assume the errors are gaussian
    # L1 = norm.pdf(Fnus[0]-ymodel1, 0, sFnus[0])
    # 
    # 
    # L2 = norm.pdf(Fnus[1]-ymodel2, 0, sFnus[1])
    return np.sum(Xis)

sampler_zhu = Sampler(prior, log_likelihood_F, n_live=1000, pool = 16)             
sampler_zhu.run(verbose=True)

points, log_w, log_l = sampler_zhu.posterior()
corner.corner(
    points, weights=np.exp(log_w), bins=20, labels=[r'$\log\left( \frac{\Sigma}{\rm g~cm^{-2}} \right)$', r'$\log \left( \frac{\rm width}{\rm au} \right)$'], color='purple',
    plot_datapoints=False, range=np.repeat(0.999, len(prior.keys)), fontsize=12)

# Save the log Z and the corner plot
np.save('logZ1.npy', log_l)
np.save('logW1.npy', log_w)
np.save('logP1.npy', points)

plt.show()

# To see equal weights
points_equalw, log_w_equalw, log_l_equalw = sampler_zhu.posterior(equal_weight=True)

# now making our errors
Sigma_maxl, width_maxl = points[np.argmax(log_l)]
Sigma_min, Sigma_max = error_param(points_equalw[:,0], Sigma_maxl, bin_inf=-5, bin_sup=5)
width_min, width_max = error_param(points_equalw[:,1], width_maxl, bin_inf=-2, bin_sup=0)

width_upper = upper_limit(points_equalw[:,1], -1)
width_upper_strict = strict_upper_limit(points_equalw[:,1])

print('Sigma =', Sigma_maxl, Sigma_min, Sigma_max)
print('width =',  width_maxl, width_min, width_max)
print('width upper limit =', width_upper)
print('width = ', width_maxl, width_upper-width_maxl)
print('width upper strict limit =', width_upper_strict)
print('log_likelihood max =', np.max(log_l))

# vs the real data
# now making our spectral index
Sigma=10**points[np.argmax(log_l)][0]
width=10**points[np.argmax(log_l)][1]*au

# now save the fluxes for the simplified model best fit
nu_arr = np.linspace(nu[0], 2*nu[-1], 100)
fluxes_ring = np.zeros(len(nu_arr))
for i in range(len(fluxes_ring)):
    fluxes_ring[i]=F(nu_arr[i], Sigma, 1*au, width, T_floor, 112.3*pc)
np.savetxt('nus_fluxes_ring.txt', np.column_stack((nu_arr, fluxes_ring)), header='nu [GHz] F_nu [mJy]', fmt='%f %f')


plt.errorbar(nu/1E9, Fnus, yerr=sFnus, fmt='o', label='Observed Data', color='blue')
plt.plot(nu_arr/1E9, fluxes_ring, label=f'Model', color='red')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Frequency (GHz)')
plt.ylabel('Flux Density (mJy)')
plt.title(f'Dusty Ring model T={T_floor} K')
plt.legend()
plt.grid(True, which="both", ls="--")
plt.savefig('best_fit_dusty_ring.pdf', dpi=300)
plt.show()


print('Spectral index:', np.log10(fluxes_ring[-1]/fluxes_ring[0])/np.log10(nu_arr[-1]/nu_arr[0]))

# we calculate the mass of the ding in dust
dust_to_gas_ratio = 0.01
dust_mass = Sigma * np.pi * ((1*au + width/2)**2 - (1*au - width/2)**2) * dust_to_gas_ratio
print('Dust mass of the ring (in Earth masses):', dust_mass/(5.974e27))
