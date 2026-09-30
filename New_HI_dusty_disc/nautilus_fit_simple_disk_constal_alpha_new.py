# best fit dusty disk for the constant alpha case
from numpy import log10
from flux_gas_dust_simple_disk import F_nu as F
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
prior.add_parameter('log_Mpdot', dist=(-15, 0))
prior.add_parameter('log_zeta', dist=(-10, 0))

def log_likelihood_F(param_dict):
    Mpdot = 10**param_dict['log_Mpdot']
    dust_to_gas = 10**param_dict['log_zeta']
    alpha=0.01 # we asume constant alpha in this fit

    ymodels = np.zeros(len(Fnus))
    Xis = np.zeros(len(Fnus))
    for i in range(len(Fnus)):
        ymodels[i] = F(nu[i], Mpdot*Mj/yr, alpha, zeta=dust_to_gas)
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
    points, weights=np.exp(log_w), bins=20, labels=[r'$\log\left( \frac{dM_p/dt}{M_j/yr} \right)$', r'$\log(\zeta)$'], color='purple',
    plot_datapoints=False, range=np.repeat(0.999, len(prior.keys)), fontsize=12)

# Save the log Z and the corner plot
np.save('logZ1.npy', log_l)
np.save('logW1.npy', log_w)
np.save('logP1.npy', points)
plt.savefig('corner_plot_simple_disk_constant_alpha.pdf')
plt.show()


# To see equal weights
points_equalw, log_w_equalw, log_l_equalw = sampler_zhu.posterior(equal_weight=True)

# now making our errors
Mpdot_maxl, zet_maxl = points[np.argmax(log_l)]
Mpdot_min, Mpdot_max = error_param(points_equalw[:,0], Mpdot_maxl, bin_inf=-15, bin_sup=0)
zeta_min, zeta_max = error_param(points_equalw[:,1], zet_maxl, bin_inf=-10, bin_sup=0)

zeta_upper = upper_limit(points_equalw[:,1], -9)
zeta_upper_strict = strict_upper_limit(points_equalw[:,1])

print('Mpdot =', Mpdot_maxl, Mpdot_min, Mpdot_max)
print('zeta =',  zet_maxl, zeta_min, zeta_max)
print('zeta upper limit =', zeta_upper)
print('zeta upper strict limit =', zeta_upper_strict)
print('log_likelihood max =', np.max(log_l))

np.savetxt('best_fit_params_magnetic_disk_with_zeta.txt', np.array([
    ['Mpdot_best_log', Mpdot_maxl],
    ['Mpdot_min_log', Mpdot_min],
    ['Mpdot_max_log', Mpdot_max],
    ['zeta_max_log', zet_maxl],
    ['zeta_upper_limit_log', zeta_upper],
    ['zeta_upper_strict_limit_log', zeta_upper_strict],
]), fmt='%s')

# vs the real data
# now making our spectral index
Mpdot=10**points[np.argmax(log_l)][0]*Mj/yr
dust_to_gas=10**points[np.argmax(log_l)][1]
alpha=0.01

fluxes = np.zeros(len(nu))
for i in range(len(nu)):
    fluxes[i]=F(nu[i], Mpdot, alpha, zeta=dust_to_gas)
# making a plot

plt.errorbar(nu/1E9, Fnus, yerr=sFnus, fmt='o', label='Observed Data', color='blue')
plt.plot(nu/1E9, fluxes, label='Model Prediction', color='red')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Frequency (GHz)')
plt.ylabel('Flux Density (mJy)')
plt.title('Flux Density vs Frequency for CPD around PDS 70 c')
plt.legend()
plt.grid(True, which="both", ls="--")
plt.savefig('best_fit_dusty_disk_constant_alpha.pdf', dpi=300)
plt.show()