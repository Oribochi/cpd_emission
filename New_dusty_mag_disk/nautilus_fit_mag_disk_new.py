# best fit dusty disk for the constant alpha case
from numpy import log10
from flux_gas_dust_mag_disk import F_nu as F
from magneticCPD import R_T1000, R_truncation
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
prior.add_parameter('log_Mpdot', dist=(-10, -2))
prior.add_parameter('log_Bps', dist=(-4, 4))
# prior.add_parameter('log_zeta', dist=(-15, 0))

def log_likelihood_F(param_dict):
    Mpdot = 10**param_dict['log_Mpdot']*Mj/yr
    Bps = 10**param_dict['log_Bps']
    dust_to_gas = 0

    R1k = R_T1000(Mpdot, Bps)/au
    Rtrunc = R_truncation(Mpdot, Bps)/au

    diff = abs(Rtrunc-R1k)/R1k
    R_diff = (Rtrunc-R1k)
    # we want to make sure that the truncation radius is bigger than the 1000 radius so the ionized gas can couple the magnetic field
    ymodels = np.zeros(len(Fnus))
    Xis = np.zeros(len(Fnus))
    for i in range(len(Fnus)):
        ymodels[i] = F(nu[i], Mpdot, Bps, zeta=dust_to_gas)
        Xis[i] = -0.5*((Fnus[i]-ymodels[i])/sFnus[i])**2
    # The more negative R_diff I want a bigger penalty, else just the sum of the likelihoods
    if R_diff < 0:
        return np.sum(Xis)-10*diff
    else:
        return np.sum(Xis)

sampler_zhu = Sampler(prior, log_likelihood_F, n_live=1000, pool = 16)             
sampler_zhu.run(verbose=True)

points, log_w, log_l = sampler_zhu.posterior()
corner.corner(
    points, weights=np.exp(log_w), bins=20, labels=[r'$\log\left( \frac{dM_p/dt}{M_j/yr} \right)$', r'$\log(B_{ps})$'], color='purple',
    plot_datapoints=False, range=np.repeat(0.999, len(prior.keys)), fontsize=12)


# Save the log Z and the corner plot
np.save('logZ1.npy', log_l)
np.save('logW1.npy', log_w)
np.save('logP1.npy', points)
plt.savefig('corner_plot_metals_Hminus.png')
plt.show()

# To see equal weights
points_equalw, log_w_equalw, log_l_equalw = sampler_zhu.posterior(equal_weight=True)

# now making our errors
Mpdot_maxl, Bps_maxl = points[np.argmax(log_l)]
Mpdot_min, Mpdot_max = error_param(points_equalw[:,0], Mpdot_maxl, bin_inf=-10, bin_sup=-2)
Bps_min, Bps_max = error_param(points_equalw[:,1], Bps_maxl, bin_inf=-4, bin_sup=4)


Bps_upper = upper_limit(points_equalw[:,1], -4)
Bps_upper_strict = strict_upper_limit(points_equalw[:,1])


print(10**Bps_maxl, 'G')
print(10**(Bps_maxl+Bps_max), 'G')
print(10**(Bps_maxl+Bps_min), 'G')

# save in a text file the best fit parameters and the errors


print('Mpdot =', Mpdot_maxl, Mpdot_min, Mpdot_max)
print('Bps =',  Bps_maxl, Bps_min, Bps_max)
print('Bps upper limit =', Bps_upper)
print('Bps = ', Bps_maxl, Bps_upper-Bps_maxl)
print('Bps upper strict limit =', Bps_upper_strict)
print('log_likelihood max =', np.max(log_l))

np.savetxt('best_fit_params_magnetic_disk.txt', np.array([
    ['Mpdot_best_log', Mpdot_maxl],
    ['Mpdot_min_log', Mpdot_min],
    ['Mpdot_max_log', Mpdot_max],
    ['Bps_best_log', Bps_maxl],
    ['Bps_min_log', Bps_min],
    ['Bps_max_log', Bps_max],
    ['Bps_upper_log', Bps_upper],
    ['Bps_upper_strict_log', Bps_upper_strict],
]), fmt='%s')

# vs the real data
# now making our spectral index
Mpdot=10**points[np.argmax(log_l)][0]*Mj/yr
Bps=10**points[np.argmax(log_l)][1]
dust_to_gas=0


fluxes = np.zeros(len(nu))
for i in range(len(nu)):
    fluxes[i]=F(nu[i], Mpdot, Bps, zeta=dust_to_gas)
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
plt.savefig('fig_magnetic_disk.png', bbox_inches='tight',dpi=300)
plt.show()