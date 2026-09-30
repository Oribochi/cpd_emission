# Tfloor diferent temperatures for the stratified disc
from numpy import log10
from flux_dusty_disk import F_nu_dust as F
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
prior.add_parameter('log_alpha', dist=(-10, 0))

def log_likelihood_20(param_dict):
    Mpdot = 10**param_dict['log_Mpdot']
    alpha = 10**param_dict['log_alpha']

    ymodels = np.zeros(len(Fnus))
    Xis = np.zeros(len(Fnus))
    for i in range(len(Fnus)):
        ymodels[i] = F(nu[i], Mpdot*Mj/yr, alpha, TISM=20,zeta=0.01)
        Xis[i] = -0.5*((Fnus[i]-ymodels[i])/sFnus[i])**2

    # ejemplo
    # the likehood is the product of the likelihoods for each data point
    # we assume the errors are gaussian
    # L1 = norm.pdf(Fnus[0]-ymodel1, 0, sFnus[0])
    # 
    # 
    # L2 = norm.pdf(Fnus[1]-ymodel2, 0, sFnus[1])
    return np.sum(Xis)

sampler_20 = Sampler(prior, log_likelihood_20, n_live=1000, pool = 16)
sampler_20.run(verbose=True)

points, log_w, log_l = sampler_20.posterior()
# corner.corner(
#     points, weights=np.exp(log_w), bins=20, labels=[r'$\log\left( \frac{dM_p/dt}{M_j/yr} \right)$', r'$\log(\alpha)$'], color='purple',
#     plot_datapoints=False, range=np.repeat(0.999, len(prior.keys)), fontsize=12)

# plt.show()

# now making our spectral index
Mpdot_20=10**points[np.argmax(log_l)][0]*Mj/yr
alpha_20=10**points[np.argmax(log_l)][1]

# now save the fluxes for the simplified model best fit
nu_arr = np.linspace(nu[0], nu[-1], 100)
fluxes_20 = np.zeros(len(nu_arr))
for i in range(len(fluxes_20)):
    fluxes_20[i]=F(nu_arr[i], Mpdot_20, alpha_20, TISM=20, zeta=0.01)

# now the same but with 15 and 10 K
def log_likelihood_15(param_dict):
    Mpdot = 10**param_dict['log_Mpdot']
    alpha = 10**param_dict['log_alpha']

    ymodels = np.zeros(len(Fnus))
    Xis = np.zeros(len(Fnus))
    for i in range(len(Fnus)):
        ymodels[i] = F(nu[i], Mpdot*Mj/yr, alpha, TISM=15,zeta=0.01)
        Xis[i] = -0.5*((Fnus[i]-ymodels[i])/sFnus[i])**2

    return np.sum(Xis)

sampler_15 = Sampler(prior, log_likelihood_15, n_live=1000, pool = 16)
sampler_15.run(verbose=True)

points, log_w, log_l = sampler_15.posterior()
# corner.corner(
#     points, weights=np.exp(log_w), bins=20, labels=[r'$\log\left( \frac{dM_p/dt}{M_j/yr} \right)$', r'$\log(\alpha)$'], color='purple',
#     plot_datapoints=False, range=np.repeat(0.999, len(prior.keys)), fontsize=12)

# plt.show()

Mpdot_15=10**points[np.argmax(log_l)][0]*Mj/yr
alpha_15=10**points[np.argmax(log_l)][1]

fluxes_15 = np.zeros(len(nu_arr))
for i in range(len(fluxes_15)):
    fluxes_15[i]=F(nu_arr[i], Mpdot_15, alpha_15, TISM=15, zeta=0.01)

def log_likelihood_10(param_dict):
    Mpdot = 10**param_dict['log_Mpdot']
    alpha = 10**param_dict['log_alpha']

    ymodels = np.zeros(len(Fnus))
    Xis = np.zeros(len(Fnus))
    for i in range(len(Fnus)):
        ymodels[i] = F(nu[i], Mpdot*Mj/yr, alpha, TISM=10,zeta=0.01)
        Xis[i] = -0.5*((Fnus[i]-ymodels[i])/sFnus[i])**2

    return np.sum(Xis)

sampler_10 = Sampler(prior, log_likelihood_10, n_live=1000, pool = 16)
sampler_10.run(verbose=True)

points, log_w, log_l = sampler_10.posterior()
# corner.corner(
#     points, weights=np.exp(log_w), bins=20, labels=[r'$\log\left( \frac{dM_p/dt}{M_j/yr} \right)$', r'$\log(\alpha)$'], color='purple',
#     plot_datapoints=False, range=np.repeat(0.999, len(prior.keys)), fontsize=12)

# plt.show()

Mpdot_10=10**points[np.argmax(log_l)][0]*Mj/yr
alpha_10=10**points[np.argmax(log_l)][1]

fluxes_10 = np.zeros(len(nu_arr))
for i in range(len(fluxes_10)):
    fluxes_10[i]=F(nu_arr[i], Mpdot_10, alpha_10, TISM=10, zeta=0.01)

# now plot everything together
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

plt.figure(figsize=(4.8,3.8))
plt.errorbar(nu/1E9, Fnus, yerr=sFnus, fmt='o', color='C0', label='Data')
plt.plot(nu_arr/1E9, fluxes_20, label=r'$T_{\rm floor}=20$ K', color='C1')
plt.plot(nu_arr/1E9, fluxes_15, label=r'$T_{\rm floor}=15$ K', color='C2')
plt.plot(nu_arr/1E9, fluxes_10, label=r'$T_{\rm floor}=10$ K', color='C3')
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'$\nu$ [GHz]', fontsize=14)
plt.ylabel(r'$F_\nu$ [mJy]', fontsize=14)
plt.legend(fontsize=12)
# make the ticks bigger
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.savefig('T_floor_comparison.pdf', bbox_inches='tight', dpi=300)
plt.show()