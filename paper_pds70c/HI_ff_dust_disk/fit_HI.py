# Fit mpdot fixed
# Using Nautilus nested sampling to obtain the best fit parameters for the Zhu model given by our real data
import numpy as np
import matplotlib.pyplot as plt
from flux_dust_gas_disk_simple import F_nu_tot as F
from nautilus import Prior
from units_astro import *
from errors import error_param, upper_limit, strict_upper_limit
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

# fixed Mpdot

prior = Prior()
prior.add_parameter('log_alpha', dist=(-10, 2))
prior.add_parameter('log_zeta', dist=(-15, 0))

# For log_Mpdot=-6
def likelihood2(param_dict):
    alpha=10**param_dict['log_alpha']
    zeta=10**param_dict['log_zeta']
    Mpdot=1E-6

    # the model prediction for the data this flux is in mJy
    ymodels = np.zeros(len(Fnus))
    Xis = np.zeros(len(Fnus))
    for i in range(len(Fnus)):
        ymodels[i] = F(10*c/nu[i], Mpdot*Mj/yr, alpha, zeta=zeta)
        Xis[i] = -0.5*((Fnus[i]-ymodels[i])/sFnus[i])**2

    return np.sum(Xis)

# running the nested sampling
sampler2 = Sampler(prior, likelihood2, n_live=1000, pool=16)
sampler2.run(verbose=True)


points2, log_w2, log_l2 = sampler2.posterior()

corner.corner(points2, weights=np.exp(log_w2), bins=20, labels=[r'$\log(\alpha)$',r'$log(\zeta)$'], color='purple',
    plot_datapoints=False, range=np.repeat(0.999, len(prior.keys)), fontsize=12)

# Save the log Z and the corner plot
np.save('logZ2.npy', log_l2)
plt.show()

# To see equal weights
points_equalw2, log_w_equalw2, log_l_equalw2 = sampler2.posterior(equal_weight=True)

# print the results
print('log(Z) =', log_l2)
print('Evidence =', np.exp(log_l2))
print('Best fit parameters =', np.mean(points2, axis=0))
print('Errors =', np.std(points2, axis=0))
print('The log likelihood is', likelihood2({'log_alpha': np.mean(points2, axis=0)[0], 'log_zeta': np.mean(points2, axis=0)[1]}))
# The best fit parameters are with the maximum likelihood
print('The maximum likelihood is', np.max(log_l2))
print('The maximum likelihood parameters are')

# now making our errors
alpha_maxl, zeta_maxl = points2[np.argmax(log_l2)]
alpha_min, alpha_max = error_param(points_equalw2[:,0], alpha_maxl,bin_inf=-10, bin_sup=2)
zeta_min, zeta_max = error_param(points_equalw2[:,1], zeta_maxl, bin_inf=-15, bin_sup=0)

zeta_upper = upper_limit(points_equalw2[:,1], -9)
zeta_upper_strict = strict_upper_limit(points_equalw2[:,1])

print('alpha =', alpha_maxl, alpha_min, alpha_max)
print('zeta =',  zeta_maxl, zeta_min, zeta_max)
print('zeta upper limit =', zeta_upper)
print('zeta = ', zeta_maxl, zeta_upper-zeta_maxl)
print('zeta upper strict limit =', zeta_upper_strict)

# now making our spectral index best fit for the fluxes
alpha = 10**points2[np.argmax(log_l2)][0]
zeta = 10**points2[np.argmax(log_l2)][1]
Mpdot = 1E-6

fluxes2 = np.zeros(len(nu))
for i in range(len(nu)):
    fluxes2[i]=F(10*c/nu[i], Mpdot*Mj/yr, alpha, zeta=zeta)
    print('Flux at', round(nus[i]/1E9, 1), 'GHz =', fluxes2[i])
    if i>0:
        print('Spectral index is ', (log(fluxes2[i])-log(fluxes2[i-1]))/(log(nus[i])-log(nus[i-1])))

        # array of frequencies
nu_arr=np.logspace(log10(nus[0]), log10(nus[-1]), 100)

# array of fluxes
F_arr2=np.zeros(len(nu_arr))


for i in range(len(nu_arr)):
    F_arr2[i]=F(nu_arr[i], Mpdot*Mj/yr, alpha, zeta=zeta)

# vs the real data
plt.errorbar(np.array(nus)/10**9, Fnus, yerr=sFnus, fmt='o', label='Data', color='blue')
plt.plot(nu_arr/10**9, F_arr2, label='Best fit', alpha=0.5, color='red')
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'$\nu$ [GHz]', fontsize=12)
plt.ylabel(r'$F_{\nu}$ [mJy]', fontsize=12)
plt.legend(loc='upper left')
plt.show()
plt.savefig('best_fit_HII_fixed_Mpdot.png', dpi=300)

# save the best loglikelohood, the best fit parameters with the errors, the upper limit and the strict upper limit
np.savetxt('best_fit_HII_fixed_Mpdot.txt', [['log(L)',np.max(log_l2)], ['alpha_maxl', alpha_maxl], ['alpha_min', alpha_min], ['alpha_max', alpha_max], ['zeta_maxl', zeta_maxl], ['zeta_min', zeta_min], ['zeta_max', zeta_max], ['zeta_upper', zeta_upper], ['zeta_upper_strict', zeta_upper_strict]], fmt='%s')
# # #############################################################################
# # #############################################################################

#Now the same but with Mpdot as a free parameter and alpha fixed to 0.01

# # fixed alpha
# prior = Prior()
# prior.add_parameter('log_Mpdot', dist=(-15, 15))
# prior.add_parameter('log_zeta', dist=(-15, 15))

# # For log_alpha=0.01
# def likelihood(param_dict):
#     Mpdot=10**param_dict['log_Mpdot']
#     zeta=10**param_dict['log_zeta']
#     alpha=0.01

#     # the model prediction for the data this flux is in mJy
#     ymodels = np.zeros(len(Fnus))
#     Xis = np.zeros(len(Fnus))
#     for i in range(len(Fnus)):
#         ymodels[i] = F(10*c/nu[i], Mpdot*Mj/yr, alpha, zeta=zeta)+F_ff(nu[i], Mpdot*Mj/yr, alpha)
#         Xis[i] = -0.5*((Fnus[i]-ymodels[i])/sFnus[i])**2

#     return np.sum(Xis)

# # running the nested sampling
# sampler = Sampler(prior, likelihood, n_live=1000, pool=16)
# sampler.run(verbose=True)


# points, log_w, log_l = sampler.posterior()

# corner.corner(
#     points, weights=np.exp(log_w), bins=20, labels=[r'$\log \left(\frac{dM_{\rm p}/dt}{M_{\rm jup}/{\rm yr}} \right)$',r'$log(\zeta)$'], color='purple',
#     plot_datapoints=False, range=np.repeat(0.999, len(prior.keys)), fontsize=12)

# # Save the log Z and the corner plot
# np.save('logZ.npy', log_l)
# plt.show()


# # To see equal weights
# points_equalw, log_w_equalw, log_l_equalw = sampler.posterior(equal_weight=True)

# # print the results
# print('log(Z) =', log_l)
# print('Evidence =', np.exp(log_l))
# print('Best fit parameters =', np.mean(points, axis=0))
# print('Errors =', np.std(points, axis=0))
# print('The log likelihood is', likelihood({'log_Mpdot': np.mean(points, axis=0)[0], 'log_zeta': np.mean(points, axis=0)[1]}))
# # The best fit parameters are with the maximum likelihood
# print('The maximum likelihood is', np.max(log_l))
# print('The maximum likelihood parameters are')

# # now making our errors
# Mpdot_maxl, zeta_maxl = points[np.argmax(log_l)]
# Mpdot_min, Mpdot_max = error_param(points_equalw[:,0], Mpdot_maxl)
# zeta_min, zeta_max = error_param(points_equalw[:,1], zeta_maxl)

# zeta_upper = upper_limit(points_equalw[:,1], -9)
# zeta_upper_strict = strict_upper_limit(points_equalw[:,1])

# print('Mpdot =', Mpdot_maxl, Mpdot_min, Mpdot_max)
# print('zeta =',  zeta_maxl, zeta_min, zeta_max)
# print('zeta upper limit =', zeta_upper)
# print('zeta = ', zeta_maxl, zeta_upper-zeta_maxl)
# print('zeta upper strict limit =', zeta_upper_strict)

# # now making our spectral index best fit for the fluxes
# alpha = 0.01
# zeta = 10**points[np.argmax(log_l)][1]
# Mpdot = 10**points[np.argmax(log_l)][0]

# fluxes = np.zeros(len(nu))
# for i in range(len(nu)):
#     fluxes[i]=F(10*c/nu[i], Mpdot*Mj/yr, alpha, zeta=zeta)+F_ff(nu[i], Mpdot*Mj/yr, alpha)
#     print('Flux at', round(nus[i]/1E9, 1), 'GHz =', fluxes[i])
#     if i>0:
#         print('Spectral index is ', (log(fluxes[i])-log(fluxes[i-1]))/(log(nus[i])-log(nus[i-1])))

#         # array of frequencies
# nu_arr=np.logspace(log10(nus[0]), log10(nus[-1]), 100)

# # array of fluxes
# F_arr=np.zeros(len(nu_arr))
# F_arr_ff=np.zeros(len(nu_arr))
# F_arr_zhu=np.zeros(len(nu_arr))


# for i in range(len(nu_arr)):
#     F_arr_ff[i]=F_ff(nu_arr[i], Mpdot*Mj/yr, alpha)
#     F_arr_zhu[i]=F(10*c/nu_arr[i], Mpdot*Mj/yr, alpha, zeta=zeta)
#     F_arr[i]=F_arr_ff[i]+F_arr_zhu[i]

# # vs the real data
# plt.errorbar(np.array(nus)/10**9, Fnus, yerr=sFnus, fmt='o', label='Data', color='blue')
# plt.plot(np.array(nus)/10**9, fluxes, label='Best fit', alpha=0.5, color='red')
# plt.xscale('log')
# plt.yscale('log')
# plt.xlabel(r'$\nu$ [GHz]', fontsize=12)
# plt.ylabel(r'$F_{\nu}$ [mJy]', fontsize=12)
# plt.legend(loc='upper left')
# plt.show()
# plt.savefig('best_fit_HII_fixed_alpha.png', dpi=300)

# # save the best loglikelohood, the best fit parameters with the errors, the upper limit and the strict upper limit
# np.savetxt('best_fit_HII_fixed_alpha.txt', [['log(L)',np.max(log_l)], ['Mpdot_maxl', Mpdot_maxl], ['Mpdot_min', Mpdot_min], ['Mpdot_max', Mpdot_max], ['zeta_maxl', zeta_maxl], ['zeta_min', zeta_min], ['zeta_max', zeta_max], ['zeta_upper', zeta_upper], ['zeta_upper_strict', zeta_upper_strict]], fmt='%s')
