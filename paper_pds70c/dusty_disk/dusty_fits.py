# Best dusty fits
# Fitting the data

# Using Nautilus nested sampling to obtain the best fit parameters for the Zhu model given by our real data

import numpy as np
import matplotlib.pyplot as plt
from dustyCPD import * # my flux function
from flux_dusty_disk import F, F_thin, S
from nautilus import Prior
from units_astro import *
from nautilus import Sampler
import corner
from errors import error_param, upper_limit, strict_upper_limit
from numpy import exp, log, log10, sqrt
import smplotlib
import matplotlib
matplotlib.rcParams['axes.linewidth'] = 1. # the default is too thin

# the following should avoid having the silly-looking "10^0, 10^1, 10^2",
#  the unnecessarily complicated and "noisy" versions of "1, 10, 100"!
matplotlib.rcParams['axes.formatter.min_exponent'] = 5
matplotlib.rcParams['axes.formatter.limits'] = (-4,4)
matplotlib.rcParams['axes.formatter.useoffset'] = False  # also usually better
plt.style.use('tableau-colorblind10')

# First our data with the error bars BAND 7 on 2019: 118.5 ± 16.6
nu = [97.5E9, 145005402197.7, 343.5E9, 671E9] # [nu1, nu2] are the x-axis values , 671E9
Fnus = [12e-06, 21.4e-06, 0.000121, 0.000143] # [Fnu1, Fnu2] are the y-axis values in Jy , 0.000122
sFnus = [4.7e-06, 4.1e-06, 0.000013, 0.000115] # [sigma_Fnu1, sigma_Fnu2] are the y-axis errors in Jy , 0.000090

# To mJy
nu = np.array(nu)
Fnus = np.array(Fnus)*1E3
sFnus = np.array(sFnus)*1E3

nus=np.logspace(log10(nu[0]),log10(nu[-1]),100)

# Now with the model of F that depends oof Mpdot and alpha we fit the likelihood
prior = Prior()
prior.add_parameter('log_Mpdot', dist=(-15, 15))
prior.add_parameter('log_alpha', dist=(-15, 15))

def likelihood_F(param_dict):
    Mpdot = 10**param_dict['log_Mpdot']
    alpha = 10**param_dict['log_alpha']

    ymodels = np.zeros(len(Fnus))
    Xis = np.zeros(len(Fnus))
    for i in range(len(Fnus)):
        ymodels[i] = F(10*c/nu[i], Mpdot*Mj/yr, alpha,zeta=0.01)
        Xis[i] = -0.5*((Fnus[i]-ymodels[i])/sFnus[i])**2

    # ejemplo
    # the likehood is the product of the likelihoods for each data point
    # we assume the errors are gaussian
    # L1 = norm.pdf(Fnus[0]-ymodel1, 0, sFnus[0])
    # 
    # 
    # L2 = norm.pdf(Fnus[1]-ymodel2, 0, sFnus[1])
    return np.sum(Xis)

sampler_zhu = Sampler(prior, likelihood_F, n_live=1000, pool = 16)             
sampler_zhu.run(verbose=True)

points, log_w, log_l = sampler_zhu.posterior()
corner.corner(
    points, weights=np.exp(log_w), bins=20, labels=[r'$\log\left( \frac{dM_p/dt}{M_j/yr} \right)$', r'$\log(\alpha)$'], color='purple',
    plot_datapoints=False, range=np.repeat(0.999, len(prior.keys)), fontsize=12)



# To see equal weights
points_equalw, log_w_equalw, log_l_equalw = sampler_zhu.posterior(equal_weight=True)

# now making our errors
Mpdot_maxl, zeta_maxl = points[np.argmax(log_l)]
Mpdot_min, Mpdot_max = error_param(points_equalw[:,0], Mpdot_maxl)
zeta_min, zeta_max = error_param(points_equalw[:,1], zeta_maxl)

zeta_upper = upper_limit(points_equalw[:,1], -9)
zeta_upper_strict = strict_upper_limit(points_equalw[:,1])

print('Mpdot =', Mpdot_maxl, Mpdot_min, Mpdot_max)
print('alpha =',  zeta_maxl, zeta_min, zeta_max)
print('alpha upper limit =', zeta_upper)
print('alpha = ', zeta_maxl, zeta_upper-zeta_maxl)
print('alpha upper strict limit =', zeta_upper_strict)


# the best fit parameters
print('log(Z) =', log_l)
print('Evidence =', np.exp(log_l))
print('Best fit parameters =', np.mean(points, axis=0))
print('Errors =', np.std(points, axis=0))
print('The log likelihood is', likelihood_F({'log_Mpdot': np.mean(points, axis=0)[0], 'log_alpha': np.mean(points, axis=0)[1]}))
print('The maximum likelihood parameters are', points[np.argmax(log_l)])
print('The maximum likelihood is ', np.max(log_l))

# vs the real data
# now making our spectral index
Mpdot=10**points[np.argmax(log_l)][0]*Mj/yr
alpha=10**points[np.argmax(log_l)][1]

fluxes = np.zeros(len(nus))
for i in range(len(nus)):
    fluxes[i]=F(10*c/nus[i], Mpdot, alpha, zeta=0.01)



def likelihood_Fthin(log_gamma):
    # the model prediction for the data this flux is in mJy
    gamma = 10**(log_gamma)
    ymodels = np.zeros(len(Fnus))
    Xis = np.zeros(len(Fnus))
    for i in range(len(Fnus)):
        ymodels[i] = F_thin(10*c/nu[i], gamma)
        Xis[i] = -0.5*((Fnus[i]-ymodels[i])/sFnus[i])**2

    # ejemplo
    # the likehood is the product of the likelihoods for each data point
    # we assume the errors are gaussian
    # L1 = norm.pdf(Fnus[0]-ymodel1, 0, sFnus[0])
    # 
    # 
    # L2 = norm.pdf(Fnus[1]-ymodel2, 0, sFnus[1])
    return np.sum(Xis)


# We search the best likelihood of the gammas
log_gammas = np.linspace(-10,0,1000)
likelihoods = [likelihood_Fthin(log_gamma) for log_gamma in log_gammas]

# we get the maximun of this relation with the error as the places where L is Lmax-1
max_log_likelihood = max(likelihoods)
max_log_likelihood_index = np.argmax(likelihoods)
max_log_likelihood_gamma = log_gammas[max_log_likelihood_index]
error = 1
error_indexes = np.where(likelihoods > max_log_likelihood - error)
error_gamma = log_gammas[error_indexes]
print('Best fit gamma =', max_log_likelihood_gamma)
print('Errors =', error_gamma[0]-max_log_likelihood_gamma, error_gamma[-1]-max_log_likelihood_gamma)
print('The log likelihood is', likelihood_Fthin(max_log_likelihood_gamma))

fluxes2 = np.zeros(len(nus))
for i in range(len(nus)):
    fluxes2[i]=F_thin(10*c/nus[i], 10**max_log_likelihood_gamma)


# to inches the column and the text width
column_width=256.0748*0.0138888889
text_width=523.5307*0.0138888889

# vs the real data
fig = plt.figure(layout='constrained', figsize=(4.8, 3.8))
plt.errorbar(np.array(nu)/10**9, Fnus, yerr=sFnus, fmt='o', label='Data', color='C0')
plt.plot(np.array(nus)/10**9, fluxes, label='Dusty disk', color='C1')
plt.plot(np.array(nus)/10**9, fluxes2, label='Simple dusty disk', color='C2')
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'$\nu$ [GHz]')
plt.ylabel(r'$F_{\nu}$ [mJy]')
plt.legend(loc='best')
plt.savefig('best_fit_dusty.png',bbox_inches='tight', dpi=300)
plt.show()

#print spectral index between 97.5 GHz and 671 GHz
alpha_spectral = (np.log10(fluxes[-1])-np.log10(fluxes[0]))/(np.log10(nus[-1])-np.log10(nus[0]))
print('Spectral index between 97.5 GHz and 671 GHz =', alpha_spectral)
#print spectral index between 97.5 GHz and 671 GHz for the simple model
alpha_spectral2 = (np.log10(fluxes2[-1])-np.log10(fluxes2[0]))/(np.log10(nus[-1])-np.log10(nus[0]))
print('Spectral index between 97.5 GHz and 671 GHz for the simple model =', alpha_spectral2)