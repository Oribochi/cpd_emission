# Fitting the data

# Using Nautilus nested sampling to obtain the best fit parameters for the Zhu model given by our real data

import numpy as np
import matplotlib.pyplot as plt
from dustyCPD import * # my flux function
from flux_dusty_disk import F, F_thin
from nautilus import Prior
from units_astro import *
from nautilus import Sampler
import corner
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

# Now with the model of Fthin that only depends on one parameter gamma we fit the likelihood

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

fluxes = np.zeros(len(nus))
for i in range(len(nus)):
    fluxes[i]=F_thin(10*c/nus[i], 10**max_log_likelihood_gamma)

# to inches the column and the text width
column_width=256.0748*0.0138888889
text_width=523.5307*0.0138888889

# vs the real data
fig = plt.figure(layout='constrained', figsize=(4.8, 3.8), dpi=300)
plt.errorbar(np.array(nu)/10**9, Fnus, yerr=sFnus, fmt='o', label='Data', color='C0')
plt.plot(np.array(nus)/10**9, fluxes, label='Simple dusty disk', color='C1')
plt.loglog()
plt.xlabel(r'$\nu$ [GHz]')
plt.ylabel(r'$F_{\nu}$ [mJy]')
plt.legend(loc='best')
plt.savefig('best_fit_simple.png',bbox_inches='tight', dpi=300)
plt.show()