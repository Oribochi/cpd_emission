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
from numpy import exp, log, log10, sqrt


# First our data with the error bars BAND 7 on 2019: 118.5 ± 16.6
nu = [97.5E9, 145005402197.7, 343.5E9, 671E9] # [nu1, nu2] are the x-axis values , 671E9
Fnus = [12e-06, 21.4e-06, 0.000121, 0.000143] # [Fnu1, Fnu2] are the y-axis values in Jy , 0.000122
sFnus = [4.7e-06, 4.1e-06, 0.000013, 0.000115] # [sigma_Fnu1, sigma_Fnu2] are the y-axis errors in Jy , 0.000090

# To mJy
nu = np.array(nu)
Fnus = np.array(Fnus)*1E3
sFnus = np.array(sFnus)*1E3

nus=np.logspace(log10(nu[0]),log10(nu[-1]),100)

alpha=10**(-6)

# Now with the model of Fthin that only depends on one parameter gamma we fit the likelihood

def likelihood_Fthin(log_gamma):
    # the model prediction for the data this flux is in mJy
    gamma = 10**(log_gamma)
    ymodels = np.zeros(len(Fnus))
    Xis = np.zeros(len(Fnus))
    for i in range(len(Fnus)):
        ymodels[i] = F(10*c/nu[i], gamma*alpha*Mj/yr, alpha)
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

gamma=10**max_log_likelihood_gamma
Mpdot=gamma*alpha*Mj/yr

fluxes = np.zeros(len(nus))
for i in range(len(nus)):
    fluxes[i]=F(10*c/nus[i], Mpdot, alpha)

# to inches the column and the text width
column_width=256.0748*0.0138888889
text_width=523.5307*0.0138888889

# vs the real data
fig = plt.figure(layout='constrained', figsize=(column_width, column_width/(1.3)), dpi=300)
plt.errorbar(np.array(nu)/10**9, Fnus, yerr=sFnus, fmt='o', label='Data', color='blue')
plt.plot(np.array(nus)/10**9, fluxes, label='Best fit', alpha=0.5, color='red')
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'$\nu$ [GHz]', fontsize=8)
plt.ylabel(r'$F_{\nu}$ [mJy]', fontsize=8)
plt.xticks(fontsize=6, minor=True)
plt.yticks(fontsize=6)
plt.xticks(fontsize=6, minor=False)
plt.legend(loc='upper left', fontsize=8)
plt.savefig('best_fit_fixed_alpha.png',bbox_inches='tight', dpi=300)
plt.show()

# profiles
T_ext_arr=[T_ext(r, Mpdot) for r in R]
T_eff_arr=[T_eff(r,Mpdot) for r in R]
T_visc_arr=[T_visc(r, Sigma(r, Mpdot, alpha), Mpdot) for r in R]
T_c_arr=[T_c_approx(r, Mpdot, alpha) for r in R]

plt.plot(R, T_ext_arr)
plt.plot(R, T_eff_arr)
plt.plot(R, T_visc_arr)
plt.plot(R, T_c_arr)
plt.loglog()
plt.legend(['T_ext', 'T_eff', 'T_visc','T_c'])
plt.show()

# and the flux
S_arr=[S(10*c/nu[-1], r, Mpdot, alpha) for r in R]

plt.plot(R,S_arr)
plt.loglog()
plt.show()

S_arr_R=np.array(S_arr)*R**2
plt.plot(R,S_arr_R)
plt.loglog()
plt.show()