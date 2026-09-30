# Adding the jet flux

# Using Nautilus nested sampling to obtain the best fit parameters for the Zhu model given by our real data

import numpy as np
import matplotlib.pyplot as plt
from dustyCPD import * # my flux function
from flux_dusty_disk import F, F_thin, S, jet_flux_lambda
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


# Now adding new priors
prior = Prior()
prior.add_parameter('log_gamma', dist=(-15, 15))
prior.add_parameter('log_zeta', dist=(-15, 15))

def likelihood_jet1(param_dict):
    gamma = 10**(param_dict['log_gamma']) #pensamos en fitear el orden de magnitud de cada parametro
    zeta = 10**(param_dict['log_zeta']) 
    alpha = 10**(-5) # we fixed alpha=0.01
    Mpdot = alpha*gamma*Mj/yr
    
    ymodels = np.zeros(len(Fnus))
    Xis = np.zeros(len(Fnus))

    for i in range(len(Fnus)):
        ymodels[i] = F(10*c/nu[i], Mpdot, alpha, zeta=zeta)+jet_flux_lambda(0.4, 10*c/nu[i], Mpdot)
        Xis[i] = -0.5*((Fnus[i]-ymodels[i])/sFnus[i])**2

    return np.sum(Xis)

# Corro el nested sampling de nautilus

sampler_jet1 = Sampler(prior, likelihood_jet1, n_live=1000, pool=16)
sampler_jet1.run(verbose=True)

points, log_w, log_l = sampler_jet1.posterior()

corner.corner(points, weights=np.exp(log_w), bins=20, labels=[r'$log(\gamma)$',r'$log(\zeta)$'], color='purple',
    plot_datapoints=False, range=np.repeat(0.999, len(prior.keys)), fontsize=12)

# the best fit parameters
print('log(Z) =', log_l)
print('Evidence =', np.exp(log_l))
print('Best fit parameters =', np.mean(points, axis=0))
print('Errors =', np.std(points, axis=0))
print('The log likelihood is', likelihood_jet1({'log_gamma': np.mean(points, axis=0)[0], 'log_zeta': np.mean(points, axis=0)[1]}))
print('The maximum likelihood parameters are', points[np.argmax(log_l)])
print('The maximum likelihood is ', np.max(log_l))

# now making our spectral index
gamma=10**points[np.argmax(log_l)][0]
zeta=10**points[np.argmax(log_l)][1]
alpha=10**(-5)
Mpdot = alpha*gamma*Mj/yr

F_arr_zhu = F(10*c/nus, Mpdot, alpha, zeta=zeta)

F_arr_jet = jet_flux_lambda(0.4, 10*c/nus, Mpdot)

F_arr = F_arr_zhu + F_arr_jet

print(F_arr[-1])
# to inches the column and the text width
column_width=256.0748*0.0138888889
text_width=523.5307*0.0138888889

# vs the real data
fig = plt.figure(layout='constrained', figsize=(column_width, column_width/(1.3)), dpi=300)
plt.errorbar(np.array(nu)/10**9, Fnus, yerr=sFnus, fmt='o', label='Data', color='blue')
plt.plot(np.array(nus)/10**9, F_arr_zhu, label='Dusty disk', alpha=0.5, color='red')
plt.plot(np.array(nus)/10**9, F_arr_jet, label='Jet', alpha=0.5, color='purple')
plt.plot(np.array(nus)/10**9, F_arr, label='Best fit', alpha=0.7, color='orange')
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'$\nu$ [GHz]', fontsize=8)
plt.ylabel(r'$F_{\nu}$ [mJy]', fontsize=8)
plt.xticks(fontsize=6, minor=True)
plt.yticks(fontsize=6)
plt.xticks(fontsize=6, minor=False)
plt.legend(loc='upper left', fontsize=8)
plt.savefig('best_fit_zhu_jet.png',bbox_inches='tight', dpi=300)
plt.show()
