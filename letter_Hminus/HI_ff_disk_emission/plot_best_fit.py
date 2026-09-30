# Ploting the profiles of the best fit
import numpy as np
import matplotlib.pyplot as plt
from free_free_CPD import F_ff
from flux_dusty_disk import F
from units_astro import *
from numpy import exp, log, log10, sqrt
from dustyCPD import Rin, Rout
import smplotlib
import matplotlib
matplotlib.rcParams['axes.linewidth'] = 1. # the default is too thin

# the following should avoid having the silly-looking "10^0, 10^1, 10^2",
#  the unnecessarily complicated and "noisy" versions of "1, 10, 100"!
matplotlib.rcParams['axes.formatter.min_exponent'] = 5
matplotlib.rcParams['axes.formatter.limits'] = (-4,4)
matplotlib.rcParams['axes.formatter.useoffset'] = False  # also usually better
plt.style.use('tableau-colorblind10')    

# First our data with the error bars
nus = [97.5E9, 145005402197.7, 343.5E9, 671E9] # [nu1, nu2] are the x-axis values , 671E9
Fnus = [12e-06, 21.4e-06, 0.000121, 0.000143] # [Fnu1, Fnu2] are the y-axis values in Jy , 0.000122
sFnus = [4.7e-06, 4.1e-06, 0.000013, 0.000115] # [sigma_Fnu1, sigma_Fnu2] are the y-axis errors in Jy , 0.000090

# To mJy
nu = np.array(nus)
Fnus = np.array(Fnus)*1E3
sFnus = np.array(sFnus)*1E3


# For log_Mpdot=-6
def likelihood1(param_dict):
    Mpdot=10**(-6)
    zeta=10**param_dict['log_zeta']
    alpha= 10**param_dict['log_alpha']

    # the model prediction for the data this flux is in mJy
    ymodels = np.zeros(len(Fnus))
    Xis = np.zeros(len(Fnus))
    for i in range(len(Fnus)):
        ymodels[i] = F(10*c/nu[i], Mpdot*Mj/yr, alpha, zeta=zeta)+F_ff(nu[i], Mpdot*Mj/yr, alpha)
        Xis[i] = -0.5*((Fnus[i]-ymodels[i])/sFnus[i])**2

    return np.sum(Xis)


# For log_alpha=0.01
def likelihood2(param_dict):
    Mpdot=10**param_dict['log_Mpdot']
    zeta=10**param_dict['log_zeta']
    alpha=0.01

    # the model prediction for the data this flux is in mJy
    ymodels = np.zeros(len(Fnus))
    Xis = np.zeros(len(Fnus))
    for i in range(len(Fnus)):
        ymodels[i] = F(10*c/nu[i], Mpdot*Mj/yr, alpha, zeta=zeta)+F_ff(nu[i], Mpdot*Mj/yr, alpha)
        Xis[i] = -0.5*((Fnus[i]-ymodels[i])/sFnus[i])**2

    return np.sum(Xis)


#just the 2nd column
parameters1=np.loadtxt('best_fit_HII_fixed_Mpdot.txt', usecols=(1,))
Mpdot1=10**(-6)*Mj/yr # in Mj/yr
alpha1=10**parameters1[1]
zeta1=10**parameters1[4]

print('log_likelihood for fixed Mpdot:', likelihood1({'log_alpha': log10(alpha1), 'log_zeta': log10(zeta1)}))

parameters2=np.loadtxt('best_fit_HII_fixed_alpha.txt', usecols=(1,))
Mpdot2=10**(parameters2[1])*Mj/yr # in Mj/yr
alpha2=10**(-2)
zeta2=10**parameters2[4]

print('log_likelihood for fixed alpha:', likelihood2({'log_Mpdot': log10(Mpdot2/Mj*yr), 'log_zeta': log10(zeta2)}))

f1=np.zeros(len(nu))
f2=np.zeros(len(nu))
for i in range(len(nu)):
    f1[i]=F(10*c/nu[i], Mpdot1, alpha1, zeta=zeta1)+F_ff(nu[i], Mpdot1, alpha1)
    print('Flux at', round(nus[i]/1E9, 1), 'GHz =', f1[i], ' (fixed Mpdot)')

    f2[i]=F(10*c/nu[i], Mpdot2, alpha2, zeta=zeta2)+F_ff(nu[i], Mpdot2, alpha2)
    print('Flux at', round(nus[i]/1E9, 1), 'GHz =', f2[i], ' (fixed alpha)')
    if i>0:
        print('Spectral index is ', (log(f1[i])-log(f1[i-1]))/(log(nus[i])-log(nus[i-1])), ' (fixed Mpdot)')
        print('Spectral index is ', (log(f2[i])-log(f2[i-1]))/(log(nus[i])-log(nus[i-1])), ' (fixed alpha)')

array_nu=np.logspace(log10(nu[0]), log10(nu[-1]*1e5), 300)
fluxes1= np.zeros(len(array_nu))
fluxes2= np.zeros(len(array_nu))

for i in range(len(array_nu)):
    fluxes1[i] = F_ff(array_nu[i], Mpdot1, alpha1)+ F(10*c/array_nu[i], Mpdot1, alpha1, zeta=zeta1)
    fluxes2[i] = F_ff(array_nu[i], Mpdot2, alpha2)+ F(10*c/array_nu[i], Mpdot2, alpha2, zeta=zeta2)

# Plotting the results
column_width=256.0748*0.0138888889
text_width=523.5307*0.0138888889

# the profiles
fig = plt.figure(figsize=(4.8,3.8), dpi=300)

plt.errorbar(nu/1E9, Fnus, yerr=sFnus, fmt='o', color='C0', label='Data')
plt.plot(array_nu/1E9, fluxes2, label=r'${\rm Fixed}\, \alpha$', color='C1')
plt.plot(array_nu/1E9, fluxes1, label=r'${\rm Fixed}\, \dot{M}_{\rm p}$', color='C2')
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'$\nu$ [GHz]')
plt.ylabel(r'$F_{\nu}$ [mJy]')
plt.legend(loc='best')
plt.savefig('plot_SED.png', dpi=300, bbox_inches='tight')
plt.show()

# save array nu and fluxes
np.savetxt('SED_HII_fluxes.txt', np.column_stack((array_nu, fluxes1, fluxes2)), header='nu (GHz) flux_fixed_Mpdot flux_fixed_alpha', fmt='%.6e')