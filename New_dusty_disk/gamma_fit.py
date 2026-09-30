# best fit for gamma=Mpdot/alpha model
import numpy as np
from numpy import sqrt, pi
from units_astro import *
from dustyCPD import Mp, Rin, Rout, mu, dis
from k_dust import kappa_mm

# First our data with the error bars BAND 7 on 2019: 118.5 ± 16.6
nu = [97.5E9, 145005402197.7, 343.5E9, 671E9] # [nu1, nu2] are the x-axis values , 671E9
Fnus = [12e-06, 21.4e-06, 0.000121, 0.000143] # [Fnu1, Fnu2] are the y-axis values in Jy , 0.000122
sFnus = [4.7e-06, 4.1e-06, 0.000013, 0.000115] # [sigma_Fnu1, sigma_Fnu2] are the y-axis errors in Jy , 0.000090

# To mJy
nu = np.array(nu)
Fnus = np.array(Fnus)*1E3
sFnus = np.array(sFnus)*1E3

# Function with the brightness temperature for the gamma model
def simple_gamma_tb(R, nu, gamma):
    Tc_Sigma=sqrt(G*Mp/(pi**2*R**3))*gamma*(mu/Rgas)**(5/4)*(1-(Rin/R)**(1/2))
    k_mm=kappa_mm(10*c/nu) # dust opacity receives wav in mm
    Tb=Tc_Sigma*k_mm # brightness temperature
    #print(Tb)
    return Tb 

def simple_gamma_flux(nu, gamma):
    R_arr=np.linspace(1.001*Rin, Rout, 100)
    F_arr=np.zeros(len(R_arr))
    wav=c/nu # here wav is on cm since we replace on the blackbody formula
    for i in range(len(R_arr)):
        F_arr[i]=2*kb*simple_gamma_tb(R_arr[i], nu, gamma)/wav**2*2*pi*R_arr[i]/dis**2
    return np.trapezoid(F_arr, R_arr)*1e26 # to mJy

def log_likelihood_F(gamma):

    ymodels = np.zeros(len(Fnus))
    Xis = np.zeros(len(Fnus))
    for i in range(len(Fnus)):
        ymodels[i] = simple_gamma_flux(nu[i], gamma*Mj/yr)
        Xis[i] = -0.5*((Fnus[i]-ymodels[i])/sFnus[i])**2

    # ejemplo
    # the likehood is the product of the likelihoods for each data point
    # we assume the errors are gaussian
    # L1 = norm.pdf(Fnus[0]-ymodel1, 0, sFnus[0])
    # 
    # 
    # L2 = norm.pdf(Fnus[1]-ymodel2, 0, sFnus[1])
    return np.sum(Xis)

gamma_arr=np.logspace(-3.2, -2.8, 100)

log_likelihood_arr=np.zeros(len(gamma_arr))
for i in range(len(gamma_arr)):
    log_likelihood_arr[i]=log_likelihood_F(gamma_arr[i])

import matplotlib.pyplot as plt
plt.plot(gamma_arr, log_likelihood_arr)
plt.xscale('log')
plt.xlabel('gamma (Mpdot/alpha)')
plt.ylabel('log likelihood')
plt.ylim(np.max(log_likelihood_arr)-10, np.max(log_likelihood_arr)+1)
plt.title('Log Likelihood for the gamma model')
plt.show()

# now print the errors as the range of gamma where log likelihood is within 0.5 of the maximum
gamma_maxl=gamma_arr[np.argmax(log_likelihood_arr)]
gamma_min=gamma_arr[np.where(log_likelihood_arr>np.max(log_likelihood_arr)-0.5)[0][0]]
gamma_max=gamma_arr[np.where(log_likelihood_arr>np.max(log_likelihood_arr)-0.5)[0][-1]]
print('gamma =', gamma_maxl, gamma_min, gamma_max)

# now in log10
print('log10(gamma) =', np.log10(gamma_maxl), np.log10(gamma_min)-np.log10(gamma_maxl), np.log10(gamma_max)-np.log10(gamma_maxl))

# now save the fluxes for the simplified model best fit
nu_arr = np.linspace(nu[0], nu[-1], 100)
fluxes_gamma = np.zeros(len(nu_arr))
for i in range(len(fluxes_gamma)):
    fluxes_gamma[i]=simple_gamma_flux(nu_arr[i], gamma_maxl*Mj/yr)
np.savetxt('nus_fluxes_gamma.txt', np.column_stack((nu_arr, fluxes_gamma)), header='nu [GHz] F_nu [mJy]', fmt='%f %f')

# print maximun likelihood and the spectral index
print('Maximum log likelihood:', np.max(log_likelihood_arr))
print('Spectral index:', np.log10(fluxes_gamma[-1]/fluxes_gamma[0])/np.log10(nu_arr[-1]/nu_arr[0]))