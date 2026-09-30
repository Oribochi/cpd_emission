# The opacity free-free of the He- atom using Gray 2021
from units_astro import *
from numpy import exp, log10, sqrt, pi, arctan
import numpy as np
import matplotlib.pyplot as plt
from simple_ionization_fraction import a4, simple_ionization_fraction_H_minus, mu, saha, fi4, a1, ionization_fraction_H_minus, partition_function

# He- opacityin units of cm2 per absorber (N(He⁻)) per electron pressure
def k_ff_He_minus(T,nu):
    theta = 5040/T
    wav = c/nu*1.0e8 # Convert frequency to wavelength (this is in angstroms)
    c0 = 9.66736 - 71.76242*theta + 105.29576*theta**2 - 56.49259*theta**3 + 10.69206*theta**4
    c1 = -10.50614 + 48.28802*theta - 70.43363*theta**2 + 37.80099*theta**3 - 7.15445*theta**4
    c2 = 2.74020 - 10.62144*theta + 15.50518*theta**2 - 8.33845*theta**3 + 1.57960*theta**4
    c3 = -0.19923 + 0.77485*theta - 1.13200*theta**2 + 0.60994*theta**3 - 0.11564*theta**4
    return 10**(c0 + c1*log10(wav) + c2*log10(wav)**2 + c3*log10(wav)**3)*1e-26

# Example T=5040 K, nu=1e14 Hz
def delta_k_squared_freq(d_K):
    # this is the relation between the diference in the wave number and the frequency
    # this is the definition in Someville 1964 eq 19 where delta_k^2=k_f^2-k_i^2=4 pi nu (a.u.)
    wav=911.27/d_K # (this is in angstroms) we return the frequency in Hz
    return c/wav*1.0e8

# wav1=delta_k_squared_freq(0.006) # example in McDowell 1966
# wav2=delta_k_squared_freq(0.008)

# print(k_ff_He_minus(5040, wav1)) # this is in cm2 per He atom per electron pressure!
# print(k_ff_He_minus(5040, wav2))

# now the absorption coeficient in cm2 per hydrogen atom is 
def k_ff_Heff(T, nu, rho):
    # the abundance of He is a4
    n = rho/(mu*mH) 
    number_density_e = simple_ionization_fraction_H_minus(T, rho)*n
    P_e = number_density_e*kb*T
    Phi_He = saha(T, number_density_e, 0.077)*P_e
    abundance_j=a4/a1
    return k_ff_He_minus(T, nu)*abundance_j/(1 + Phi_He/P_e)  # sin el Pe extra (aun no entiendo de donde sale)


# for example 
rho=1e-7 # g/cm3
T=3000
nu_arr = np.logspace(12, 15, 100) # frequency array in Hz

# this opacity was in cm2 per hydrogen atom
n_H = rho/(mu*mH)*a1
kappa_arr = k_ff_Heff(T, nu_arr, rho)*n_H
# plt.plot(nu_arr, kappa_arr)
# plt.loglog()
# plt.show()

#now just using the cross section and calculating n_e and He-
def He_minus_number_density(T, rho):
    ntot = rho / (mu * mH)
    number_density_e = simple_ionization_fraction_H_minus(T, rho) * ntot
    He_minus_abundance = 1/(1+saha(T, number_density_e, 0.077)) # this has the N1=He and N0=He⁻
    #print(He_minus_abundance)
    He_number_density = a4 * ntot # almost all is neutral He
    return He_minus_abundance * He_number_density

# # now the kappa_arr is
# kappa_arr2 = k_ff_He_minus(T,nu_arr) * He_minus_number_density(T, rho)

# plt.plot(nu_arr, kappa_arr, label='from opacity')
# plt.plot(nu_arr, kappa_arr2, '--', label='from cross section')
# plt.loglog()
# plt.legend()
# plt.show()
