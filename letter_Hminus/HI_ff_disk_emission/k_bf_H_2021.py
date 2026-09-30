# The opacity bound-free of the hydrogen atom using Gray 2021
from units_astro import *
from numpy import exp, log, sqrt, pi, arctan
import numpy as np
import matplotlib.pyplot as plt

# Some useful constants
hbar=h/(2*pi)

Rydberg=13.598*eV/(h*c)

alpha0=32/sqrt(3**3)*pi**2*qe**6/(h*c)**3*Rydberg

# the bound free Gaunt factor for hydrogen-like atoms
def gaunt_factor_bf(nu, n):
    wav = c/nu
    x=Rydberg*wav # adimentional parameter
    g_bf=1-0.3456/x**(1/3)*(x/n**2-1/2)
    if g_bf>0:
        return g_bf
    else:
        return 0

# The absorption coefficient corresponding to the bound–free transitions
def absorption_coefficient_bf(nu, n, Z=1):
    nu_0=13.598*eV/h/n**2
    if nu < nu_0:
        return 0  # no absorption if frequency is below ionization threshold
    else:
        wav = c/nu
        return alpha0*wav**3/n**5*gaunt_factor_bf(nu, n)

# wav_array = np.linspace(0,20000,200)*1e-8 # from 0 to 20000 Angstroms

# plt.figure(figsize=(10, 6))
# for n in range(1, 6):
#     plt.plot(wav_array*10**8, [absorption_coefficient_bf(c/wav, n)*10**17 for wav in wav_array], label=f'n={n}')
# plt.xlabel('Wavelength (Angstroms)')
# plt.ylabel('Absorption Coefficient (10^-17 cm^2/neutral H absorber)')
# plt.title('Absorption Coefficient for Bound-Free Transitions in Hydrogen')
# plt.legend()
# plt.grid()
# plt.ylim(0,4)
# plt.show()

# # print the absorption coefficient for the 13.6 eV transition
# nu_0 = 13.598 * eV / h
# print(f"Absorption coefficient at 13.6 eV is A0 = {absorption_coefficient_bf(nu_0, 1) * 10**18:.2f} 10^-18 cm^-2 for neutral H absorber")

# The partition function for the hydrogen atom and other elements
def partition_function(T, fi):
    if fi==24.59 or fi==7.65:
        # for He and Mg we have to use a different partition function
        return 1/2
    else:
        Z=0
        for m in range(1,4):
            zj=2*m**2*exp(-fi*(1-1/m**2)*eV/(kb*T))
            Z+=zj
        return Z
    
# now we have to sum over all the n levels and take into acount the level population using boltzmann distribution
def k_bf_H(T, nu):
    total_absorption = 0
    for n in range(1, 6):  # summing over all n levels
        nu_0 = 13.598 * eV / h / n**2
        if nu >= nu_0:  # only consider transitions above the threshold
            g_n = 2 * n**2  # degeneracy of level n
            partition_Q = partition_function(T, 13.598)
            boltzmann_factor = g_n * exp(-13.598*eV*(1-1/n**2)/(kb*T)) / partition_Q  # Boltzmann factor for level population
            total_absorption += absorption_coefficient_bf(nu, n) * boltzmann_factor
    return total_absorption

# Example usage
# T = 20000  # Temperature in Kelvin

# absorption_array = np.array([k_bf_H(T,c/wav) for wav in wav_array])
# plt.figure(figsize=(10, 6))
# plt.plot(wav_array*10**8, absorption_array*10**17, label='Total Absorption Coefficient', color='green')
# plt.xlabel('Wavelength (Angstroms)')
# plt.ylabel('Absorption Coefficient (10^-17 cm^2) per neutral H')
# plt.title('Total Absorption Coefficient for Bound-Free Transitions in Hydrogen')
# plt.legend()    
# plt.grid()
# plt.show()

