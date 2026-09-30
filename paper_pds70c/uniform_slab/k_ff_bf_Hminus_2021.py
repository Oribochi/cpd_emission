# Calculator bound-free opacity of the H- using David F. Gray (2021) "The Observation and Analysis of Stellar Photospheres" 4th edition
import numpy as np
from units_astro import *
from numpy import exp, pi, sqrt, log10
import matplotlib.pyplot as plt
from k_ff_metals import k_ff_metals

# This function is the H- bound-free opacity of the H- in units of cm^2 per H- atom
# Here wavelengh unit is in Angstroms
# a0 = +0.1199654,
# a1 = -1.18267 x 10^-6 ,
# a2 = +2.64243 x 10^-7 ,
# a3 = -4.40524 x 10^-11 ,
# a4 = +3.23992 x 10^-15 ,
# a5 = -1.39568 x 10^-19 ,
# a6 = +2.78701 x 10^-24 ,
def a_bf(wav):
    Cn=[0.1199654, -1.18267e-6, 2.64243e-7, -4.40524e-11, 3.23992e-15, -1.39568e-19, 2.78701e-24]
    suma=0
    for i in range(7):
        suma+=Cn[i]*wav**i
    return suma*1.0e-17

# This function is the H- bound-free opacity of the H- in units of electron presure per number density of neutral hydrogen per cm⁻¹
def k_bf_Hminus(T, nu):
    wav=c/nu*1.0e8 # convert to Angstroms
    # wav in Angstroms
    if wav>1.6419*1.0e4: # 1.6419 um = 0.754 eV is the threshold
        return 0
    else:
        # T in Kelvin
        # k_bf in cm^2 per H- atom
        # k_bf = a_bf * (1.0 - exp(-h*c/(wav*k*T)))
        theta=5040/T
        return a_bf(wav)*4.158e-10*theta**(5/2)*10**(0.754*theta)

# Functions for the free-free H- opacity, wav in Angstroms
def f_0(wav):
    # -2.2763 - 1.6850 log λ + 0.76661 log2 λ - 0.0533464 log3 λ
    return -2.2763 - 1.6850*log10(wav) + 0.76661*log10(wav)**2 - 0.0533464*log10(wav)**3

def f_1(wav):
    # +15.2827 - 9.2846 log λ + 1.99381 log2 λ - 0.142631 log3 λ
    return 15.2827 - 9.2846*log10(wav) + 1.99381*log10(wav)**2 - 0.142631*log10(wav)**3

def f_2(wav):
    # -197.789 + 190.266 log λ - 67.9775 log2 λ + 10.6913 log3 λ - 0.625151 log4 λ
    return -197.789 + 190.266*log10(wav) - 67.9775*log10(wav)**2 + 10.6913*log10(wav)**3 - 0.625151*log10(wav)**4


# This function is the H- free-free opacity of the H- in units of electron presure per number density of neutral hydrogen per cm⁻¹
def k_ff_Hminus(T, nu):
    wav=c/nu*1.0e8 # convert to Angstroms
    # wav in Angstroms
    # T in Kelvin
    # rho in g/cm^3
    # k_ff in cm^2 per H- atom
    # k_ff = 1.0e-24 * (rho/1.0e-3) * (T/1.0e4)**(-7/2) * exp(-h*c/(wav*k*T))
    theta=5040/T
    return 1.0e-26*10**(f_0(wav) + f_1(wav)*log10(theta) + f_2(wav)*log10(theta)**2)

# # # Example
# k_bf_array=[]
# k_ff_array=[]
# k_tot=[]
# k_ff_metals_array=[]
# waves=np.logspace(3, 5, 100) # in Angstroms
# T=5040
# for wav in waves:
#     k_bf_array.append(k_bf_Hminus(T, c/wav*10**8))
#     k_ff_array.append(k_ff_Hminus(T, c/wav*10**8))
#     k_tot.append(k_bf_Hminus(T, c/wav*10**8)+k_ff_Hminus(T, c/wav*10**8))
#     k_ff_metals_array.append(k_ff_metals(T, c/wav*10**8))

# plt.plot(waves, k_bf_array)
# plt.plot(waves, k_ff_array)
# plt.plot(waves, k_tot)
# plt.plot(waves, k_ff_metals_array)
# plt.xlabel('Wavelength (Angstroms)')
# plt.ylabel('Opacities (cm^2/dynes)')
# plt.title('k_bf for H-')
# plt.legend(['k_bf_H-', 'k_ff_H-', 'k_tot_H-', 'k_ff_metals'])
# plt.loglog()
# plt.show()

# print([float(np.round((k_bf_Hminus(T, wav)+k_ff_Hminus(T, wav))*10**(26),2)) for wav in waves])

# print([float(np.round((k_bf_Hminus(T, wav))*10**(26),2)) for wav in waves])

# print([float(np.round((k_ff_Hminus(T, wav))*10**(26),2)) for wav in waves])

# print([float(np.round((k_ff_metals(T, c/wav*10**4))*10**(26),2)) for wav in waves])

# in log10
# print([float(np.round(log10(k_bf_Hminus(T, wav)+k_ff_Hminus(T, wav))+30,2)) for wav in waves])