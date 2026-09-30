# The opacity free-free of the H2+ atom using Gray 2021
from units_astro import *
from numpy import exp, log10, sqrt, pi, arctan
import numpy as np
import matplotlib.pyplot as plt
from simple_ionization_fraction import saha, simple_ionization_fraction_H_minus, mu, ionization_fraction_H_minus
# from astropy import constants as const
# from astropy import units as u

# The abundance of five key contributing elements (Lide 2004) hydrogen, helium, sodium, magnesium and potassium
a1=9.21e-1 # H
a4=7.84e-2 # He
a11=1.6e-6 # Na
a12=3.67e-5 # Mg
a19=9.87e-8 # K

# The ionization potential for each element
fi1=13.6 # H
fi4=24.59 # He
fi11=5.14 # Na
fi12=7.65 # Mg
fi19=4.34 # K

# The mass of each element
m1=1.67e-24 # H
m4=6.65e-24 # He
m11=3.82e-23 # Na
m12=2.82e-23 # Mg
m19=1.2e-22 # K


# From Gray 2021 Hydrogen molecules appear in large numbers in cool stars. The H2+ molecules continuous 
# Absorption Coefficient has been studied by Bates (1951, 1952), Buckingham et al. (1952), Bates et al. (1953),
# Matsushima (1964), and Stancil (1994). It is a signiﬁcant absorber in the UV, but quite
# generally is only a few percent of the H2 absorption for λ > 3800 Å.

# useful constant a0 the Bohr radius
hbar = h/(2*pi)
a0 = hbar**2/(me*qe**2)

# #now in cgs
# h = const.h.cgs
# a0 = const.a0.cgs
# c = const.c.cgs
# qe = const.e.esu
# constante=16*pi**4*a0**5*qe**2/(3*h*c)
# print(constante.cgs)

def func_u1(nu):
    wav=c/nu*1.0e8 # Convert frequency to wavelength (this is in angstroms)
    return - 54.0532 + 32.713*log10(wav) - 6.6699*log10(wav)**2 + 0.4574*log10(wav)**3

def cross_section1(nu):
    wav=c/nu*1.0e8 # Convert frequency to wavelength (this is in angstroms)
    return - 1040.54 + 1345.71*log10(wav) - 547.628*log10(wav)**2 + 71.9684*log10(wav)**3

# atomic cross section (is adimensional)
def atomic_cross_section(T, nu):
    u1=func_u1(nu)
    theta=5040/T
    return cross_section1(nu)*exp(-u1*theta)

# the absorption coefficient for H2+ in units of cm5 per neutral hydrogen atom per ionised hydrogen atom is
def k_ff_H2_plus(T, nu):
    const1 = 16*pi**4*a0**5*qe**2/(3*h*c)
    return const1*atomic_cross_section(T,nu)

# now the full version of Gray 2021 includes the transformation of N(H+)
def Phi(T,ne,fi):
    Pe=ne*kb*T
    return saha(T, ne, fi)*Pe # our saha function is the Phi/Pe in Gray 2021

def sum_elements(T, ne, fi, ai):
    Pe=ne*kb*T
    abundance_j=ai/a1 # Abundance is relative to hydrogen in Gray 2021 (Aj)
    return abundance_j*Phi(T, ne, fi)/(1+Phi(T, ne, fi)/(Pe))

# this opacity is in cm2 per neutral hydrogen atom
def k_ff_H2_gray(T,nu,ne):
    const2 = 16*pi**4*a0**5*qe**2/(3*h*c)/(5040*kb)
    atomic_cross = atomic_cross_section(T, nu)
    theta = 5040/T
    Pe=ne*kb*T
    Phi_H=Phi(T, ne, fi1) # 
    sum = sum_elements(T, ne, fi1, a1)+sum_elements(T, ne, fi4, a4)+sum_elements(T, ne, fi11, a11)+sum_elements(T, ne, fi12, a12)+sum_elements(T, ne, fi19, a19)
    n_H_plus=Phi_H/(1+Phi_H/Pe)/sum # this is the n(H+) in Gray 2021
    # print (Pe, = , sum/Pe*KT*n_H)
    # the sum of all contributions
    return const2*atomic_cross*theta*Pe*n_H_plus


# # Example using T=3000, nu=1e14, rho=1e-7
# T=3000
# nu_array=np.logspace(12, 15, 100)
# rho=1e-7
# ntot=rho/(mu*mH)
# ne=simple_ionization_fraction_H_minus(T, rho)*ntot # Careful this is just asumming all the free electrons come from a single ionized metal!!!
# ne_real=ionization_fraction_H_minus(T, rho)*ntot

# suma=sum_elements(T, ne, fi1, a1)+sum_elements(T, ne, fi4, a4)+sum_elements(T, ne, fi11, a11)+sum_elements(T, ne, fi12, a12)+sum_elements(T, ne, fi19, a19)
# print('Pe = ', ne*kb*T, ' sum/Pe*KT*n_H = ', suma/(ne*kb*T)*kb*T*ntot*a1, 'real_Pe = ', ne_real*kb*T)

# k_ff_H2_gray_array=k_ff_H2_gray(T, nu_array, ne)
# n_H_plus=saha(T, ne, fi1)/(1+saha(T, ne, fi1))*ntot*a1
# k_ff_H2_array=k_ff_H2_plus(T, nu_array)*n_H_plus

# # asuming most of the hydrogen is neutral ntot*a1~n_H_0
# plt.plot(nu_array, k_ff_H2_gray_array*ntot*a1, label='Gray 2021')
# plt.plot(nu_array, k_ff_H2_array*ntot*a1, label='H2+')
# plt.xlabel('Frequency (Hz)')
# plt.ylabel('Opacity (cm^2/g)')
# plt.legend()
# plt.xscale('log')
# plt.yscale('log')
# plt.title('Opacity of H2 and H2+')
# plt.show()
