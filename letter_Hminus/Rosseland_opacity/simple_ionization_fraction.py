# Adding free free emission from the CPD
import numpy as np
from units_astro import *
from numpy import exp, pi, sqrt, log10
from scipy.optimize import fsolve
from matplotlib import pyplot as plt
from H_H2_ratio import H_H2_ratio

# From saha equation we can calculate the ionization fraction f = n+/n for each radius in the disk

# First for a given element of ionization potential fi (eV) we have the partition function for alkali metals and Hidrogen
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

# The Saha equation (Ni+1/Ni) for any element is given by:
def saha(T, n_e, fi):
    if n_e==0 or T==0:
        return 0
    else:
        return 2/partition_function(T, fi)*((2*pi*me*kb*T/h**2)**(3/2) * exp(-fi*eV/(kb*T))/n_e)

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

mu=1 # mean molecular weight

# For metals the real abundance is then given by the product of the abundance and 10^delta

# to have the total ion abundance we have to sum the ion abundance of each element
# this returns the abundance that are ions 
def total_ion_abundance(n_e, T):
    # since saha=ni+1/ni and ni is just the neutral number density we can solve
    # ni+1=n_tot*saha/(1+saha) with ntot the total number density of that element
    # for H 
    if n_e==0 or T==0:
        return 0
    else:
        n1=saha(T, n_e, fi1)*a1/(1+saha(T, n_e, fi1))
        n4=saha(T, n_e, fi4)*a4/(1+saha(T, n_e, fi4))
        # for metals we have to multiply by 10^delta to account for the depletion
        n11=saha(T, n_e, fi11)*a11/(1+saha(T, n_e, fi11))
        n12=saha(T, n_e, fi12)*a12/(1+saha(T, n_e, fi12))
        n19=saha(T, n_e, fi19)*a19/(1+saha(T, n_e, fi19))
        ni=n1+n4+n11+n12+n19
        return ni

def neutrality(log_ne, T, rho):
    n=rho/(mu*mH)
    ne=10.0**log_ne
    ni=total_ion_abundance(ne, T)
    # mi=averange_ion_mass(ne, n, T, a1, a4, a11, a12, a19, delta)
    # n_grain=n_grains(rho, zeta)
    # Z_grain_calculator=Z_grains_calculator(ne, ni, mi, T)
    # return the difference between the abundance of electrons and ions
    return ne/n-ni
    

# The ionization fraction is then given by the solution of the equation
def ionization_fraction(T, rho):
    # the total number density is given by the gas density
    log_ne=fsolve(neutrality, 30, args=(T, rho))
    n_e=10.0**log_ne
    # the ionization fraction is then given by
    n=rho/(mu*mH)
    ftot=n_e/n
    return ftot[0]

# now if we asumme all the electrons comes from only the most abundant element that is ionized
def simple_ionization_fraction(T, rho):
    n=rho/(mu*mH)
    ne1=(sqrt(saha(T,1,fi1)**2+4*saha(T,1,fi1)*n*a1)-saha(T,1,fi1))/2
    ne4=(sqrt(saha(T,1,fi4)**2+4*saha(T,1,fi4)*n*a4)-saha(T,1,fi4))/2
    ne11=(sqrt(saha(T,1,fi11)**2+4*saha(T,1,fi11)*n*a11)-saha(T,1,fi11))/2
    ne12=(sqrt(saha(T,1,fi12)**2+4*saha(T,1,fi12)*n*a12)-saha(T,1,fi12))/2
    ne19=(sqrt(saha(T,1,fi19)**2+4*saha(T,1,fi19)*n*a19)-saha(T,1,fi19))/2
    ne=max([ne1, ne4, ne11, ne12, ne19])
    return ne/n

# T_array=np.logspace(3, 5, 100)
# ionization_array=np.zeros(len(T_array))
# ionization_array_simple=np.zeros(len(T_array))
# saha_array=np.zeros(len(T_array))
# rho=10**(-9)
# for i in range(len(T_array)):
#     ionization_array[i]=ionization_fraction(T_array[i], rho)
#     ionization_array_simple[i]=simple_ionization_fraction(T_array[i], rho)
#     saha_array[i]=saha(T_array[i], rho, fi1)

# plt.plot(T_array, ionization_array, label='ionization fraction')
# plt.plot(T_array, ionization_array_simple, label='ionization fraction simple')
# plt.loglog()
# plt.legend()
# plt.show()

# print('max ionization fraction:', np.max(ionization_array))
# print('max ionization fraction simple:', np.max(ionization_array_simple))


# Now adding the H_minus
# this is the total number density of one element asuming all is atomic gas
def ion_abundance(n_e, T, a, f):
    ni=saha(T, n_e, f)*a/(1+saha(T, n_e, f))
    return ni

saha_const=3/2*log10(2*pi*me*kb/h**2) # a constant in Saha equation

# this is the number density of H- in cm^-3
def saha_H_minus(T, n_e):
    if n_e==0 or T==0:
        return 0
    else:
    # returns the logarithm of the Saha equation for H-
    # n_e is the electron density
    # T is the temperature
        log_saha_H_minus = log10(4)-log10(n_e)+saha_const-5036.36/T*0.754195+3/2*log10(T)
        # this is log(N(H)/N(H-)) so i return log(N(H-)/N(H)) multiplying by -1
        return -log_saha_H_minus

# now taking into acount the abundance of H- we have that some hydrogen is molecular so we have to calculate the atomic hydrogen abundance
def H_minus_abundance(n_e, T):
    # this is the abundance of H- in cm^-3 with respect the neutral hydrogen
    n_H_minus=10**saha_H_minus(T, n_e)/(1+10**saha_H_minus(T, n_e))
    return n_H_minus

def neutrality_H_minus(log_ne, T, rho):
    n=rho/(mu*mH)
    ne=10.0**log_ne
    # now we have to take into account the H- abundance with the atomic hydrogen abundance
    Pressure=n*kb*T
    n_H_rel, n_H2_rel=H_H2_ratio(T, Pressure) # relative abundances
    a1_atomic=n_H_rel/(n_H_rel+2*n_H2_rel)*a1 # abundance of atomic hydrogen
    ni=total_ion_abundance(ne, T)-H_minus_abundance(ne, T)*a1_atomic # this has the H- contribution and the ion contribution
    # return the difference between the abundance of electrons and ions
    return ne/n-ni
    

# The ionization fraction is then given by the solution of the equation
def ionization_fraction_H_minus(T, rho):
    # the total number density is given by the gas density
    log_ne=fsolve(neutrality_H_minus, 30, args=(T, rho))
    n_e=10.0**log_ne
    # the ionization fraction is then given by
    n=rho/(mu*mH)
    ftot=n_e/n
    return ftot[0]


# now if we asumme all the electrons comes from only the most abundant element that is ionized
def simple_ionization_fraction_H_minus(T, rho):
    n=rho/(mu*mH)
    ne1=(sqrt(saha(T,1,fi1)**2+4*saha(T,1,fi1)*n*a1)-saha(T,1,fi1))/2
    ne4=(sqrt(saha(T,1,fi4)**2+4*saha(T,1,fi4)*n*a4)-saha(T,1,fi4))/2
    ne11=(sqrt(saha(T,1,fi11)**2+4*saha(T,1,fi11)*n*a11)-saha(T,1,fi11))/2
    ne12=(sqrt(saha(T,1,fi12)**2+4*saha(T,1,fi12)*n*a12)-saha(T,1,fi12))/2
    ne19=(sqrt(saha(T,1,fi19)**2+4*saha(T,1,fi19)*n*a19)-saha(T,1,fi19))/2
    ne=max([ne1, ne4, ne11, ne12, ne19])

    if ne==0 or T==0:
        return 0
       
    else:
        Pressure=n*kb*T
        n_H_rel, n_H2_rel=H_H2_ratio(T, Pressure) # relative abundances
        a1_atomic=n_H_rel/(n_H_rel+2*n_H2_rel)*a1 # abundance of atomic hydrogen
        nH_minus=H_minus_abundance(ne, T)*n*a1_atomic # number density of H- in cm^-3

        # print('nH_minus/ne', nH_minus/ne)
        if nH_minus/ne<0.1:
            # if the H- abundance is less than 1% of the electron abundance we can assume that most of the electrons are free
            return ne/n
        else: # we have to solve the equation
            # print('The H- abundance is too high')
            # return ionization_fraction_H_minus(T, rho)
            return 0