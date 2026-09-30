# Adding free free emission from the CPD
import numpy as np
from units_astro import *
from numpy import exp, pi, sqrt, log10, min, max
from scipy.optimize import fsolve
from matplotlib import pyplot as plt
from H_H2_ratio import H_H2_ratio
from dustyCPD import mu
plt.style.use('tableau-colorblind10')


# From saha equation we can calculate the ionization fraction f = n+/n for each radius in the disk

# First for a given element of ionization potential fi (eV) we have the partition function for alkali metals and Hidrogen
def partition_function(T, fi):
    # this receives an array of temperatures and the ionization potential in eV
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
    mask1=(T==0) | (n_e==0)
    return np.where(mask1, 0, 2/partition_function(T, fi)*((2*pi*me*kb*T/h**2)**(3/2) * exp(-fi*eV/(kb*T))/n_e))


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
 

# For metals the real abundance is then given by the product of the abundance and 10^delta

# to have the total ion abundance we have to sum the ion abundance of each element
# this returns the abundance that are ions 
def total_ion_abundance(n_e, T):
    # since saha=ni+1/ni and ni is just the neutral number density we can solve
    # ni+1=n_tot*saha/(1+saha) with ntot the total number density of that element
    # for H 
    mask = (n_e==0) | (T==0)
    return np.where(mask, 0, saha(T, n_e, fi1)*a1/(1+saha(T, n_e, fi1))+saha(T, n_e, fi4)*a4/(1+saha(T, n_e, fi4))+saha(T, n_e, fi11)*a11/(1+saha(T, n_e, fi11))+saha(T, n_e, fi12)*a12/(1+saha(T, n_e, fi12))+saha(T, n_e, fi19)*a19/(1+saha(T, n_e, fi19)))


def neutrality(log_ne, T, rho):
    n=rho/(mu*mH)
    log_n=np.log10(n)
    ne=10.0**log_ne
    ni=total_ion_abundance(ne, T)
    log_ni=np.log10(ni)
    # I want to make 0 ne/n-ni this is equivalent to log10(ne)-log10(n)-log10(ni)
    return log_ne-log_n-log_ni

# The ionization fraction is then given by the solution of the equation
def ionization_fraction(T, rho):
    if np.isscalar(T):
        T=np.array([T])
        log_ne=fsolve(neutrality, 30, args=(T, rho))
        n_e=10.0**log_ne
        n=rho/(mu*mH)
        ftot=n_e/n
        return ftot[0]
    else:
        length_T=len(T)
        arr_initial_guess=np.full(length_T,30)
        log_ne=fsolve(neutrality, arr_initial_guess, args=(T, rho))
        n_e=10.0**log_ne
        n=rho/(mu*mH)
        ftot=n_e/n
        return ftot

# now if we asumme all the electrons comes from only the most abundant element that is ionized
def simple_ionization_fraction(T, rho):
    n=rho/(mu*mH)
    ne1=(sqrt(saha(T,1,fi1)**2+4*saha(T,1,fi1)*n*a1)-saha(T,1,fi1))/2
    ne4=(sqrt(saha(T,1,fi4)**2+4*saha(T,1,fi4)*n*a4)-saha(T,1,fi4))/2
    ne11=(sqrt(saha(T,1,fi11)**2+4*saha(T,1,fi11)*n*a11)-saha(T,1,fi11))/2
    ne12=(sqrt(saha(T,1,fi12)**2+4*saha(T,1,fi12)*n*a12)-saha(T,1,fi12))/2
    ne19=(sqrt(saha(T,1,fi19)**2+4*saha(T,1,fi19)*n*a19)-saha(T,1,fi19))/2
    ne=max(np.array([ne1, ne4, ne11, ne12, ne19]), axis=0)
    return ne/n

# Now adding the H_minus
# this is the total number density of one element asuming all is atomic gas
def ion_abundance(n_e, T, a, f):
    ni=saha(T, n_e, f)*a/(1+saha(T, n_e, f))
    return ni

# this is the number density of H- in cm^-3
def saha_H_minus(T, n_e):
    saha_const=3/2*log10(2*pi*me*kb/h**2) # a constant in Saha equation
    return np.where((n_e==0) | (T==0), 0, -(log10(4)-log10(n_e)+saha_const-5036.36/T*0.754195+3/2*log10(T)))


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
    n_H_rel, n_H2_rel=H_H2_ratio(T, Pressure).T # relative abundances
    a1_atomic=n_H_rel/(n_H_rel+2*n_H2_rel)*a1 # abundance of atomic hydrogen
    ni=total_ion_abundance(ne, T)-H_minus_abundance(ne, T)*a1_atomic # this has the H- contribution and the ion contribution
    # return the difference between the abundance of electrons and ions
    return ne/n-ni
    

# The ionization fraction is then given by the solution of the equation
def ionization_fraction_H_minus(T, rho):
    if np.isscalar(T):
        T=np.array([T])
        log_ne=fsolve(neutrality_H_minus, 30, args=(T, rho))
        n_e=10.0**log_ne
        n=rho/(mu*mH)
        ftot=n_e/n
        return ftot[0]
    else:
        length_T=len(T)
        arr_initial_guess=np.full(length_T,30)
        log_ne=fsolve(neutrality_H_minus, arr_initial_guess, args=(T, rho))
        n_e=10.0**log_ne
        n=rho/(mu*mH)
        ftot=n_e/n
        return ftot

# now if we asumme all the electrons comes from only the most abundant element that is ionized
def compute_nH_minus(T, n, ne):
    Pressure=n*kb*T
    n_H_rel, n_H2_rel=H_H2_ratio(T, Pressure).T # relative abundances
    a1_atomic=n_H_rel/(n_H_rel+2*n_H2_rel)*a1 # abundance of atomic hydrogen
    nH_minus=H_minus_abundance(ne, T)*n*a1_atomic # number density of H- in cm^-3
    return nH_minus

def simple_ionization_fraction_H_minus(T, rho):
    n=rho/(mu*mH)
    ne1=(sqrt(saha(T,1,fi1)**2+4*saha(T,1,fi1)*n*a1)-saha(T,1,fi1))/2
    ne4=(sqrt(saha(T,1,fi4)**2+4*saha(T,1,fi4)*n*a4)-saha(T,1,fi4))/2
    ne11=(sqrt(saha(T,1,fi11)**2+4*saha(T,1,fi11)*n*a11)-saha(T,1,fi11))/2
    ne12=(sqrt(saha(T,1,fi12)**2+4*saha(T,1,fi12)*n*a12)-saha(T,1,fi12))/2
    ne19=(sqrt(saha(T,1,fi19)**2+4*saha(T,1,fi19)*n*a19)-saha(T,1,fi19))/2
    ne=max([ne1, ne4, ne11, ne12, ne19], axis=0)

    mask1=(ne==0) | (T==0)
    nH_minus_arr=np.where(mask1, 0 , compute_nH_minus(T,n,ne))
    return np.where(nH_minus_arr/ne<0.1, ne/n, 0)

# T_array=np.logspace(3, 5, 300)
# rho=10**(-9)

# ionization_array=ionization_fraction_H_minus(T_array, rho)
# ionization_array_simple=simple_ionization_fraction_H_minus(T_array, rho)
# saha_array=saha(T_array, rho, fi1)

# plt.figure(figsize=(4.8,3.8))
# plt.plot(T_array, ionization_array, label='Ionization fraction')
# plt.plot(T_array, ionization_array_simple, label='Ionization fraction simple', linestyle='dashed')
# plt.xlabel('Temperature (K)', fontsize=14)
# plt.ylabel('Ionization fraction', fontsize=14)
# plt.loglog()
# plt.legend(fontsize=12)
# # make the ticks bigger 
# plt.xticks(fontsize=12)
# plt.yticks(fontsize=12)
# plt.savefig('ionization_fraction.pdf', dpi=300, bbox_inches='tight')
# plt.show()

# print('max ionization fraction:', np.max(ionization_array))
# print('max ionization fraction simple:', np.max(ionization_array_simple))
