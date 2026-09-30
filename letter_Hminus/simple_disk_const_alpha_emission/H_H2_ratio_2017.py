# H_H2_ratio_2017 using Appendix B of Berardo et al. (2017)
# first we define the mean molecular weight
import numpy as np
import matplotlib.pyplot as plt
from units_astro import *
from numpy import pi, exp, log, sqrt
from simple_ionization_fraction import a1, a4
from H_H2_ratio import H_H2_ratio

# first we define the mean molecular weight
def mean_molecular_weight(X_H2, Y):
    # X_H2 molecular fraction of hydrogen: (n_H2/(n_H + n_H2))
    # Y is the helium mass fraction usually 0.25
    mean_mu_reciprocal = (1-Y)/(1+X_H2) + Y/4
    return 1/mean_mu_reciprocal

# now we define the function that calculates dimentional equilibrium constant K_eq'=n_H2/(n_H)^2
def saha_molecular_hydrogen(T):
    # Saha equation for the reaction: H2 <-> 2H
    # returns K_eq' in cgs units (cm^3)
    n_QH2 = (2*pi*2*mH*kb*T/h**2)**(3/2)
    n_QH = (2*pi*mH*kb*T/h**2)**(3/2)
    chi_diss = 4.48*eV # dissociation energy of H2
    # the partition function z_rot can be aproximated using the rotational temperature
    theta_rot = 85.4 # K from (Hill 1986)
    z_rot = T/(2*theta_rot)
    K_eq = n_QH2/n_QH**2 * z_rot * exp(chi_diss/(kb*T))
    return K_eq

# The pressure of the gas is given by
def pressure_gas(n_tot, T, X_H2, Y):
    mu = mean_molecular_weight(X_H2, Y)
    return n_tot*kb*T/mu

#def molecular_fraction_H2(rho, T, Y):
    # K_eq = saha_molecular_hydrogen(T)
    # ntot_approx = a1*rho/mH # approximation of the total number density of hydrogen assuming all atomic
    # if K_eq*ntot_approx<1e-6 : # in the case of high temperatures where K_eq is 0
    #     return 0
    
    # if K_eq*ntot_approx>1e6 : # in the case of low temperatures where K_eq is very high
    #     return 1
    # # coefficients of the cubic equation a*n_H^3 + b*n_H^2 + c*n_H + d = 0
    # a = 1
    # b = 1/K_eq
    # c = - (a1*rho/mH*(1-Y/2)*1/(4*K_eq) - 1/(4*K_eq**2))
    # d = - (a1*rho/mH*(1-3*Y/4)*1/(4*K_eq**2))
    # # we use numpy to find the roots of the cubic equation
    # coeffs = [a, b, c, d]
    # roots = np.roots(coeffs)
    # # we only want the real positive root
    # real_roots = roots[np.isreal(roots)].real
    # positive_real_roots = real_roots[real_roots >= 0]
    # if len(positive_real_roots) == 0: # no positive real root found then there is no solution and we return 0
    #     return 0
    # n_H = np.max(positive_real_roots)
    # n_H2 = K_eq * n_H**2
    # X_H2 = n_H2 / (n_H + n_H2)
    # return X_H2

# we return the molecular fraction X_H2 = n_H2/(n_H + n_H2) 
def molecular_fraction_H2(rho, T, Y):

    K_eq = saha_molecular_hydrogen(T)
    ntot_approx = a1*rho/mH # approximation of the total number density of hydrogen assuming all atomic
    if K_eq*ntot_approx<1e-6: # in the case of high temperatures where K_eq is 0
        return 1, 0
    if K_eq*ntot_approx>1e6: # in the case of low temperatures where K_eq is very high
        return 0, 0.5
    # coefficients of the cubic equation a*n_H^3 + b*n_H^2 + c*n_H + d = 0
    a = 1
    b = 1/K_eq
    c = - (a1*rho/mH*(1-Y/2)*1/(4*K_eq) - 1/(4*K_eq**2))
    d = - (a1*rho/mH*(1-3*Y/4)*1/(4*K_eq**2))
    # we can calculate the real positive root 
    # transforming to the depressed cubic t^3 + pt + q = 0 with the substitution n_H = t - b/(3a)
    p=(3*a*c - b**2)/(3*a**2)
    q=(2*b**3 - 9*a*b*c + 27*a**2*d)/(27*a**3)
    discriminant=(q/2)**2 + (p/3)**3
    if discriminant>0: # one real root
        A = (-q/2 + sqrt(discriminant))**(1/3)
        B = (-q/2 - sqrt(discriminant))**(1/3)
        t = A + B
        n_H = t - b/(3*a)
        if n_H<0: # no positive root found
            return 1, 0
    elif discriminant==0: # all roots real and at least two are equal
        if q==0: # all three roots are equal
            n_H = -b/(3*a)
            if n_H<0: # no positive root found
                return 1, 0
        else: # one single and one double root
            A = (-q/2)**(1/3)
            t1 = 2*A
            t2 = -A
            n_H1 = t1 - b/(3*a)
            n_H2 = t2 - b/(3*a)
            positive_roots = [root for root in [n_H1, n_H2] if root >= 0]
            if len(positive_roots) == 0: # no positive root found
                return 1, 0
            n_H = max(positive_roots) # take the largest positive root
    else: # three distinct real roots
        r = sqrt(-p**3/27)
        phi = np.arccos(-q/(2*r))
        t1 = 2*(r**(1/3))*np.cos(phi/3)
        t2 = 2*(r**(1/3))*np.cos((phi + 2*pi)/3)
        t3 = 2*(r**(1/3))*np.cos((phi + 4*pi)/3)
        n_H1 = t1 - b/(3*a)
        n_H2 = t2 - b/(3*a)
        n_H3 = t3 - b/(3*a)
        positive_roots = [root for root in [n_H1, n_H2, n_H3] if root >= 0]
        if len(positive_roots) == 0: # no positive root found
            return 1, 0
        
        n_H = max(positive_roots) # take the largest positive root
        
    n_H2 = K_eq * n_H**2
    mu=mean_molecular_weight(n_H2/(n_H + n_H2), Y)
    n_H_relative=n_H/(rho/(mu*mH)*a1) # relative number density of H
    n_H2_relative=n_H2/(rho/(mu*mH)*a1) #
    #X_H2 = n_H2 / (n_H + n_H2)
    return n_H_relative, n_H2_relative

# # Example usage
# rho1=1e-8 # g/cm³
# Y=a4*4
# print('Using a helium mass fraction Y =', Y)
# rho1=1e-8 # g/cm³
# T_example=np.linspace(100, 6000, 100)
# X_H2_array = np.array([molecular_fraction_H2(rho1, T, Y) for T in T_example]).T
# mu_array = np.array([mean_molecular_weight(X_H2, Y) for X_H2 in X_H2_array])
# n_tot_H=a1*rho1/(mu_array*mH) # total number density of hydrogen in cm^-3 assuming an abundance of hydrogen a1
# H_rel=1-X_H2_array # relative number density of atomic hydrogen
# H2_rel=X_H2_array/2 # relative number density of molecular hydrogen

# rho1=1e-8 # g/cm³
# n0_tot=rho1/mH # total number density of hydrogen in cm^-3 assuming pure hydrogen gas
# T_example=np.linspace(100, 6000, 100)
# P_example3=n0_tot*kb*T_example  # ideal gas law


# n_H_rel_example3, n_H2_rel_example3 = np.array([H_H2_ratio(T, P) for T, P in zip(T_example, P_example3)]).T


# plt.figure()
# plt.plot(T_example, H_rel, label='ñ_H (ρ=1e-8 g/cm³)')
# plt.plot(T_example, H2_rel, label='ñ_H2 (ρ=1e-8 g/cm³)')
# plt.plot(T_example, n_H_rel_example3, label='ñ_H from old code (ρ=1e-8 g/cm³)', linestyle='--')
# plt.plot(T_example, n_H2_rel_example3, label='ñ_H2 from old code (ρ=1e-8 g/cm³)', linestyle='--')
# plt.xlabel("Temperature (K)")
# plt.ylabel('Number Density (cm$^{-3}$)/n_tot')
# plt.gca().invert_xaxis()
# plt.title("Relative Number Densities of H and H$_2$ vs Temperature at P=1 bar")
# plt.legend()
# plt.grid()
# plt.show()