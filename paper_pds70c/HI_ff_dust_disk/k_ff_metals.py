# The opacity for metals 
from units_astro import *
from numpy import exp, log, sqrt, pi, where, zeros
import numpy as np
def gaunt_factor(nu, T):
    gaunt_constant=49546092.4114831
    value=log(gaunt_constant*nu**(-1))+1.5*log(T)
    # we use the gaunt factor of Oster 1961, and for very high frequencies or low temperatures set it to 1
    value = np.where(value<1, value, 1)
    return value

# Some constants that appear in the free free optical depth

k_ff_const=4*pi*qe**6/(3*sqrt(3)*me**2*h*c)

v_prom_const=sqrt(pi*kb/(2*me))

cte=k_ff_const/v_prom_const

# in units of electron presure per ion number density 
def k_ff_metals(T, nu):
    x=h*nu/(kb*T)
    mask1=x<1e-3
    mask2=x>1e3
    kappa_metals=zeros(np.shape(x))
    kappa_metals[mask1]=(cte*x[mask1]*T**(-1/2)*nu[mask1]**(-3)*gaunt_factor(nu[mask1],T))/(kb*T)
    kappa_metals[mask2]=(cte*T**(-1/2)*nu[mask2]**(-3)*gaunt_factor(nu[mask2],T))/(kb*T)
    mask3=(x>=1e-3) & (x<=1e3)
    kappa_metals[mask3]=(cte*(1-exp(-x[mask3]))*T**(-1/2)*nu[mask3]**(-3)*gaunt_factor(nu[mask3],T))/(kb*T)
    return kappa_metals  # in cm² per electron
 
 # for example for an array of temperatures and frequencies
T_arr=np.array([5000, 6000, 7000])
nu_arr=np.array([1e11, 1e12, 1e13])
k_ff_metals_arr=k_ff_metals(T_arr, nu_arr)
print('k_ff_metals (cm² per electron):', k_ff_metals_arr)