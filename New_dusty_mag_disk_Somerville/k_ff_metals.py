# The opacity for metals 
from units_astro import *
from numpy import exp, log, sqrt, pi
import numpy as np

def gaunt_factor(nu, T):
    gaunt_constant=49546092.4114831
    value=log(gaunt_constant*nu**(-1))+1.5*log(T)
    # we use the gaunt factor of Oster 1961
    return np.where(value<1, 1, value)


# in units of electron presure per ion number density 
def k_ff_metals(T, nu):
    # Some constants that appear in the free free optical depth
    k_ff_const=4*pi*qe**6/(3*sqrt(3)*me**2*h*c)
    v_prom_const=sqrt(pi*kb/(2*me))
    cte=k_ff_const/v_prom_const

    x=h*nu/(kb*T)
    # to make it in units of Pe*nion=kTne*nion
    mask1=x<1e-3 # low frequency limit
    mask2=x>1e3 # high frequency limit

    result=np.zeros_like(x)
    result[mask1]=((cte*x*T**(-1/2)*nu**(-3)*gaunt_factor(nu,T))/(kb*T))[mask1]

    result[mask2]=((cte*T**(-1/2)*nu**(-3)*gaunt_factor(nu,T))/(kb*T))[mask2]
    mask3=~(mask1 | mask2) # intermediate values
    result[mask3]=((cte*(1-exp(-x))*T**(-1/2)*nu**(-3)*gaunt_factor(nu,T))/(kb*T))[mask3]
    return result