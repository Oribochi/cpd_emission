# The opacity for metals 
from units_astro import *
from numpy import exp, log, sqrt, pi
def gaunt_factor(nu, T):
    gaunt_constant=49546092.4114831
    value=log(gaunt_constant*nu**(-1))+1.5*log(T)
    # we use the gaunt factor of Oster 1961
    if value>1:
        return value   # for the low frequency limit
    else: 
        return 1  # for the high frequency limit

# Some constants that appear in the free free optical depth

k_ff_const=4*pi*qe**6/(3*sqrt(3)*me**2*h*c)

v_prom_const=sqrt(pi*kb/(2*me))

cte=k_ff_const/v_prom_const

# in units of electron presure per ion number density 
def k_ff_metals(T, nu):
    x=h*nu/(kb*T)
    # to make it in units of Pe*nion=kTne*nion
    if x<1e-3: # low frequency limit
        return (cte*x*T**(-1/2)*nu**(-3)*gaunt_factor(nu,T))/(kb*T)
    elif x>1e3: # high frequency limit
        return (cte*T**(-1/2)*nu**(-3)*gaunt_factor(nu,T))/(kb*T)
    else:
        return (cte*(1-exp(-x))*T**(-1/2)*nu**(-3)*gaunt_factor(nu,T))/(kb*T)
