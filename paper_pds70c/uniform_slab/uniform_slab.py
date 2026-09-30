# Making a simple model of free-free emission from a uniform slab with a solid angle 4pi*r^2
import numpy as np
from units_astro import *
from numpy import exp, sqrt, pi, log
from k_ff_bf_Hminus_2021 import k_ff_Hminus, k_bf_Hminus
from k_ff_metals import k_ff_metals
from scipy.special import erf
import matplotlib.pyplot as plt
from scipy.integrate import quad

dis=112.3*pc
# From free_free_CPD.py
###############################################################################################

def plank(T, nu):
    x=h*nu/(kb*T)
    if x<1e-3: # low frequency limit
        return 2*nu**2/c**2*kb*T
    elif x>1e1: # high frequency limit
        return 2*h*nu**3/c**2*exp(-h*nu/(kb*T))
    else:
        return 2*h*nu**3/c**2/(exp(h*nu/(kb*T))-1)
###############################################################################################


# A uniform slab has constant temperature T and a constand emission measure so the tau can be calculated as
def tau_uniform_slab(T, EM, nu):
    return (k_ff_Hminus(T,nu)+k_bf_Hminus(T,nu))*EM*kb*T

# the flux received from a infinitesimal ring is (R must be on cm) 
def dS_nu_uniform_slab(R, T, tau, nu):
    B_nu=plank(T, nu)
    if tau>1000:
        return B_nu*2*pi*R/dis**2*10**26
    elif tau<1e-3:
        return B_nu*tau*2*pi*R/dis**2*10**26
    else:
        return B_nu*(1-exp(-tau))*2*pi*R/dis**2*10**26 #to mJy

# The total flux R must be on cm
def F_uniform_slab(Rmax, T, EM, nu):
    tau=tau_uniform_slab(T, EM, nu)
    B_nu=plank(T, nu)
    if tau>1000:
        return B_nu*pi*Rmax**2/dis**2*10**26
    elif tau<1e-3:
        return B_nu*tau*pi*Rmax**2/dis**2*10**26
    else:
        return B_nu*(1-exp(-tau))*pi*Rmax**2/dis**2*10**26 #to mJy
