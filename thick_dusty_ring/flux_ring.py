# We are going to make a simple model of a ring at 1 au and leave the width of the ring as a free parameter
# we are going to choose a constant surface density and constant temperature of 22 K, 27K and 45 K
from units_astro import *
from k_dust import kappa_mm
from numpy import pi, exp
import numpy as np

# the optical depth of the ring given dust opacity and the surface density
def tau_ring(nu, Sigma):
    wav=c/(nu)*10 # in mm
    return Sigma*kappa_mm(wav)

# the plank function
def plank(T, nu):
    x=h*nu/(kb*T)
    if x<1e-3: # low frequency limit
        return 2*nu**2/c**2*kb*T
    elif x>1e1: # high frequency limit
        return 2*h*nu**3/c**2*exp(-x)
    else:
        return 2*h*nu**3/c**2/(exp(x)-1)
    

# the flux of the source depending on the frequency, the surface density, the radius of the ring,
# the width of the ring, the temperature and the distance to the source
def F(nu, Sigma, R_ring, width, T, d):
    tau = tau_ring(nu, Sigma)
    R_inner = R_ring - width/2
    R_outer = R_ring + width/2

    # from the radiative transfer equation the solution is 

    flux = pi*(R_outer**2 - R_inner**2)/d**2*(1 - exp(-tau))*plank(T, nu)*1e26 # this was in erg/s/cm^2/Hz/ster we pass to mJy
    return flux

# # example of usage
# flux_345=F(345e9, 10000, 1*au, 0.2*au, 27, 112.3*pc)

# print("The flux of the ring at 345 GHz is: ", flux_345, "mJy")