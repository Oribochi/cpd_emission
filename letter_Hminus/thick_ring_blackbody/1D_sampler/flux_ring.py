# We are going to make a simple model of a ring at 1 au and leave the width of the ring as a free parameter
# we asume a blackbody emission of a completely thick ring and choose priors for the temperature 
# first 22 to 50 and second 27 to 50
from units_astro import *
from numpy import pi, exp
import numpy as np


# the plank function
def plank(T, nu):
    x=h*nu/(kb*T)
    if x<1e-3: # low frequency limit
        return 2*nu**2/c**2*kb*T
    elif x>1e3: # high frequency limit
        return 2*h*nu**3/c**2*exp(-x)
    else:
        return 2*h*nu**3/c**2/(exp(x)-1)
    

# the flux of the source depending on the frequency, the radius of the ring,
# the width of the ring, the temperature and the distance to the source
def F(nu, R_ring, width, T, d):
    R_inner = R_ring - width/2
    R_outer = R_ring + width/2

    # from the radiative transfer equation the solution is 

    flux = pi*(R_outer**2 - R_inner**2)/d**2*plank(T, nu)*1e26 # this was in erg/s/cm^2/Hz/ster we pass to mJy
    return flux

# # example of usage
# flux_345=F(345e9, 1*au, 0.2*au, 27, 112.3*pc)

# print("The flux of the ring at 345 GHz is: ", flux_345, "mJy")
# print centrifugal radius in au 
rc = 2645*Rj

print("The centrifugal radius is: ", rc/au, "au")
# and the width of the ring in au 
width=97*Rj

print("The width of the ring is: ", width/au, "au")