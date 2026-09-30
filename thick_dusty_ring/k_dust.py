# The flux of the dusty disk
from dustyCPD import *

# the opacity from dust at mm wavelengths
def kappa_mm(wavelength,zeta=0.01):
    # we should receive the wavelength in mm
    return zeta*3.4*(0.87/wavelength)

# With the optical depth 
def tau_mm(R,wavelength,Mpdot,alpha,TISM=27,zeta=0.01):
    return 1/2*kappa_mm(wavelength,zeta)*Sigma(R,Mpdot,alpha,TISM)