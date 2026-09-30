# The opacity free-free of the He- atom using Gray 2021
from units_astro import c
from numpy import log10


# He- opacityin units of cm2 per absorber (N(He⁻)) per electron pressure
def k_ff_He_minus(T,nu):
    theta = 5040/T
    wav = c/nu*1.0e8 # Convert frequency to wavelength (this is in angstroms)
    c0 = 9.66736 - 71.76242*theta + 105.29576*theta**2 - 56.49259*theta**3 + 10.69206*theta**4
    c1 = -10.50614 + 48.28802*theta - 70.43363*theta**2 + 37.80099*theta**3 - 7.15445*theta**4
    c2 = 2.74020 - 10.62144*theta + 15.50518*theta**2 - 8.33845*theta**3 + 1.57960*theta**4
    c3 = -0.19923 + 0.77485*theta - 1.13200*theta**2 + 0.60994*theta**3 - 0.11564*theta**4
    return 10**(c0 + c1*log10(wav) + c2*log10(wav)**2 + c3*log10(wav)**3)*1e-26