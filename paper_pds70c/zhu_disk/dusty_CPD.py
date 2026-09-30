# This is the code with all the functions to calculate the flux and brightness temperature of the CPD model

# Replicating the Zhu 2018 model of the CPD
import numpy as np
from numpy import pi, sqrt, min, log10
from units_astro import *

# For the CPD model we need to define the following parameters:
# The stellar mass in Msun
Ms = 1.0*Msun
# The planet mass in Mjup
Mp = 1*Mj
# The planet radius en Rjup
Rp = Rj
# The planet distance to the star in AU
a = 20*au
# The planet temperature in K
Tp = 1000

# Mean molecular weight
#########################################
mu = 1 # we should make a range between 0.5 and 2.3 from the hot to the cold regions
#########################################

kappa_r = 10 # The kappa_r is the Rosseland mean opacity we choose 10 cm^2/g as a nominal value

# With this we can calculate the CPD inner and outer radii
Rin = 1*Rp
Rout = 1/3*(Mp/(3*Ms))**(1/3)*a # This is 1/3 of the Hill radius


# To avoid the singularity at the inner radius we can define R from 1.0001*Rin to Rout
R=np.logspace(log10(1.0001*Rin/au), log10(Rout/au), 500)
R=R*au

# Finally the distance to the source
dis=112.3*pc 

# we can also have the CPD angular size

######################################################################################################################
######################################################################################################################

# Now we can calculate the temperature of the CPD in different ways: midplane, effective and brightness temperature

# Now we are going to define the the disk effective temperature from the viscous theory
def T_eff(R,Mpdot):
    return (3*G*Mp*Mpdot*(1-sqrt(Rin/R))/(8*pi*sigma*R**3))**(1/4)

# # Now the bolometric luminosity of the protoplanet is given by the integral of the effective temperature over the surface of the CPD
def Lacc(Mpdot):
    # This is the irradiation luminosity
    return G*Mp*Mpdot/(2*Rin)

# The external temperature is given by the irradiation of the star and the ISM
def T_ext(R,Mpdot):
    Tirr = (Lacc(Mpdot)/(40*pi*sigma*R**2))**(1/4) # This is the irradiation temperature we assume that the disc receives 1/10th of 
    #the irradiation luminosity in the direction perpendicular to the disc surface. H/R=0.1
    TISM = 10 # This is the temperature of the ISM in K
    return (TISM**4+Tirr**4)**(1/4)

# This is the viscous heating dominated temperature
def T_visc(R, Sigma, Mpdot):
    return ((9*G*Mp*Mpdot*Sigma*kappa_r)/(128*pi*sigma*R**3)*(1-sqrt(Rin/R)))**(1/4)

# The temperature at the midplane can be calculated as funtion of R and Sigma the surface density
def T_c(R,Sigma,Mpdot): 
    Text = T_ext(R, Mpdot) # This is the external irradiation temperature
    Tvis = T_visc(R, Sigma, Mpdot) # This is the viscous heating temperature
    return (Text**4+Tvis**4)**(1/4)

######################################################################################################################
######################################################################################################################

def nu_ext(R,Mpdot,alpha):
    Omega = sqrt(G*Mp/R**3)
    c_s = sqrt(Rgas*T_ext(R,Mpdot)/(mu))
    nu = alpha*c_s**2/Omega
    return nu

def Sigma_visc(R,Mpdot,alpha):
    # This is the surface density from the viscous heating
    cont=2**(7/5)/3**(6/5)
    val1 = (sigma*G*Mp*Mpdot**3/(alpha**4*pi**3*kappa_r*R**3))**(1/5)
    val2 = (mu/Rgas)**(4/5)
    val3 = (1-sqrt(Rin/R))**(3/5)
    Sigma_visc = cont*val1*val2*val3
    return Sigma_visc

def Sigma_ext(R,Mpdot,alpha):
    # This is the surface density from the external irradiation
    Sigma_ext = 1/nu_ext(R,Mpdot,alpha)*Mpdot/(3*pi)
    return Sigma_ext

def Sigma(R,Mpdot,alpha):
    # We choose the surface density to be the smallest between the viscous heating and the external radiation temperature
    # Because lager T_c -> smaller Sigma
    Sig = min([Sigma_visc(R,Mpdot,alpha), Sigma_ext(R,Mpdot,alpha)])
    return Sig

######################################################################################################################
######################################################################################################################

# Now we can calculate the temperature at the midplane T_c a function of R
def T_c_approx(R,Mpdot,alpha):
    return T_c(R, Sigma(R,Mpdot,alpha), Mpdot)

# With this we can calculate the temperature of brighness depending on the wavelength (if optically thick or thin)
# this depends on the opacity, we We adopt the mass opacity from Andrews et al. (2012), κ mm = zeta x 3.4 × (0.87mm/λ) cm2 g−1
# with zeta the dust-to-gas ratio (we assume 0.01)
def kappa_mm(wavelength,zeta=0.01):
    # we should receive the wavelength in mm
    return zeta*3.4*(0.87/wavelength)

# With the optical depth 
def tau_mm(R,wavelength,Mpdot,alpha,zeta=0.01):
    return 1/2*kappa_mm(wavelength,zeta)*Sigma(R,Mpdot,alpha)

# And the brightness temperature then depending on tau_mm > 0.5 or < 0.5 is optically thick or thin
def T_b(wavelength,R,Mpdot,alpha,zeta=0.01):
    tau = tau_mm(R,wavelength,Mpdot,alpha,zeta)
    if tau > 0.5:
        # opticall thick we choose the temperature at tau_mm=1
        Tb = (3/8*kappa_r/kappa_mm(wavelength,zeta)*T_eff(R,Mpdot)**4+T_ext(R,Mpdot)**4)**(1/4)
    else:
        # optically thin we choose the temperature at the midplane T_c weighted by the optical depth
        Tb = Sigma(R,Mpdot,alpha)*kappa_mm(wavelength,zeta)*T_c_approx(R,Mpdot,alpha)
    return Tb

# Now the S is the flux density in erg/s/cm^2/Hz/ster
def S(wavelength,R,Mpdot,alpha,zeta=0.01):
    wav=wavelength*1e-1 # we need to convert to cm (wavelength in mm)
    # T_b receives the wavelength in mm
    return 2*kb*T_b(wavelength,R,Mpdot,alpha,zeta)/((wav)**2)*1e26 # this was in erg/s/cm^2/Hz/ster we pass to mJy

# Now we can calculate the total flux as the integral of the flux density over the disc in function of Mpdot and alpha
# The flux is in erg/s/cm^2/Hz
def F(wavelength,Mpdot,alpha,zeta=0.01):
    flux=[]
    # if it's a list of wavelengths we calculate the flux for each en
    try:
        for wav in wavelength:
            # Importante calcular para un r unico pues la funcion S depende de T_b que tiene un if i funciona de a un r
            flux.append(np.trapezoid([S(wav, r, Mpdot, alpha, zeta)*2*pi*r/dis**2 for r in R], R))
        return flux
            
    # if it's a single wavelength we calculate the flux for that wavelength
    except:

        return np.trapezoid([S(wavelength, r, Mpdot, alpha, zeta)*2*pi*r/dis**2 for r in R], R)
    
def S_thin(wavelength,R,gamma,zeta=0.01):
    # gamma is the ratio between Mpdot and alpha
    T_b_thin=kappa_mm(wavelength,zeta)*sqrt(G*Mp/(pi**2*R**3))*gamma*(Mj/yr)*(mu/Rgas)**(5/4)*(1-sqrt(Rin/R))**(1/2)
    S_thin=2*kb*T_b_thin/(wavelength*1e-1)**2*1e26 # this was in erg/s/cm^2/Hz/ster we pass to mJy
    return S_thin

def F_thin(wavelength,gamma,zeta=0.01):
    # gamma is the dimenssionless ratio between Mpdot and alpha
    flux=[]
    # if it's a list of wavelengths we calculate the flux for each en
    try:

        for wav in wavelength:
            # Importante calcular para un r unico pues la funcion S depende de T_b que tiene un if i funciona de a un r
            flux.append(np.trapezoid([S_thin(wav, r, gamma,zeta)*2*pi*r/dis**2 for r in R], R))
        return flux
            
    # if it's a single wavelength we calculate the flux for that wavelength
    except:

        return np.trapezoid([S_thin(wavelength, r, gamma, zeta)*2*pi*r/dis**2 for r in R], R)

######################################################################################################################
######################################################################################################################

# Adding the jet emission

def jet_1cm(Mpdot):
    Lbol=Lacc(Mpdot)
    flux_mJy=0.008*(Lbol/Lsun)**(0.6)*(1000*pc/dis)**2
    return flux_mJy # The flux in mJy

# Now a function that calculates the flux in other wavelenght using the spectral index of the jet
def jet_flux_lambda(spec_ind, wavelength, Mpdot):
    # we have the flux in cm and the wavelength is in mm
    flux_1cm=jet_1cm(Mpdot)
    flux_wavelength=(10/wavelength)**(spec_ind)*flux_1cm
    return flux_wavelength
