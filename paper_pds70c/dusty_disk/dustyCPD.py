# This is the code with all the functions to calculate the flux and brightness temperature of the CPD model

# Replicating the Zhu 2018 model of the CPD
import numpy as np
from numpy import pi, sqrt, min, log10, log
from units_astro import *

# For the CPD model we need to define the following parameters:
# The stellar mass in Msun
Ms = 1.0*Msun
# The planet mass in Mjup
Mp = 4*Mj
# The planet radius en Rjup
Rp = Rj
# The planet distance to the star in AU
a = 34*au
# The planet temperature in K
Tp = 1000

# Mean molecular weight
#########################################
mu = 1 # we should make a range between 0.5 and 2.3 from the hot to the cold regions
#########################################

kappa_r = 10 # The kappa_r is the Rosseland mean opacity we choose 10 cm^2/g as a nominal value we should change to the
# appropriate value for the region of the CPD see Bell 1997

# With this we can calculate the CPD inner and outer radii
Rin = 1*Rp
Rout = 1/3*(Mp/(3*Ms))**(1/3)*a # This is 1/3 of the Hill radius

# To avoid the singularity at the inner radius we can define R from 1.0001*Rin to Rout
R=np.logspace(log10(1.0001*Rin/au), log10(Rout/au), 50)
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
def T_ext(R,Mpdot, TISM=27):
    Tirr = (Lacc(Mpdot)/(40*pi*sigma*R**2))**(1/4) # This is the irradiation temperature we assume that the disc receives 1/10th of 
    #the irradiation luminosity in the direction perpendicular to the disc surface. H/R=0.1
    # This is the temperature of the ISM in K
    return (TISM**4+Tirr**4)**(1/4)

# This is the viscous heating dominated temperature
def T_visc(R, Sigma, Mpdot):
    return ((9*G*Mp*Mpdot*Sigma*kappa_r)/(128*pi*sigma*R**3)*(1-sqrt(Rin/R)))**(1/4)

# The temperature at the midplane can be calculated as funtion of R and Sigma the surface density
def T_c(R,Sigma,Mpdot, TISM=27): 
    Text = T_ext(R, Mpdot, TISM) # This is the external irradiation temperature
    Tvis = T_visc(R, Sigma, Mpdot) # This is the viscous heating temperature
    return (Text**4+Tvis**4)**(1/4)

######################################################################################################################
######################################################################################################################

def nu_ext(R,Mpdot,alpha,TISM=27):
    Omega = sqrt(G*Mp/R**3)
    c_s = sqrt(Rgas*T_ext(R,Mpdot,TISM)/(mu))
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

def Sigma_ext(R,Mpdot,alpha,TISM=27):
    # This is the surface density from the external irradiation
    Sigma_ext = 1/nu_ext(R,Mpdot,alpha,TISM)*Mpdot/(3*pi)
    return Sigma_ext

def Sigma(R,Mpdot,alpha,TISM=27):
    # We choose the surface density to be the smallest between the viscous heating and the external radiation temperature
    # Because lager T_c -> smaller Sigma
    Sig = min([Sigma_visc(R,Mpdot,alpha), Sigma_ext(R,Mpdot,alpha,TISM)])
    return Sig

# Now we can calculate the temperature at the midplane T_c a function of R
def T_c_approx(R,Mpdot,alpha,TISM=27):
    return T_c(R, Sigma(R,Mpdot,alpha,TISM), Mpdot, TISM)