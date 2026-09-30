import numpy as np
import scipy as sp
import os
import sys
import astropy.units as u
import astropy.constants as const
from copy import deepcopy
from pprint import pprint



def Tbrightness(I_nu,nu):
    # input I_nu in erg/s/cm2/sr/Hz
    # input nu in Hz
    h_P=const.h.cgs.value
    k_B=const.k_B.cgs.value
    c_light=const.c.cgs.value
    Tb=h_P*nu/(k_B*np.log( 1. + (2. * h_P * nu**3  / (c_light**2 * I_nu))))
        
    return Tb

def Tbrightness_RJ(I_nu,nu):
    # input I_nu in erg/s/cm2/sr/Hz
    # input nu in Hz
    h_P=const.h.cgs.value
    k_B=const.k_B.cgs.value
    c_light=const.c.cgs.value
    
    Tb=  c_light**2 * I_nu / (2. * k_B  * nu**2)
        
    return Tb


bmaj=0.1 / 3600.
bmin=0.074 / 3600.
Inu = 25.3E-6  
nu=108E9

bmaj=0.09 / 3600.
bmin=0.063 / 3600.
Inu = 10.9E-6
nu=155E9


bmaj=0.054 / 3600.
bmin=0.045 / 3600.
Inu = (3.5*10**(-3))
nu=671E9

# Jy/beam -> CGS
omegabeam = (np.pi/(4.*np.log(2.))) * (np.pi/180.)**2 * (bmaj * bmin)
unitfactor = 1E-26 * 1E7 * 1E-4 / omegabeam
Inu *= unitfactor
print("Inu",Inu)
Tb=Tbrightness(Inu,nu)
print("Tb ",Tb)
Tb_RJ=Tbrightness_RJ(Inu,nu)
print("Tb_RJ ",Tb_RJ)