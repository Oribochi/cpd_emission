from units_astro import *
from numpy import pi, sqrt
# just trying some constants
hbar=h/(2*pi)
print('a0 =', hbar**2/(me*qe**2))
Rydberg=13.598*eV/(h*c)
print('Rydberg =', Rydberg)
alpha0=32/sqrt(3**3)*pi**2*qe**6/(h*c)**3*Rydberg
print('Constant alpha0 =', alpha0)

# now doing the same but with astropy constants
from astropy import constants as const
from astropy import units as u
hbar_astropy=const.hbar.cgs
print('a0 (astropy) =', (hbar_astropy**2/(const.m_e.cgs*const.e.esu**2)).cgs)
Rydberg_astropy=13.598*u.eV.to(u.erg)*1*u.erg/(const.h.cgs*const.c.cgs)
print('Rydberg (astropy) =', Rydberg_astropy.cgs)
alpha0_astropy=32/sqrt(3**3)*pi**2*const.e.esu**6/(const.h.cgs*const.c.cgs)**3*Rydberg_astropy
print('Constant alpha0 (astropy) =', alpha0_astropy.cgs)