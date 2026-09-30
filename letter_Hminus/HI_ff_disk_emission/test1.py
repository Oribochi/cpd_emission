from units_astro import *
from numpy import log10, pi

print(10**(-1.4558301479873652)*au/Rj)

print(log10(10**(12)/(3.8*10**8)))

# acretion rate
def acretion_rate(n0,mu,Rmax):
    v0=3000000 # cm/s
    print(mu*mH)
    mass_flux=v0*mu*mH*n0
    print('mass flux = ', mass_flux)
    area=2*pi*Rmax**2
    return mass_flux*area

print('acretion rate = ', acretion_rate(10**12,1.3451934164752763,10*Rj)/(Mj/yr))

print('acretion rate = ', acretion_rate(10**(14.42),1.3451934164752763,52*Rj)/(Mj/yr))