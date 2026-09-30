from units_astro import *
from numpy import pi

r=62*Rj
v=30*10**5 # cm/s

M=v**2*r/(2*G)

#print("Mass of the planet is", M, "g")
print("Mass of the planet is", M/Mj, "Mj")

def acretion_flux(mu,n_0,v_0):
    return mu*mH*n_0*v_0

def acretion_rate(mu,n_0,v_0,R):
    size=2*pi*R**2
    return acretion_flux(mu,n_0,v_0)*size

mu=2.25e-24/mH
print('mu =',mu)

n=1e15
#example
print('the acretion flux is ', acretion_flux(mu,n,v))
print('The acretion rate is', acretion_rate(mu,n,v,r)/(Mj/yr),'Mj/yr')

# the luminosity for this would be 
Rp=62*Rj

L=G*acretion_rate(mu,n,v,r)*M/Rp
print('The luminosity is', L, 'erg/s')
print('The luminosity is', L/Lsun, 'Lsun')