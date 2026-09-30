# test for astrophysics of stars
from units_astro import *
from numpy import pi, sqrt
hbar=h/(2*pi)
T=10**7

M=(4*pi)**(-1/2)/3*(kb*T*10/(G*mH))**(3/2)*(0.5)**(1/2)*mH**(-1/2)*(5*me*kb*T/(hbar**2*(3*pi)**(3/2)))**(-3/4)

print("Chandrasekhar mass limit for white dwarfs: ", M/Msun, " solar masses")

#degeneracy condition
D=hbar**2/(3*me*kb)*(3*pi**2/mH*0.5)**(2/3)
print("Density above which the electron degeneracy pressure becomes significant: ", D, " K cm^2 g^(−2/3)")

Dsun=1.570 * 10**7/(1.5527*10**5)**(2/3)
print("Degeneracy parameter for the center of the Sun ", Dsun, " K cm^2 g^(−2/3)")

DSiriusB=7.6*10**7/(3.236*10**9)**(2/3)
print("Degeneracy parameter for the center of Sirius B ", DSiriusB, " K cm^2 g^(−2/3)")

M_wd=(hbar*c/G)**(3/2)*(0.5/mH)**2*3*sqrt(2*pi)/8
print("Chandrasekhar mass limit for relativistic white dwarfs: ", M_wd/Msun, " solar masses")