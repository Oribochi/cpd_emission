# fit temperature at 34 au
from units_astro import *
from numpy import pi

d=112.3*pc
a=34*au
theta=a/d*(180/pi)*3600  # arcsec

print(f"theta = {theta:.2f} arcsec")

R0=0.6 # arcsec
T0=15
beta=-1.7
T1=T0*(0.5/R0)**beta
print(f"T = {T1:.2f} K at 0.5 arcsec")
T2=T1*(theta/0.5)**(-0.5)
print(f"T = {T2:.2f} K at {theta:.2f} arcsec")