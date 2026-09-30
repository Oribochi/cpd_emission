# distance in arcsec of PDS70c
from units_astro import *
from numpy import pi, log


a=34*au # PDS70c distance to the star
d=112.3*pc # distance of PDS70 from us

theta=a/d*(180/pi)*3600 # in arcsec

print(f"Distance of PDS70c from the central star is: {theta:.2f} arcsec")

# the slope of temperature is

beta=1.714915-2*9.240756*1E-2

print(f"Beta is: {beta:.2f}")

# and the temperature at PDS70c is
T=1.560256*10*(theta/0.6)**(-beta)

print(f"Temperature at PDS70c is: {T:.2f} K")

beta=1.714915

print(f"Beta is: {beta:.2f}")

# and the temperature at PDS70c is
T2=1.560256*10*(0.5/0.6)**(-beta)

print(f"Temperature at 0.5 arcsec is: {T2:.2f} K")
# if the slope was 0.5 ad 0.52 arcsec
T3=T2*(theta/0.5)**(-0.5)
print(f"Temperature at PDS70c is: {T3:.2f} K")