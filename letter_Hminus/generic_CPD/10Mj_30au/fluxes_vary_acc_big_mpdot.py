# generating the flux using the range of magnetic fields and accretion rates for the magnetospheric accretion model
from flux_gas_dust_mag_disk import F_nu
from dustyCPD import *
from units_astro import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

print('Planet mass =', Mp/Mj, 'Mjup')
print('Orbital distance =', a/au, 'AU')

Mpdot_arr=np.array([10**(-6), 10**(-5.5), 10**(-5)])*Mj/yr
Bps_range_arr=np.array([[150,600], [265,600], [472,600]]) # Gauss
zeta=0

# we calculate the flux in Bands 4, 7 and 9 for each Mpdot and Bps range
B4=145e9
B7=343.5e9
B9=671e9

# where does the flux originates from? at 671 GHz
# calculate the flux per annulus