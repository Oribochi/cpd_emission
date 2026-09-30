# generating the flux using the range of magnetic fields and accretion rates for the magnetospheric accretion model
from flux_gas_dust_mag_disk import F_nu
from dustyCPD import *
from units_astro import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

print('Planet mass =', Mp/Mj, 'Mjup')
print('Orbital distance =', a/au, 'AU')

Mpdot_arr=np.array([6.95e-07, 1e-6, 10**(-5.5), 10**(-5)])*Mj/yr
Bps_range_arr=np.array([[70,70], [84, 366], [150,600], [265,600]]) # Gauss
zeta=0

# we calculate the flux in Bands 4, 7 and 9 for each Mpdot and Bps range
B4=145e9
B7=343.5e9
B9=671e9

# calculate the flux from 50 to 1000 GHz
flux_min_Bps_arr=np.zeros((len(Mpdot_arr), 3))
flux_max_Bps_arr=np.zeros((len(Mpdot_arr), 3))
for i in range(len(Mpdot_arr)):
    Mpdot=Mpdot_arr[i]
    Bps_min=Bps_range_arr[i,0]
    Bps_max=Bps_range_arr[i,1]
    flux_min_Bps_arr[i,0]=F_nu(B4, Mpdot, Bps_min, zeta)
    flux_min_Bps_arr[i,1]=F_nu(B7, Mpdot, Bps_min, zeta)
    flux_min_Bps_arr[i,2]=F_nu(B9, Mpdot, Bps_min, zeta)
    flux_max_Bps_arr[i,0]=F_nu(B4, Mpdot, Bps_max, zeta)
    flux_max_Bps_arr[i,1]=F_nu(B7, Mpdot, Bps_max, zeta)
    flux_max_Bps_arr[i,2]=F_nu(B9, Mpdot, Bps_max, zeta)

# saving the fluxes in a text file
np.savetxt(f'fluxes_magnetospheric_accretion_{int(Mp/Mj)}Mj_{int(a/au)}au.txt', 
           np.column_stack((Mpdot_arr/Mj*yr, Bps_range_arr[:,0], Bps_range_arr[:,1], flux_min_Bps_arr, flux_max_Bps_arr)), 
           header='Mpdot(Mj/yr) Bps_min(G) Bps_max(G) flux_min_B4(mJy) flux_min_B7(mJy) flux_min_B9(mJy) flux_max_B4(mJy) flux_max_B7(mJy) flux_max_B9(mJy)')