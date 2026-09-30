# example of dust disk emission from nus = [97.5E9, 145005402197.7, 343.5E9, 671E9] # [nu1, nu2] are the x-axis values , 671E9
import numpy as np
import matplotlib.pyplot as plt
from flux_dusty_disk import F
from units_astro import *

nus = np.array([97.5E9, 145.0054021977E9, 343.5E9, 671E9]) # [nu1, nu2] are the x-axis values
wavelength = 3E11/nus # in mm
Mpdot = 10**(-5.68)*Mj/yr # in Mjup/yr
alpha=1e-5
fluxes = F(wavelength,Mpdot,alpha,TISM=27,zeta=0.000001)

plt.plot(nus,fluxes,'o-')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Flux (mJy)')    
plt.title('Dust Disk Emission')
plt.grid()
plt.show()