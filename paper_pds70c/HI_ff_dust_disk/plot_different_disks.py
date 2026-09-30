# Calculate the fluxes for the MHD disk model with given parameters
import numpy as np
import matplotlib.pyplot as plt
from fluxes_params import fluxes_model, nu
from units_astro import *
import corner
from numpy import pi, log, log10
plt.style.use('tableau-colorblind10')

# The magnetic field force
Bps=600  # Gauss

# the dust to gas ratio can be 1e-2 to 1e-6 for comparison
dust_to_gas_ratio1=1e-6
dust_to_gas_ratio2=1e-5
dust_to_gas_ratio3=1e-4
dust_to_gas_ratio4=1e-3
dust_to_gas_ratio5=1e-2

# the acretion rate in Mjup/yr
Mpdot=10**(-5)*Mj/yr

# making the plots
flux_ff1, flux_dusty1, flux_tot1 = fluxes_model(Mpdot, Bps, dust_to_gas_ratio1)
flux_ff2, flux_dusty2, flux_tot2 = fluxes_model(Mpdot, Bps, dust_to_gas_ratio2)
flux_ff3, flux_dusty3, flux_tot3 = fluxes_model(Mpdot, Bps, dust_to_gas_ratio3)
flux_ff4, flux_dusty4, flux_tot4 = fluxes_model(Mpdot, Bps, dust_to_gas_ratio4)
flux_ff5, flux_dusty5, flux_tot5 = fluxes_model(Mpdot, Bps, dust_to_gas_ratio5)
# the total fluxes are:
flux1=flux_tot1
flux2=flux_tot2
flux3=flux_tot3
flux4=flux_tot4
flux5=flux_tot5

# now we can plot each case separating in dashed line when free-free dominates and a solid line when dust dominates
plt.figure(figsize=(8,6))
plt.loglog(nu[flux_ff1<=flux_dusty1]/1E9, flux1[flux_ff1<=flux_dusty1], label='dust-to-gas ratio = 1e-6', color='C0')
plt.loglog(nu/1E9, flux1, '--', color='C0')
plt.loglog(nu[flux_ff2<=flux_dusty2]/1E9, flux2[flux_ff2<=flux_dusty2], label='dust-to-gas ratio = 1e-5', color='C1')
plt.loglog(nu/1E9, flux2, '--', color='C1')
plt.loglog(nu[flux_ff3<=flux_dusty3]/1E9, flux3[flux_ff3<=flux_dusty3], label='dust-to-gas ratio = 1e-4', color='C2')
plt.loglog(nu/1E9, flux3, '--', color='C2')
plt.loglog(nu[flux_ff4<=flux_dusty4]/1E9, flux4[flux_ff4<=flux_dusty4], label='dust-to-gas ratio = 1e-3', color='C3')
plt.loglog(nu/1E9, flux4, '--', color='C3')
plt.loglog(nu[flux_ff5<=flux_dusty5]/1E9, flux5[flux_ff5<=flux_dusty5], label='dust-to-gas ratio = 1e-2', color='C4')
plt.loglog(nu/1E9, flux5, '--', color='C4')
plt.xlabel('Frequency (GHz)', fontsize=14)
plt.ylabel('Flux (mJy)', fontsize=14)
plt.title('MHD Disk Model Fluxes for Different Dust-to-Gas Ratios', fontsize=16)
plt.legend()
plt.grid(True, which="both", ls="--", linewidth=0.5)
plt.savefig('MHD_disk_fluxes_dust_to_gas_variation.pdf', dpi=300, bbox_inches='tight')
plt.show()