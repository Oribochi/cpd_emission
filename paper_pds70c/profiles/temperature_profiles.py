# Temperature profiles of the magnetic disk and the HI disk
import numpy as np
import matplotlib.pyplot as plt

T_HI_array=np.loadtxt('profiles_HI.txt')[:,3]
T_mag_array=np.loadtxt('profiles.txt')[:,3]
R_array=np.loadtxt('profiles_HI.txt')[:,0]

# The figure
fig = plt.figure(layout='constrained', figsize=(4, 3), dpi=300)
ax1 = fig.add_subplot(111)
ax1.plot(R_array, T_HI_array, label=r'$\log_{10}(\alpha)=-2$', color='blue')
ax1.plot(R_array, T_mag_array, label='Magnetic disk', color='red')
ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_xlabel('R (au)', fontsize=12)
ax1.set_ylabel('T (K)', fontsize=12)
ax1.legend()
plt.savefig('temperature_profiles.png', dpi=300, bbox_inches='tight')