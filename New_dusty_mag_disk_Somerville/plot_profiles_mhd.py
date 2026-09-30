# This plot the profiles for the CPD
import numpy as np
import matplotlib.pyplot as plt
from magneticCPD import R_truncation
from profiles_mhd import profiles
from units_astro import *
from numpy import log10
import matplotlib
matplotlib.rcParams['axes.linewidth'] = 1. # the default is too thin

# the following should avoid having the silly-looking "10^0, 10^1, 10^2",
#  the unnecessarily complicated and "noisy" versions of "1, 10, 100"!
matplotlib.rcParams['axes.formatter.min_exponent'] = 5
matplotlib.rcParams['axes.formatter.limits'] = (-4,4)
matplotlib.rcParams['axes.formatter.useoffset'] = False  # also usually better
plt.style.use('tableau-colorblind10')

Mpdot=10**(-5.607369895579998)*Mj/yr
Bps=10**(2.796625406817652) # G
dust_to_gas=10**(-10.067572749115545)

print('Using Mdot =', Mpdot/(Mj/yr), 'Mj/yr, Bps =', Bps, 'G', 'and zeta =', dust_to_gas)

R_array, alpha_array, Sigma_array, T_c_array, EM_array, f_array, z_array_mid_f = profiles(Mpdot,Bps)
Rt=R_truncation(Mpdot,Bps)

# the profiles
column_width=256.0748*0.0138888889
text_width=523.5307*0.0138888889

# the profiles
fig, ax1 = plt.subplots(3, 1, figsize=(4.8, 7.8), sharex=True, dpi=300)
fig.subplots_adjust(hspace=0.0, wspace=0.2)


ax1[0].plot(R_array/au, log10(T_c_array), label=r'$\log\left(\frac{T_{c\rm }}{\rm K}\right)$', color='C0')
ax1[0].axvline(x=Rt/au, color='C1', linestyle='--', label=r'$R_{\rm t}$')
ax1[0].set_ylim(1.8,3.8)
ax1[0].set_xscale('log')
ax1[0].legend(loc='best', frameon=True, framealpha=0.8, fontsize=14)



ax1[1].plot(R_array/au, log10(alpha_array), label=r'$\log(\alpha)$', color='C0')
ax1[1].plot(R_array/au, log10(Sigma_array/1000), label=r'$\log\left(\frac{10^{-3} \,\Sigma}{\rm g\,cm^{-2}}\right)$', color='C2')
ax1[1].axvline(x=Rt/au, color='C1', linestyle='--')
# make the ticks in -4, -2, 0 and 2
ax1[1].set_ylim(-6,2.5)
ax1[1].set_xscale('log')
#ax1[1].set_yticks([-5.0,-3.0,-1.0,1.0])
ax1[1].legend(loc='best', frameon=True, framealpha=0.8, fontsize=14)

# ax1[1].plot(R_array/au, log10(Sigma_array), label=r'\Sigma')
# ax1[1].axvline(x=Rt/au, color='r', linestyle='--', label='Rt')
# ax1[1].set_xscale('log')
# ax1[1].set_yscale('log')
# ax1[1].set_ylabel(r'Σ (g/cm²)', fontsize=8)
# ax1[1].tick_params(axis='both', which='minor', labelsize=6)
# ax1[1].tick_params(axis='both', which='major', labelsize=6)
# ax1[1].tick_params(direction='in', which='both')

# ax1[2].plot(R_array/au, T_c_array, label=r'T_c \rm (K)')
# ax1[2].axvline(x=Rt/au, color='r', linestyle='--', label='Rt')
# ax1[2].set_xscale('log')
# ax1[2].set_yscale('log')
# ax1[2].set_ylabel(r'$T_c$ (K)', fontsize=8)
# ax1[2].tick_params(axis='both', which='minor', labelsize=6)
# ax1[2].tick_params(axis='both', which='major', labelsize=6)
# ax1[2].tick_params(direction='in', which='both')

# replace 0 values in EM_array, f_array and z_array_mid_f with the minimum non-zero value
EM_array = np.copy(EM_array)
EM_array[EM_array == 0] = np.min(EM_array[EM_array > 0])*10**(-6)

f_array = np.copy(f_array)
f_array[f_array == 0] = np.min(f_array[f_array > 0])*10**(-6)

z_array_mid_f = np.copy(z_array_mid_f)
z_array_mid_f[z_array_mid_f == 0] = np.min(z_array_mid_f[z_array_mid_f > 0])*10**(-6)

ax1[2].plot(R_array/au, log10(EM_array/(10**8*pc)), label=r'$\log\left(\frac{10^{-8}\,{\rm EM}}{\rm cm^{-6} pc}\right)$', color='C0')
ax1[2].plot(R_array/au, log10(f_array/10**(-6)), label=r'$\log\left(\frac{f}{10^{-6}}\right)$', color='C2')
ax1[2].plot(R_array/au, log10(z_array_mid_f/(0.01*au)), label=r'$\log\left(\frac{ H_{\rm hwhm}}{10^{-2}\,\rm au}\right)$', color='C3')
ax1[2].set_ylim(-3.5,0.5)
ax1[2].set_xscale('log')
ax1[2].axvline(x=Rt/au, color='C1', linestyle='--')
# ax1[2].set_yticks([-3.0,-2.0,-1.0,0.0])
ax1[2].legend(loc='best', frameon=True, framealpha=0.8, fontsize=14)
ax1[2].set_xlabel(r'${R}({\rm au})$')

plt.savefig('MHD_disk_profiles.pdf', bbox_inches='tight', dpi=300)
plt.show()
# ax1[4].plot(R_array/au, f_array, label='f'