import numpy as np
from numpy import pi, sqrt, log10, exp
from units_astro import *
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from free_free_CPD import *
import smplotlib
import matplotlib
matplotlib.rcParams['axes.linewidth'] = 1. # the default is too thin

# the following should avoid having the silly-looking "10^0, 10^1, 10^2",
#  the unnecessarily complicated and "noisy" versions of "1, 10, 100"!
matplotlib.rcParams['axes.formatter.min_exponent'] = 5
matplotlib.rcParams['axes.formatter.limits'] = (-4,4)
matplotlib.rcParams['axes.formatter.useoffset'] = False  # also usually better
plt.style.use('tableau-colorblind10')

def profiles_ff(Mpdot, alpha):
    # for each R return the ionization fraction, the temperature, the density number of electrons and the surface density Sigma
    R_arr=np.logspace(log10(1.0001*Rin/au), log10(Rout/au), 100)*au
    f_arr=np.zeros(len(R_arr))
    ne_arr_z=np.zeros(len(R_arr))
    sigma_arr=np.zeros(len(R_arr))
    ne_arr=np.zeros(len(R_arr))
    T_c_arr=np.zeros(len(R_arr))
    EM_arr=np.zeros(len(R_arr))
    z_arr_mid_f=np.zeros(len(R_arr))
    for i in range(len(R_arr)):
        T_c_arr[i]=T_c_approx(R_arr[i],Mpdot,alpha)
        # calculo la fraccion de ionización en el R dado para cada z
        z_arr=np.linspace(0,3*H(R_arr[i],Mpdot,alpha),len(R_arr))
        sigma_arr[i]=Sigma(R_arr[i],Mpdot,alpha)
        ne_arr[i]=n_e(T_c_arr[i], R_arr[i], 0, Mpdot, alpha)
        f_arr[i]=ne_arr[i]/n_H(rho(R_arr[i],0,Mpdot,alpha))
        EM_arr[i]=emission_measure(R_arr[i], Mpdot, alpha)

        for j in range(len(z_arr)):
            ne_arr_z[j]=n_e(T_c_arr[i], R_arr[i], z_arr[j], Mpdot, alpha)
            # we find where f_arr_z is less than 0.5*f_arr_z[0] and we take the z value
            #print(n_e(T_c_arr[i], R_arr[i], z_arr[j], Mpdot, alpha))
            if  ne_arr_z[j]<0.5*ne_arr[i]:
                z_arr_mid_f[i]=z_arr[j] # we normalize to the scale height
                break
    return R_arr, f_arr, T_c_arr, ne_arr, sigma_arr, EM_arr, z_arr_mid_f

Mpdot=10**(-6)*Mj/yr
alpha=10**(-6.478771752442208)

R_array, f_array, T_c_array, ne_array, Sigma_array, EM_array, z_array_mid_f = profiles_ff(Mpdot, alpha)

column_width=256.0748*0.0138888889
text_width=523.5307*0.0138888889

# the profiles
fig, ax1 = plt.subplots(2, 1, figsize=(4.8,5.2), sharex=True, dpi=300)
fig.subplots_adjust(hspace=0.0, wspace=0.2)

ax1[0].plot(R_array/au, log10(T_c_array), label=r'$\log\left(\frac{T_{c\rm }}{\rm K}\right)$', color='C0')
ax1[0].plot(R_array/au, log10(Sigma_array/100), label=r'$\log\left(\frac{10^{-2} \, \Sigma}{\rm g\,cm^{-2}}\right)$', color='C1')
ax1[0].set_ylim(2,5)
ax1[0].set_xscale('log')
ax1[0].legend(loc='best', frameon=True, framealpha=0.8, fontsize=14)
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

ax1[1].plot(R_array/au, log10(EM_array/(10**30*pc)), label=r'$\log\left(\frac{10^{-30}\,{\rm EM}}{{\rm cm^{-6} pc}}\right)$', color='C0')
ax1[1].plot(R_array/au, log10(f_array), label=r'$\log\left(f\right)$', color='C1')
ax1[1].plot(R_array/au, log10(z_array_mid_f/(au)), label=r'$\log\left(\frac{H_{\rm hwhm}}{\rm au}\right)$', color='C2')
ax1[1].set_ylim(-3.5,0.5)
ax1[1].set_xscale('log')
#ax1[1].set_yticks([-3.0,-2.0,-1.0,0.0])
ax1[1].legend(loc='best', frameon=True, framealpha=0.8, fontsize=14)
ax1[1].set_xlabel(r'${R}\,({\rm au})$')


# ax1[0].plot(R_array/au, T_c_array, label=r'T_c \rm (K)')
# ax1[0].set_xscale('log')
# ax1[0].set_yscale('log')
# ax1[0].set_ylabel(r'$T_c$ (K)', fontsize=8)
# ax1[0].tick_params(axis='both', which='minor', labelsize=6)
# ax1[0].tick_params(axis='both', which='major', labelsize=6)
# ax1[0].tick_params(direction='in', which='both')

# ax1[1].plot(R_array/au, Sigma_array, label=r'\Sigma')
# ax1[1].set_xscale('log')
# ax1[1].set_yscale('log')
# ax1[1].set_ylabel(r'Σ (g/cm²)', fontsize=8)
# ax1[1].tick_params(axis='both', which='minor', labelsize=6)
# ax1[1].tick_params(axis='both', which='major', labelsize=6)
# ax1[1].tick_params(direction='in', which='both')

# ax1[2].plot(R_array/au, EM_array/pc, label='EM (cm⁻⁶ pc)')
# ax1[2].set_xscale('log')
# ax1[2].set_yscale('log')
# ax1[2].set_ylim(1E-10*max(EM_array/pc), max(EM_array/pc)*5)
# ax1[2].set_ylabel('EM (cm⁻⁶ pc)', fontsize=8)
# ax1[2].tick_params(axis='both', which='minor', labelsize=6)
# ax1[2].tick_params(axis='both', which='major', labelsize=6)
# ax1[2].tick_params(direction='in', which='both')

# ax1[3].plot(R_array/au, f_array, label='f')
# ax1[3].set_xscale('log')
# ax1[3].set_yscale('log')
# ax1[3].set_ylim(1E-10*max(f_array), max(f_array)*5)
# ax1[3].set_ylabel('f', fontsize=8)
# ax1[3].tick_params(axis='both', which='minor', labelsize=6)
# ax1[3].tick_params(axis='both', which='major', labelsize=6)
# ax1[3].tick_params(direction='in', which='both')

# ax1[4].plot(R_array/au, z_arr_mid_f/au, label=r'ρ (g/cm³)')
# ax1[4].set_xscale('log')
# ax1[4].set_yscale('log')
# ax1[4].set_ylim((z_arr_mid_f[0]/au), max(z_arr_mid_f/au))
# ax1[4].set_ylabel(r'$H_{\rm hwhm}$ (au)', fontsize=8)
# ax1[4].tick_params(axis='both', which='minor', labelsize=6)
# ax1[4].tick_params(axis='both', which='major', labelsize=6)
# ax1[4].tick_params(direction='in', which='both')
# ax1[4].set_xlabel('R (au)', fontsize=8)

plt.savefig('profile_HII_fixed_Mpdot.png', bbox_inches='tight',dpi=300)
plt.show()
print('The maximun ionization fraction is', max(f_array))

# use txt.save to save the temperature profile
np.savetxt('profiles_HII_fixed_Mpdot.txt', np.column_stack((R_array/au, ne_array, f_array, T_c_array, Sigma_array, EM_array/pc, z_array_mid_f)), header='R (au) ne f T_c (K) Sigma (g/cm²) EM (cm⁻⁶ pc) z_arr_mid_f (au)')

