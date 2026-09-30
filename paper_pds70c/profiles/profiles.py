import numpy as np
from numpy import pi, sqrt, log10, exp
from units_astro import *
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from simple_ionization_fraction import *
from dustyCPD import *
from magneticCPD import *


def profiles_ff(Mpdot, Bps):
    # for each R return the ionization fraction, the temperature, the density number of electrons and the surface density Sigma
    R_arr=np.logspace(log10(1.0001*Rin/au), log10(Rout/au), 50)*au
    alpha_arr=np.zeros(len(R_arr))
    f_arr=np.zeros(len(R_arr))
    sigma_arr=np.zeros(len(R_arr))
    ne_arr=np.zeros(len(R_arr))
    T_c_arr=np.zeros(len(R_arr))
    EM_arr=np.zeros(len(R_arr))
    z_arr_mid_f=np.zeros(len(R_arr))
    for i in range(len(R_arr)):
        alpha=alpha_calculator(R_arr[i],Mpdot,Bps)
        alpha_arr[i]=alpha
        T_c_arr[i]=T_c_approx(R_arr[i],Mpdot,alpha)
        sigma_arr[i]=Sigma(R_arr[i],Mpdot,alpha)
        rho_0=sigma_arr[i]/(sqrt(2*pi)*H(R_arr[i],Mpdot,alpha))
        f_arr[i]=ionization_fraction(T_c_arr[i], rho_0)
        ne_arr[i]=f_arr[i]*rho_0/(mu*mH)
        EM_arr[i]=EM(R_arr[i], Mpdot, alpha)

        # calculo la fraccion de ionización en el R dado para cada z
        z_arr=np.linspace(0,3*H(R_arr[i],Mpdot,alpha),len(R_arr))
        for j in range(len(z_arr)):
            Tz=T_z(z_arr[i],R_arr[i],Mpdot,alpha)
            rho_z=rho_0*exp(-z_arr[j]**2/(2*H(R_arr[i],Mpdot,alpha)**2))
            ne_arr_z=ionization_fraction(Tz, rho_z)*rho_z/(mu*mH)
            # we find where f_arr_z is less than 0.5*f_arr_z[0] and we take the z value
            #print(n_e(T_c_arr[i], R_arr[i], z_arr[j], Mpdot, alpha))
            if  ne_arr_z<0.5*ne_arr[i]:
                z_arr_mid_f[i]=z_arr[j] # we normalize to the scale height
                break
    return R_arr, alpha_arr, f_arr, T_c_arr, sigma_arr, EM_arr, z_arr_mid_f

Mpdot=10**(-5.49)*Mj/yr
Bps=10**(-2.5)
R_array, alpha_array, f_array, T_c_array, Sigma_array, EM_array, z_arr_mid_f = profiles_ff(Mpdot, Bps)

Rt=R_truncation(Mpdot, Bps)

# the profiles
fig, ax1 = plt.subplots(2, 3, figsize=(12, 6), sharex=True)
fig.subplots_adjust(hspace=0.2, wspace=0.3)
fig.supxlabel('R (au)', fontsize=12)

ax1[0,0].plot(R_array/au, alpha_array, label=r'\alpha')
ax1[0,0].axvline(x=Rt/au, color='r', linestyle='--', label='Rt')
ax1[0,0].set_xscale('log')
ax1[0,0].set_yscale('log')
ax1[0,0].set_ylabel(r'α', fontsize=12)

ax1[1,0].plot(R_array/au, Sigma_array, label=r'\Sigma')
ax1[1,0].axvline(x=Rt/au, color='r', linestyle='--', label='Rt')
ax1[1,0].set_xscale('log')
ax1[1,0].set_yscale('log')
ax1[1,0].set_ylabel(r'Σ (g/cm²)', fontsize=12)

ax1[0,1].plot(R_array/au, T_c_array, label=r'T_c \rm (K)')
ax1[0,1].axvline(x=Rt/au, color='r', linestyle='--', label='Rt')
ax1[0,1].set_xscale('log')
ax1[0,1].set_yscale('log')
ax1[0,1].set_ylabel(r'$T_c$ (K)', fontsize=12)

ax1[1,1].plot(R_array/au, EM_array/pc, label='EM (cm⁻⁶ pc)')
ax1[1,1].axvline(x=Rt/au, color='r', linestyle='--', label='Rt')
ax1[1,1].set_xscale('log')
ax1[1,1].set_yscale('log')
ax1[1,1].set_ylim(1E-10*max(EM_array/pc), max(EM_array/pc))
ax1[1,1].set_ylabel('EM (cm⁻⁶ pc)', fontsize=12)

ax1[0,2].plot(R_array/au, f_array, label='f')
ax1[0,2].axvline(x=Rt/au, color='r', linestyle='--', label='Rt')
ax1[0,2].set_xscale('log')
ax1[0,2].set_yscale('log')
ax1[0,2].set_ylim(1E-10*max(f_array), max(f_array))
ax1[0,2].set_ylabel('f', fontsize=12)

ax1[1,2].plot(R_array/au, z_arr_mid_f, label=r'ρ (g/cm³)')
ax1[1,2].axvline(x=Rt/au, color='r', linestyle='--', label='Rt')
ax1[1,2].set_xscale('log')
ax1[1,2].set_yscale('log')
ax1[1,2].set_ylim(1E-10*max(z_arr_mid_f), max(z_arr_mid_f))
ax1[1,2].set_ylabel(r'$n_e$ (cm⁻³)', fontsize=12)

plt.show()
print('The maximun ionization fraction is', max(f_array))

# use txt.save to save the temperature profile
np.savetxt('profiles.txt', np.column_stack((R_array/au, alpha_array, f_array, T_c_array, Sigma_array, EM_array/pc, z_arr_mid_f)), header='R (au) alpha f T_c (K) Sigma (g/cm²) EM (cm⁻⁶ pc) z_arr_mid_f (au)')