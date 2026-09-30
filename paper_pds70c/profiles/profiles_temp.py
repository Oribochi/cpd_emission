import numpy as np
from numpy import pi, sqrt, log10, exp
from units_astro import *
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from simple_ionization_fraction import *
from dustyCPD import T_c, Sigma
from magneticCPD import *


def profiles_temp(Mpdot, Bps):
    # for each R return the ionization fraction, the temperature, the density number of electrons and the surface density Sigma
    R_arr=np.logspace(log10(1.0001*Rin/au), log10(Rout/au), 50)*au
    alpha_arr=np.zeros(len(R_arr))
    T_c_arr=np.zeros(len(R_arr))
    Sigma_arr=np.zeros(len(R_arr))
    for i in range(len(R_arr)):
        alpha_i=alpha_calculator(R_arr[i],Mpdot,Bps)
        alpha_arr[i]=alpha_i
        Sigma_i=Sigma(R_arr[i],Mpdot,alpha_i)
        Sigma_arr[i]=Sigma_i
        T_c_arr[i]=T_c(R_arr[i],Sigma_i,Mpdot)
        # calculo la fraccion de ionización en el R dado para cada z

    return R_arr, alpha_arr, T_c_arr, Sigma_arr

Mpdot=10**(-6)*Mj/yr
Bps=200
R_array, alpha_array, T_c_array, Sigma_array = profiles_temp(Mpdot, Bps)

Rt=R_truncation(Mpdot, Bps)

# the profiles
fig, ax1 = plt.subplots(3, 1, figsize=(6, 6), sharex=True)
fig.subplots_adjust(hspace=0.2, wspace=0.3)
fig.supxlabel('R (au)', fontsize=12)

ax1[0].plot(R_array/au, alpha_array, label=r'\alpha')
ax1[0].axvline(x=Rt/au, color='r', linestyle='--', label='Rt')
ax1[0].set_xscale('log')
ax1[0].set_yscale('log')
ax1[0].set_ylabel(r'α', fontsize=12)

ax1[1].plot(R_array/au, Sigma_array, label=r'\Sigma')
ax1[1].axvline(x=Rt/au, color='r', linestyle='--', label='Rt')
ax1[1].set_xscale('log')
ax1[1].set_yscale('log')
ax1[1].set_ylabel(r'Σ (g/cm²)', fontsize=12)

ax1[2].plot(R_array/au, T_c_array, label=r'T_c \rm (K)')
ax1[2].axvline(x=Rt/au, color='r', linestyle='--', label='Rt')
ax1[2].set_xscale('log')
ax1[2].set_yscale('log')
ax1[2].set_ylabel(r'$T_c$ (K)', fontsize=12)

plt.show()