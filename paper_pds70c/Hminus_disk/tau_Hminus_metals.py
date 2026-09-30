# This is the code with all the functions to calculate the optical depth in which the maximun contribution to the flux comes from
from units_astro import *
import numpy as np
from numpy import pi, sqrt, log10, exp
from dustyCPD import Sigma, Rin, Rout, dis, mu
from magneticCPD import H, alpha_calculator, T_z
from simple_ionization_fraction import simple_ionization_fraction_H_minus, total_ion_abundance, a1
from k_ff_metals import k_ff_metals
from k_ff_bf_Hminus_2021 import k_ff_Hminus, k_bf_Hminus
import matplotlib.pyplot as plt
import smplotlib
import matplotlib
matplotlib.rcParams['axes.linewidth'] = 1. # the default is too thin

# the following should avoid having the silly-looking "10^0, 10^1, 10^2",
#  the unnecessarily complicated and "noisy" versions of "1, 10, 100"!
matplotlib.rcParams['axes.formatter.min_exponent'] = 5
matplotlib.rcParams['axes.formatter.limits'] = (-4,4)
matplotlib.rcParams['axes.formatter.useoffset'] = False  # also usually better
plt.style.use('tableau-colorblind10')

def plank(T, nu):
    x=h*nu/(kb*T)
    if x<1e-3: # low frequency limit
        return 2*nu**2/c**2*kb*T
    elif x>1e1: # high frequency limit
        return 2*h*nu**3/c**2*exp(-x)
    else:
        return 2*h*nu**3/c**2/(exp(x)-1)

###### Case with stratification ######

def find_1(arr):
    k=len(arr)-1
    j=0
    while j<k-1:
        p=(j+k)//2
        if arr[p]>1:
            k=p
        else:
            j=p
    # print(tau_menos, tau_mas)
    return j, k

def I_metals_Hminus_and_taus(R, nu, Mpdot, alpha):
    N=101
    h=H(R, Mpdot, alpha)
    z_arr=np.linspace(-3*h, 3*h, N)
    ne_arr=np.zeros(len(z_arr))
    n_ion_arr=np.zeros(len(z_arr))
    n_H_arr=np.zeros(len(z_arr))
    k_ff_metals_arr=np.zeros(len(z_arr))
    k_ff_bf_Hminus_arr=np.zeros(len(z_arr))
    tau_arr=np.zeros(len(z_arr))
    Temp_z_arr=np.zeros(len(z_arr))
    plank_arr=np.zeros(len(z_arr))
    tau_Hminus_arr=np.zeros(len(z_arr))
    tau_metals_arr=np.zeros(len(z_arr))
    k=0
    # we go from 3H to -3H
    for i in range(len(z_arr)):
        m=i
        Temp_z=T_z(z_arr[m], R, Mpdot, alpha)
        Temp_z_arr[m]=Temp_z
        k_ff_metals_arr[m]=k_ff_metals(Temp_z, nu)
        # this function receives the wavelength in Angstroms and the temperature in Kelvin
        k_ff_bf_Hminus_arr[m]=k_ff_Hminus(Temp_z, nu)+k_bf_Hminus(Temp_z, nu)
        # print('k_ff', k_ff_arr[m]) the kappa opacity is of the order of 10**(-27)
        rho=Sigma(R, Mpdot, alpha)/(sqrt(2*pi)*h)*exp(-z_arr[m]**2/(2*h**2))
        f=simple_ionization_fraction_H_minus(Temp_z, rho)
        # f=10**(-4)
        ntot=rho/(mu*mH) # the number density of atoms
        n_H_arr[m]=a1*ntot # in general almost all the Hydrogen is neutral.
        ne_arr[m]=f*ntot # the number density of electrons
        n_ion_arr[m]=total_ion_abundance(ne_arr[m], Temp_z)*ntot # the number density of ions
        # tau is the integral of ne**2*k_ff from z to 3H
        tau_metals_arr[m]=np.trapezoid(kb*Temp_z_arr[:m]*ne_arr[:m]*n_ion_arr[:m]*k_ff_metals_arr[:m], z_arr[:m])
        tau_Hminus_arr[m]=np.trapezoid(kb*Temp_z_arr[:m]*ne_arr[:m]*n_H_arr[:m]*k_ff_bf_Hminus_arr[:m], z_arr[:m])  # We asumme only H- contributes to the opacity
        tau_arr[m]=tau_Hminus_arr[m]+tau_metals_arr[m]
        plank_arr[m]=plank(Temp_z, nu)

        # in the case tau is too big we return to the non stratified case
        if tau_arr[m]>1000:
            k=m
            break
        
    tau_tot=tau_arr[-1]

    if k==0:
        integrate=np.array([plank_arr[i]*exp(-tau_tot+tau_arr[i]) for i in range(N)])*10**26
        return np.trapezoid(integrate, tau_arr), tau_metals_arr[-1], tau_Hminus_arr[-1]

    # for the case optically thinner
    #print(tau_arr[0], tau_arr[int(N/2)], tau_arr[-1])
    #print(ne_arr[0]**2*h, ne_arr[int(N/2)]**2*h, ne_arr[-1]**2*h)
    #print(k_ff_arr[0], k_ff_arr[int(N/2)], k_ff_arr[-1])
    # for the case optically thicker tau is too big and only the last value is important
    else:
        l, p = find_1(tau_arr[:k])
        z1=(z_arr[l]+z_arr[p])/2
        # z1=3*h
        plank_val=plank(T_z(z1, R, Mpdot, alpha), nu)

        return plank_val*10**26, 1000, 1000
    
# R_array=np.logspace(log10(1.001*Rin/au), log10(Rout/au), 50)*au
# I_ff_arr=np.zeros(len(R_array))
# for i in range(len(R_array)):
#     alpha=alpha_calculator(R_array[i], Mpdot, Bps)
#     I_ff_arr[i]=I_ff_metals(R_array[i], 100E9, Mpdot, alpha)
# plt.plot(R_array/au, I_ff_arr)
# plt.xscale('log')
# plt.yscale('log')
# plt.xlabel('R (au)')
# plt.ylabel('I_ff (W/m²/Hz/sr)')
# plt.ylim(1e-10, 1e17)
# plt.show()
    
# Mpdot=10**(-5.49)*Mj/yr
# Bps=600

def tau_maximun_contribution_Hminus(nu, Mpdot, Bps):
    # R_arr=np.logspace(log10(1.0001*Rin/au), log10(Rout/au), 100)*au
    # I_nu0_r=0
    # tau_metals_max=0
    # tau_Hminus_max=0
    # for r in R_arr:
    #     alpha=alpha_calculator(r, Mpdot, Bps)
    #     I_nu, tau_metals, tau_Hminus = I_metals_Hminus_and_taus(r, nu, Mpdot, alpha)
    #     if I_nu*r**2>I_nu0_r:
    #         I_nu0_r=I_nu*r**2
    #         tau_metals_max=tau_metals
    #         tau_Hminus_max=tau_Hminus
    #     else: # we stop when the intensity starts to decrease
    #         break
    r=0.05*au
    alpha=alpha_calculator(r, Mpdot, Bps)
    I_nu, tau_metals, tau_Hminus = I_metals_Hminus_and_taus(r, nu, Mpdot, alpha)
    tau_metals_max=tau_metals
    tau_Hminus_max=tau_Hminus
    return r, tau_metals_max, tau_Hminus_max

Mpdot=10**(-5.369475692902487)*Mj/yr
Bps=610.1779212189386

# example with our frequencies
nus = [97.5E9, 145005402197.7, 343.5E9, 671E9] # [nu1, nu2] are the x-axis values , 671E9

array_nus = np.logspace(log10(nus[0]), log10(nus[-1]), 100)
tau_metals_array = np.zeros(len(array_nus))
tau_Hminus_array = np.zeros(len(array_nus))
r=0
for i in range(len(array_nus)):
    r, tau_metals_array[i], tau_Hminus_array[i] = tau_maximun_contribution_Hminus(array_nus[i], Mpdot, Bps)
    print(r/au, tau_metals_array[i]+tau_Hminus_array[i])


plt.figure(figsize=(4.8,3.8), dpi=300)
plt.plot(array_nus/1E9, tau_Hminus_array, label=r'$\rm H^{-}$', color='C0')
plt.plot(array_nus/1E9, tau_metals_array, label='Metals', color='C1')
plt.plot(array_nus/1E9, tau_Hminus_array+tau_metals_array, label='Total', color='C2', linestyle='--')
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'$\nu$ [GHz]')
plt.ylabel(r'$\tau$')
plt.legend(loc='best')
plt.savefig('tau_Hminus_metals.png', bbox_inches='tight', dpi=300)
plt.show()
# nu_arr=np.logspace(log10(nus[0]), log10(nus[-1])+4, 100)
# fluxes_arr = np.zeros(len(nu_arr))
# for i in range(len(nu_arr)):
#     fluxes_arr[i]=F_metals_Hminus(nu_arr[i], Mpdot, Bps)


# plt.errorbar(np.array(nus)/10**9, Fnus, yerr=sFnus, fmt='o', label='Data', color='blue')
# plt.plot(np.array(nu_arr)/10**9, fluxes_arr, label='model', alpha=0.5, color='red')
# plt.xscale('log')
# plt.yscale('log')
# plt.xlabel(r'$\nu$ [GHz]', fontsize=12)
# plt.ylabel(r'$F_{\nu}$ [mJy]', fontsize=12)
# plt.ylim(1e-6, 1e3)
# plt.legend(loc='upper left')
# plt.savefig('fluxes_metals_Hminus.png')
# plt.show()