# This is the code with all the functions to calculate the optical depth in which the maximun contribution to the flux comes from
from units_astro import *
import numpy as np
from numpy import pi, sqrt, log10, exp
from dustyCPD import Sigma, Rin, Rout, dis, mu
from magneticCPD import H, alpha_calculator, T_z
from simple_ionization_fraction import simple_ionization_fraction_H_minus, total_ion_abundance, a1, a4
from k_ff_metals import k_ff_metals
from k_ff_bf_Hminus_2021 import k_ff_Hminus, k_bf_Hminus
from k_bf_H_2021 import k_bf_H
from k_ff_H2_2021_minus import k_ff_H2_minus
from H_H2_ratio import H_H2_ratio
from k_ff_He_minus import k_ff_He_minus, He_minus_number_density
import matplotlib.pyplot as plt
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
    n_H2_arr=np.zeros(len(z_arr))
    n_He_arr=np.zeros(len(z_arr))
    k_ff_metals_arr=np.zeros(len(z_arr))
    k_ff_bf_Hminus_arr=np.zeros(len(z_arr))
    k_bf_H_arr=np.zeros(len(z_arr))
    k_ff_H2_minus_arr=np.zeros(len(z_arr))
    k_ff_He_minus_arr=np.zeros(len(z_arr))
    tau_arr=np.zeros(len(z_arr))
    Temp_z_arr=np.zeros(len(z_arr))
    plank_arr=np.zeros(len(z_arr))
    tau_Hminus_arr=np.zeros(len(z_arr))
    tau_metals_arr=np.zeros(len(z_arr))
    tau_H2_minus_arr=np.zeros(len(z_arr))
    tau_He_minus_arr=np.zeros(len(z_arr))
    k=0
    # we go fromals_max=0
    # tau_Hminus_max=0
    # for r in R_arr:
    #     alpha=alpha_calculator(r, Mpdot, Bps)
    #     I_nu, tau_metals, tau_Hminus = I_metals_Hminus_and_taus(r, nu, Mpdot, alpha)
    #     if 3H to -3H
    for i in range(len(z_arr)):
        m=i
        Temp_z=T_z(z_arr[m], R, Mpdot, alpha)
        Temp_z_arr[m]=Temp_z
        k_ff_metals_arr[m]=k_ff_metals(Temp_z, nu)
        # this function receives the wavelength in Angstroms and the temperature in Kelvin
        k_ff_bf_Hminus_arr[m]=k_ff_Hminus(Temp_z, nu)+k_bf_Hminus(Temp_z, nu)
        # print('k_ff', k_ff_arr[m]) the kappa opacity is of the order of 10**(-27)
        k_bf_H_arr[m]=k_bf_H(Temp_z, nu) # this opacity is in cm^2 per neutral H atom
        # The opacity of H2 minus
        k_ff_H2_minus_arr[m]=k_ff_H2_minus(Temp_z, nu)
        # The opacity of He minus
        k_ff_He_minus_arr[m]=k_ff_He_minus(Temp_z, nu)
        # now the density at this height
        rho=Sigma(R, Mpdot, alpha)/(sqrt(2*pi)*h)*exp(-z_arr[m]**2/(2*h**2))
        f=simple_ionization_fraction_H_minus(Temp_z, rho)
        # f=10**(-4)
        ntot=rho/(mu*mH) # the number density of atoms
        # obtaining the number densities of H2 and H
        n_H_arr[m], n_H2_arr[m]=np.array(H_H2_ratio(Temp_z, ntot*kb*Temp_z))*ntot*a1 # number density of H vs the H2 molecules by minimizing the Gibbs energy 
        # print('n_H', n_H_arr[m], 'n_H2', n_H2_arr[m])       
        ne_arr[m]=f*ntot # the number density of electrons
        n_ion_arr[m]=total_ion_abundance(ne_arr[m], Temp_z)*ntot # the number density of ions
        n_He_arr[m]=ntot*a4 # the number density of He- atoms
        # tau is the integral of ne**2*k_ff from z to 3H
        tau_metals_arr[m]=np.trapezoid(kb*Temp_z_arr[:m]*ne_arr[:m]*n_ion_arr[:m]*k_ff_metals_arr[:m], z_arr[:m])
        tau_Hminus_arr[m]=np.trapezoid(kb*Temp_z_arr[:m]*ne_arr[:m]*n_H_arr[:m]*k_ff_bf_Hminus_arr[:m], z_arr[:m])  # We asumme H- contributes to the opacity
        tau_H2_minus_arr[m]=np.trapezoid(kb*Temp_z_arr[:m]*ne_arr[:m]*n_H2_arr[:m]*k_ff_H2_minus_arr[:m], z_arr[:m])  # We asumme H2- contributes to the opacity
        tau_He_minus_arr[m]=np.trapezoid(kb*Temp_z_arr[:m]*ne_arr[:m]*n_He_arr[:m]*k_ff_He_minus_arr[:m], z_arr[:m])  # We asumme He- contributes to the opacity
        tau_arr[m]=tau_Hminus_arr[m]+tau_metals_arr[m]+tau_H2_minus_arr[m]+tau_He_minus_arr[m]
        plank_arr[m]=plank(Temp_z, nu)

        # in the case tau is too big we return to the non stratified case
        if tau_arr[m]>1000:
            k=m
            break
        
    tau_tot=tau_arr[-1]

    if k==0:
        integrate=np.array([plank_arr[i]*exp(-tau_tot+tau_arr[i]) for i in range(N)])*10**26
        return np.trapezoid(integrate, tau_arr), tau_metals_arr[-1], tau_Hminus_arr[-1], tau_H2_minus_arr[-1], tau_He_minus_arr[-1]

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

        return plank_val*10**26,  tau_metals_arr[k], tau_Hminus_arr[k], tau_H2_minus_arr[k], tau_He_minus_arr[k]
    
# R_array=np.logspace(log10(1.001*Rin/au), log10(Rout/au), 50)*au
# I_ff_arr=np.zeros(len(R_array))
# for i in range(len(R_array)):
#     alpha=alpha_calculator(R_ar9475692902487ray[i], Mpdot, Bps)
#     I_ff_arr[i]=I_ff_metals(R_array[i], 100E9, Mpdot, alpha)
# plt.plot(R_array/au, I_ff_arr)
# plt.xscale('log')
# plt.yscale('log')
    # for r in R_arr:
    #     alpha=alpha_calculator(r, Mpdot, Bps)
    #     I_nu, tau_metals, tau_Hminus = I_metals_Hminus_and_taus(r, nu, Mpdot, alpha)
    #     if I_nu*r**2>I_nu0_r:
    #         I_nu0_r=I_nu*r**2
    #         tau_metals_max=tau_metals
    #         tau_Hminus_max=tau_Hminus
    #     else: # we stop when the intensity starts to decrease
    #         break

Mpdot=10**(-5.62)*Mj/yr
Bps=600

# R_array=np.logspace(log10(1.001*Rin/au), log10(Rout/au), 100)*au
# I_ff_arr=np.zeros(len(R_array))
# for i in range(len(R_array)):
#     alpha=alpha_calculator(R_array[i], Mpdot, Bps)
#     I_ff_arr[i]=I_metals_Hminus_and_taus(R_array[i], 100E9, Mpdot, alpha)[0]
# plt.plot(R_array/au, I_ff_arr)
# plt.xscale('log')
# plt.yscale('log')
# plt.xlabel('R (au)')
# plt.ylabel('I_ff (W/m²/Hz/sr)')
# plt.ylim(1e-10, 1e17)
# plt.show()
def F_metals_Hminus(nu, Mpdot, Bps):
    R_arr=np.logspace(log10(1.0001*Rin/au), log10(Rout/au), 100)*au
    F_ff_arr=np.zeros(len(R_arr))
    for i in range(len(R_arr)):
        alpha=alpha_calculator(R_arr[i], Mpdot, Bps)
        F_ff_arr[i]=I_metals_Hminus_and_taus(R_arr[i], nu, Mpdot, alpha)[0]*2*pi*R_arr[i]/dis**2
    # where is the maximum contribution to the flux
    max_index=np.argmax(F_ff_arr)
    r_max=R_arr[max_index]
    print('The maximum contribution to the flux comes from R=', r_max/au, 'au')
    return np.trapezoid(F_ff_arr, R_arr)  # to mJy

# # First our data with the error bars
nus = [97.5E9, 145005402197.7, 343.5E9, 671E9] # [nu1, nu2] are the x-axis values , 671E9
Fnus = [12e-06, 21.4e-06, 0.000121, 0.000143] # [Fnu1, Fnu2] are the y-axis values in Jy , 0.000122
sFnus = [4.7e-06, 4.1e-06, 0.000013, 0.000115] # [sigma_Fnu1, sigma_Fnu2] are the y-axis errors in Jy , 0.000090

# To mJy
nu = np.array(nus)
Fnus = np.array(Fnus)*1E3
sFnus = np.array(sFnus)*1E3

fluxes = np.zeros(len(nu))
for i in range(len(nu)):
    fluxes[i]=F_metals_Hminus(nu[i], Mpdot, Bps)
    print('Flux at', round(nu[i]/1E9, 1), 'GHz =', fluxes[i], 'mJy')
    if i>0:
        print('Spectral index is ', (log10(fluxes[i])-log10(fluxes[i-1]))/(log10(nu[i])-log10(nu[i-1])))

plt.errorbar(np.array(nus)/10**9, Fnus, yerr=sFnus, fmt='o', label='Data', color='blue')
plt.plot(np.array(nus)/10**9, fluxes, label='model', alpha=0.5, color='red')
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'$\nu$ [GHz]', fontsize=12)
plt.ylabel(r'$F_{\nu}$ [mJy]', fontsize=12)
plt.legend(loc='upper left')
plt.show()


# nu_arr=np.logspace(log10(nus[0]), log10(nus[-1])+4, 100)
# fluxes_arr = np.zeros(len(nu_arr))
# for i in range(len(nu_arr)):
#     fluxes_arr[i]=F_metals_Hminus( Mpdot, Bps)
    #     I_nu, tau_metals, tau_Hminus = I_metals_Hminus_and_taus(r, nu, Mpdot, alpha)
    #     if, 1e17)
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
    r=0.06*au
    alpha=alpha_calculator(r, Mpdot, Bps)
    I_nu, tau_metals, tau_Hminus, tau_H2minus, tau_Heminus = I_metals_Hminus_and_taus(r, nu, Mpdot, alpha)
    tau_metals_max=tau_metals
    tau_Hminus_max=tau_Hminus
    tau_H2minus_max=tau_H2minus
    tau_Heminus_max=tau_Heminus
    return r, tau_metals_max, tau_Hminus_max, tau_H2minus_max, tau_Heminus_max

Mpdot=10**(-5.596172)*Mj/yr
Bps=10**(2.81155774) # G

# example with our frequencies
nus = [97.5E9, 145005402197.7, 343.5E9, 671E9] # [nu1, nu2] are the x-axis values , 671E9

array_nus = np.logspace(log10(nus[0]), log10(nus[-1]), 100)
tau_metals_array = np.zeros(len(array_nus))
tau_Hminus_array = np.zeros(len(array_nus))
tau_H2_minus_array = np.zeros(len(array_nus))
tau_He_minus_array = np.zeros(len(array_nus))
r=0
for i in range(len(array_nus)):
    r, tau_metals_array[i], tau_Hminus_array[i], tau_H2_minus_array[i], tau_He_minus_array[i] = tau_maximun_contribution_Hminus(array_nus[i], Mpdot, Bps)
    #print(r/au, tau_metals_array[i]+tau_Hminus_array[i]+tau_H2_minus_array[i])


plt.figure(figsize=(4.8,3.8), dpi=300)
#plt.plot(array_nus/1E9, tau_Hminus_array, label=r'$\rm H^{-}$', color='C0')
plt.plot(array_nus/1E9, tau_H2_minus_array, label=r'$\rm H_{2}^{-}$', color='C0')
plt.plot(array_nus/1E9, tau_metals_array, label='Metals', color='C1')
# plt.plot(array_nus/1E9, tau_He_minus_array, label=r'$\rm He^{-}$', color='C4')
plt.plot(array_nus/1E9, tau_Hminus_array+tau_metals_array+tau_H2_minus_array+tau_He_minus_array, label='Total', color='C2', linestyle='--')
plt.xscale('log')
plt.yscale('log')
plt.tick_params(axis='both', which='both', labelsize=12)
plt.xlabel(r'$\nu$ [GHz]', fontsize=14)
plt.ylabel(r'$\tau$', fontsize=14)
plt.legend(loc='best', fontsize=12)
plt.savefig('tau_Hminus_metals.pdf', bbox_inches='tight', dpi=300)
plt.show()