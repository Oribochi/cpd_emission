# This is the code with all the functions to calculate the flux and brightness temperature of the magnetic disk model
from units_astro import *
import numpy as np
from numpy import pi, sqrt, log10, exp
from dustyCPD import Sigma, Rin, Rout, dis, mu
from magneticCPD import H, alpha_calculator, T_z
from simple_ionization_fraction import simple_ionization_fraction_H_minus, total_ion_abundance, a1
from k_ff_metals import k_ff_metals
from k_ff_bf_Hminus_2021 import k_ff_Hminus, k_bf_Hminus
import matplotlib.pyplot as plt


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

def I_metals_Hminus(R, nu, Mpdot, alpha):
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
        tau_metals=np.trapezoid(kb*Temp_z_arr[:m+1]*ne_arr[:m+1]*n_ion_arr[:m+1]*k_ff_metals_arr[:m+1], z_arr[:m+1])
        tau_Hminus=np.trapezoid(kb*Temp_z_arr[:m+1]*ne_arr[:m+1]*n_H_arr[:m+1]*k_ff_bf_Hminus_arr[:m+1], z_arr[:m+1])  # We asumme only H- contributes to the opacity
        tau_arr[m]=tau_Hminus+tau_metals
        plank_arr[m]=plank(Temp_z, nu)

        # in the case tau is too big we return to the non stratified case
        if tau_arr[m]>1000:
            k=m
            break
        
    tau_tot=tau_arr[-1]

    # # print  the temp_ pank arr_ tau arr
    # print('Temp_z_arr', Temp_z_arr)
    # print('plank_arr', plank_arr)
    # print('tau_arr', tau_arr)
    if k==0:
        integrate=np.array([plank_arr[i]*exp(-tau_tot+tau_arr[i]) for i in range(N)])*10**26
        return np.trapezoid(integrate, tau_arr)
        
    # for the case optically thinner
    #print(tau_arr[0], tau_arr[int(N/2)], tau_arr[-1])
    #print(ne_arr[0]**2*h, ne_arr[int(N/2)]**2*h, ne_arr[-1]**2*h)
    #print(k_ff_arr[0], k_ff_arr[int(N/2)], k_ff_arr[-1])
    # for the case optically thicker tau is too big and only the last value is important
    else:
        l, p =find_1(tau_arr[:k])
        z1=(z_arr[l]+z_arr[p])/2
        # z1=3*h
        plank_val=plank(T_z(z1, R, Mpdot, alpha), nu)
        return plank_val*10**26
    
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

# print Inu at R=0.05 au


def F_metals_Hminus(nu, Mpdot, Bps):
    R_arr=np.logspace(log10(1.0001*Rin/au), log10(Rout/au), 100)*au
    F_ff_arr=np.zeros(len(R_arr))
    for i in range(len(R_arr)):
        alpha=alpha_calculator(R_arr[i], Mpdot, Bps)
        F_ff_arr[i]=I_metals_Hminus(R_arr[i], nu, Mpdot, alpha)*2*pi*R_arr[i]/dis**2
    return np.trapezoid(F_ff_arr, R_arr)  # to mJy




points=np.load('logP1.npy')
log_w=np.load('logW1.npy')
log_l=np.load('logZ1.npy')

Mpdot = 10**points[np.argmax(log_l)][0]*Mj/yr
Bps = 10**points[np.argmax(log_l)][1]

# Mpdot=10**(-5.381483683986507)*Mj/yr
# Bps=10**(2.7604387543852456)

# # First our data with the error bars
nus = [97.5E9, 145005402197.7, 343.5E9, 671E9] # [nu1, nu2] are the x-axis values , 671E9
Fnus = [12e-06, 21.4e-06, 0.000121, 0.000143] # [Fnu1, Fnu2] are the y-axis values in Jy , 0.000122
sFnus = [4.7e-06, 4.1e-06, 0.000013, 0.000115] # [sigma_Fnu1, sigma_Fnu2] are the y-axis errors in Jy , 0.000090

# To mJy
nu = np.array(nus)
Fnus = np.array(Fnus)*1E3
sFnus = np.array(sFnus)*1E3

# I_nu_try=I_metals_Hminus(0.05*au, nu[0], Mpdot, alpha_calculator(0.05*au, Mpdot, Bps))
# print('I_nu at R=0.05 au and nu =', nu[0]/1E9, 'GHz is', I_nu_try/au, 'mJy/sr/1au')

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


nu_arr=np.logspace(log10(nus[0]), log10(nus[-1])+4, 100)
fluxes_arr = np.zeros(len(nu_arr))
for i in range(len(nu_arr)):
    fluxes_arr[i]=F_metals_Hminus(nu_arr[i], Mpdot, Bps)


plt.errorbar(np.array(nus)/10**9, Fnus, yerr=sFnus, fmt='o', label='Data', color='blue')
plt.plot(np.array(nu_arr)/10**9, fluxes_arr, label='model', alpha=0.5, color='red')
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'$\nu$ [GHz]', fontsize=12)
plt.ylabel(r'$F_{\nu}$ [mJy]', fontsize=12)
plt.ylim(1e-6, 1e3)
plt.legend(loc='upper left')
plt.savefig('fluxes_metals_Hminus.png')
plt.show()