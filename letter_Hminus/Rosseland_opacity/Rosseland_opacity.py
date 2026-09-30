# This is the code with all the functions to calculate the flux and brightness temperature of the magnetic disk model
from units_astro import *
import numpy as np
from numpy import pi, sqrt, log10, exp
from dustyCPD import Sigma, Rin, Rout, dis, mu
from magneticCPD import H, alpha_calculator, T_z
from simple_ionization_fraction import simple_ionization_fraction_H_minus, total_ion_abundance, a1, a4
from k_ff_metals import k_ff_metals
from k_ff_bf_Hminus import k_ff_Hminus, k_bf_Hminus
from k_bf_H_2021 import k_bf_H
from k_ff_H2_2021_minus import k_ff_H2_minus
from H_H2_ratio_2017 import molecular_fraction_H2
from k_ff_He_minus import k_ff_He_minus
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

# defining the opacity in function of the frequency, total number density and temperature
def opacity_tot(nu, n_tot, T):
    rho=n_tot*mu*mH
    f=simple_ionization_fraction_H_minus(T, rho)
    n_e=f*n_tot # the number density of electrons
    n_ion=total_ion_abundance(n_e, T)*n_tot # the number density of ions
    Y=a4*4/(a4*4+a1*1) # mass fraction of helium approx 0.25
    n_H, n_H2=np.array(molecular_fraction_H2(rho, T, Y))*n_tot*a1 # number density of H vs the H2 molecules
    n_He=n_tot*a4 # we assume almost all He is neutral (less than 5000 K)
    k_ff_metals_val=k_ff_metals(T, nu) # cm^5 K^1/2
    k_ff_bf_Hminus_val=k_ff_Hminus(T, nu)+k_bf_Hminus(T, nu) # cm^2
    k_ff_H2_minus_val=k_ff_H2_minus(T, nu) # cm^2
    k_ff_He_minus_val=k_ff_He_minus(T, nu) # cm^2
    P_e=n_e*kb*T # electron pressure in erg cm^-3
    opacity_metals=P_e*n_ion*k_ff_metals_val/rho # cm^2 g^-1
    opacity_Hminus=P_e*n_H*k_ff_bf_Hminus_val/rho # cm^2 g^-1
    opacity_H2minus=P_e*n_H2*k_ff_H2_minus_val/rho # cm^2 g^-1
    opacity_He_minus=P_e*n_He*k_ff_He_minus_val/rho # cm^2 g^-1
    return opacity_metals+opacity_Hminus+opacity_H2minus+opacity_He_minus

# Now the rosseland mean opacity is given by the integral in frequency of the opacity weighted by 
# the derivative of the plank function divided by the integral in frequency of the derivative of the plank function

def d_plank_dT(T, nu):
    x=h*nu/(kb*T)
    if x<1e-3: # low frequency limit
        return 2*nu**2/c**2*kb
    elif x>1e1: # high frequency limit
        return 2*h**2*nu**4/(c**2*kb*T**2)*exp(-x)
    else:
        return 2*h**2*nu**4/(c**2*kb*T**2)*exp(x)/(exp(x)-1)**2
    
def rosseland_mean_opacity(n_tot, T):
    nu_array=np.logspace(9, 18, 1000) # frequency array from 1 GHz to 1e18 Hz
    dB_dT_array=[d_plank_dT(T, nu) for nu in nu_array]
    dB_dT_array=np.array(dB_dT_array)
    kappa_array=[opacity_tot(nu, n_tot, T) for nu in nu_array]
    kappa_array=np.array(kappa_array)
    integrand=dB_dT_array/kappa_array
    integral1=np.trapz(integrand, nu_array)
    integral2=np.trapz(dB_dT_array, nu_array)
    return integral2/integral1

# example for rho=1.0105429410021774e-08 (my maximun mass density in the disk model)
n_tot1=1.0105429410021774e-08/(mu*mH)
rosseland1=rosseland_mean_opacity(n_tot1, 1000)
rosseland2=rosseland_mean_opacity(n_tot1, 2000)
rosseland3=rosseland_mean_opacity(n_tot1, 3000)
rosseland4=rosseland_mean_opacity(n_tot1, 4000)
rosseland5=rosseland_mean_opacity(n_tot1, 5000)
print(rosseland1, rosseland2, rosseland3, rosseland4, rosseland5)
# Now we can calculate the rosseland mean opacity from 10-5000 K and from 0.1 to 10 g/cm^3

# now in a grid from rho=1e-12 to 1e-8 g/cm^3 and from T=1000 to 5000 K of 100x100 points and a grid plot 
# of the rosseland mean opacity (save fig and data in a file)
rho_array=np.logspace(-12, -8, 100) # g/cm^3
T_array=np.logspace(3, 3.7, 100) # K
kappa_rosseland_array=np.zeros((100, 100))
for i in range(100):
    for j in range(100):
        n_tot=rho_array[i]/(mu*mH)
        T=T_array[j]
        kappa_rosseland_array[i, j]=rosseland_mean_opacity(n_tot, T)
        #print(f"rho={rho_array[i]:.2e}, T={T:.2f}, kappa_rosseland={kappa_rosseland_array[i, j]:.2e}")
# save the data in a file
np.savetxt("rosseland_opacity_grid.txt", kappa_rosseland_array)
# plot the data
R, T = np.meshgrid(rho_array, T_array)
plt.figure(figsize=(8,6))
cp = plt.contourf(R, T, kappa_rosseland_array.T, levels=50, norm=matplotlib.colors.LogNorm())
plt.colorbar(cp, label='Rosseland Mean Opacity (cm$^2$ g$^{-1}$)')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Mass Density (g cm$^{-3}$)')
plt.ylabel('Temperature (K)')
plt.title('Rosseland Mean Opacity')
plt.savefig("rosseland_opacity_grid.png", dpi=300)
plt.show()
