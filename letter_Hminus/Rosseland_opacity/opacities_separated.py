# this is a code to plot opacities in a range 
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
def opacity_H_minus(nu, n_tot, T):
    rho=n_tot*mu*mH
    f=simple_ionization_fraction_H_minus(T, rho)
    n_e=f*n_tot # the number density of electrons
    Y=a4*4/(a4*4+a1*1) # mass fraction of helium
    n_H, n_H2=np.array(molecular_fraction_H2(rho, T, Y))*n_tot*a1 # number density of H vs the H2 molecules
    k_ff_bf_Hminus_val=k_ff_Hminus(T, nu)+k_bf_Hminus(T, nu) # cm^2
    P_e=n_e*kb*T # electron pressure in erg cm^-3
    opacity_Hminus=P_e*n_H*k_ff_bf_Hminus_val/rho # cm^2 g^-1
    return opacity_Hminus

def opacity_H2_minus(nu, n_tot, T):
    rho=n_tot*mu*mH
    f=simple_ionization_fraction_H_minus(T, rho)
    n_e=f*n_tot # the number density of electrons
    Y=a4*4/(a4*4+a1*1) # mass fraction of helium
    n_H, n_H2=np.array(molecular_fraction_H2(rho, T, Y))*n_tot*a1 # number density of H vs the H2 molecules
    k_ff_H2_minus_val=k_ff_H2_minus(T, nu) # cm^2
    P_e=n_e*kb*T # electron pressure in erg cm^-3
    opacity_H2minus=P_e*n_H2*k_ff_H2_minus_val/rho # cm^2 g^-1
    return opacity_H2minus

def opacity_metals(nu, n_tot, T):
    rho=n_tot*mu*mH
    f=simple_ionization_fraction_H_minus(T, rho)
    n_e=f*n_tot # the number density of electrons
    n_ion=total_ion_abundance(n_e, T)*n_tot # the number density of ions
    k_ff_metals_val=k_ff_metals(T, nu) # cm^5 K^1/2
    P_e=n_e*kb*T # electron pressure in erg cm^-3
    opacity_metals=P_e*n_ion*k_ff_metals_val/rho # cm^2 g^-1
    return opacity_metals

# example for rho=1.0105429410021774e-08 (my maximun mass density in the disk model)
n_tot1=1.0105429410021774e-08/(mu*mH)
# and T=1000 K, 2000 K and 5000 K
# with nu from 1 to 1000 GHz
nu_array=np.logspace(9, 15, 100) # frequency array from 1 GHz to 1000000 GHz
opacity_Hminus_array=np.array([opacity_H_minus(nu, n_tot1, 1000) for nu in nu_array])
opacity_H2minus_array=np.array([opacity_H2_minus(nu, n_tot1, 1000) for nu in nu_array])
opacity_metals_array=np.array([opacity_metals(nu, n_tot1, 1000) for nu in nu_array])
# total opacity
opacity_tot_array=opacity_Hminus_array+opacity_H2minus_array+opacity_metals_array

plt.figure(figsize=(8,6))
plt.loglog(nu_array/1e9, opacity_Hminus_array, label='H$^-$ opacity')
plt.loglog(nu_array/1e9, opacity_H2minus_array, label='H$_2^-$ opacity')
plt.loglog(nu_array/1e9, opacity_metals_array, label='Metals opacity')
plt.loglog(nu_array/1e9, opacity_tot_array, label='Total opacity', linestyle='--')
plt.xlabel('Frequency (GHz)')
plt.ylabel('Opacity (cm$^2$ g$^{-1}$)')
plt.title('Opacity components at T=1000 K, rho=1.01e-8 g cm$^{-3}$')
plt.legend()
plt.ylim(1e-14, 1e-1)
plt.savefig("opacity_components_1000K.png", dpi=300)
plt.show()

# now the same plot but in units of 1/cm instead of GHz in the x-axis, from 1/lambda 
lambda_array=c/nu_array # in cm
inv_lambda_array=1/lambda_array # in 1/cm
plt.figure(figsize=(8,6))
plt.plot(inv_lambda_array, opacity_Hminus_array, label='H$^-$ opacity')
plt.plot(inv_lambda_array, opacity_H2minus_array, label='H$_2^-$ opacity')
plt.plot(inv_lambda_array, opacity_metals_array, label='Metals opacity')
plt.plot(inv_lambda_array, opacity_tot_array, label='Total opacity', linestyle='--')
plt.yscale('log')
plt.xscale('log')
plt.xlabel('1 / Wavelength (cm$^{-1}$)')
plt.ylabel('Opacity (cm$^2$ g$^{-1}$)') 
plt.title('Opacity components at T=1000 K, rho=1.01e-8 g cm$^{-3}$')
plt.legend()
plt.ylim(1e-14, 1e-1)
plt.savefig("opacity_components_1000K_inv_lambda.png", dpi=300)
plt.show()

#repeat for T=2000 K 
opacity_Hminus_array=np.array([opacity_H_minus(nu, n_tot1, 2000) for nu in nu_array])
opacity_H2minus_array=np.array([opacity_H2_minus(nu, n_tot1, 2000) for nu in nu_array])
opacity_metals_array=np.array([opacity_metals(nu, n_tot1, 2000) for nu in nu_array])
# total opacity
opacity_tot_array=opacity_Hminus_array+opacity_H2minus_array+opacity_metals_array
plt.figure(figsize=(8,6))
plt.loglog(nu_array/1e9, opacity_Hminus_array, label='H$^-$ opacity')
plt.loglog(nu_array/1e9, opacity_H2minus_array, label='H$_2^-$ opacity')
plt.loglog(nu_array/1e9, opacity_metals_array, label='Metals opacity')
plt.loglog(nu_array/1e9, opacity_tot_array, label='Total opacity', linestyle='--')
plt.xlabel('Frequency (GHz)')
plt.ylabel('Opacity (cm$^2$ g$^{-1}$)')
plt.title('Opacity components at T=2000 K, rho=1.01e-8 g cm$^{-3}$')
plt.legend()
plt.ylim(1e-8, 1e2)
plt.savefig("opacity_components_2000K.png", dpi=300)
plt.show()

# now the same plot but in units of 1/cm instead of GHz in the x-axis, from 1/lambda 
lambda_array=c/nu_array # in cm
inv_lambda_array=1/lambda_array # in 1/cm
plt.figure(figsize=(8,6))
plt.plot(inv_lambda_array, opacity_Hminus_array, label='H$^-$ opacity')
plt.plot(inv_lambda_array, opacity_H2minus_array, label='H$_2^-$ opacity')
plt.plot(inv_lambda_array, opacity_metals_array, label='Metals opacity')
plt.plot(inv_lambda_array, opacity_tot_array, label='Total opacity', linestyle='--')
plt.yscale('log')
plt.xscale('log')
plt.xlabel('1 / Wavelength (cm$^{-1}$)')
plt.ylabel('Opacity (cm$^2$ g$^{-1}$)') 
plt.title('Opacity components at T=2000 K, rho=1.01e-8 g cm$^{-3}$')
plt.legend()
plt.ylim(1e-8, 1e2)
plt.savefig("opacity_components_2000K_inv_lambda.png", dpi=300)
plt.show()

#repeat for T=5000 K
opacity_Hminus_array=np.array([opacity_H_minus(nu, n_tot1, 5000) for nu in nu_array])
opacity_H2minus_array=np.array([opacity_H2_minus(nu, n_tot1, 5000) for nu in nu_array])
opacity_metals_array=np.array([opacity_metals(nu, n_tot1, 5000) for nu in nu_array])
# total opacity
opacity_tot_array=opacity_Hminus_array+opacity_H2minus_array+opacity_metals_array
plt.figure(figsize=(8,6))
plt.loglog(nu_array/1e9, opacity_Hminus_array, label='H$^-$ opacity')
plt.loglog(nu_array/1e9, opacity_H2minus_array, label='H$_2^-$ opacity')
plt.loglog(nu_array/1e9, opacity_metals_array, label='Metals opacity')
plt.loglog(nu_array/1e9, opacity_tot_array, label='Total opacity', linestyle='--')
plt.xlabel('Frequency (GHz)')
plt.ylabel('Opacity (cm$^2$ g$^{-1}$)')
plt.title('Opacity components at T=5000 K, rho=1.01e-8 g cm$^{-3}$')
plt.legend()
plt.ylim(1e-3, 1e7)
plt.savefig("opacity_components_5000K.png", dpi=300)
plt.show()

# now the same plot but in units of 1/cm instead of GHz in the x-axis, from 1/lambda 
lambda_array=c/nu_array # in cm
inv_lambda_array=1/lambda_array # in 1/cm
plt.figure(figsize=(8,6))
plt.plot(inv_lambda_array, opacity_Hminus_array, label='H$^-$ opacity')
plt.plot(inv_lambda_array, opacity_H2minus_array, label='H$_2^-$ opacity')
plt.plot(inv_lambda_array, opacity_metals_array, label='Metals opacity')
plt.plot(inv_lambda_array, opacity_tot_array, label='Total opacity', linestyle='--')
plt.yscale('log')
plt.xscale('log')
plt.xlabel('1 / Wavelength (cm$^{-1}$)')
plt.ylabel('Opacity (cm$^2$ g$^{-1}$)') 
plt.title('Opacity components at T=5000 K, rho=1.01e-8 g cm$^{-3}$')
plt.legend()
plt.ylim(1e-3, 1e7)
plt.savefig("opacity_components_5000K_inv_lambda.png", dpi=300)
plt.show()

#repeat for T=3000 K
opacity_Hminus_array=np.array([opacity_H_minus(nu, n_tot1, 3000) for nu in nu_array])
opacity_H2minus_array=np.array([opacity_H2_minus(nu, n_tot1, 3000) for nu in nu_array])
opacity_metals_array=np.array([opacity_metals(nu, n_tot1, 3000) for nu in nu_array])
# total opacity
opacity_tot_array=opacity_Hminus_array+opacity_H2minus_array+opacity_metals_array
plt.figure(figsize=(8,6))
plt.loglog(nu_array/1e9, opacity_Hminus_array, label='H$^-$ opacity')
plt.loglog(nu_array/1e9, opacity_H2minus_array, label='H$_2^-$ opacity')
plt.loglog(nu_array/1e9, opacity_metals_array, label='Metals opacity')
plt.loglog(nu_array/1e9, opacity_tot_array, label='Total opacity', linestyle='--')
plt.xlabel('Frequency (GHz)')
plt.ylabel('Opacity (cm$^2$ g$^{-1}$)')
plt.title('Opacity components at T=3000 K, rho=1.01e-8 g cm$^{-3}$')
plt.legend()
plt.ylim(1e-8, 1e2)
plt.savefig("opacity_components_3000K.png", dpi=300)
plt.show()

# now the same plot but in units of 1/cm instead of GHz in the x-axis, from 1/lambda 
lambda_array=c/nu_array # in cm
inv_lambda_array=1/lambda_array # in 1/cm
plt.figure(figsize=(8,6))
plt.plot(inv_lambda_array, opacity_Hminus_array, label='H$^-$ opacity')
plt.plot(inv_lambda_array, opacity_H2minus_array, label='H$_2^-$ opacity')
plt.plot(inv_lambda_array, opacity_metals_array, label='Metals opacity')
plt.plot(inv_lambda_array, opacity_tot_array, label='Total opacity', linestyle='--')
plt.yscale('log')
plt.xscale('log')
plt.xlabel('1 / Wavelength (cm$^{-1}$)')
plt.ylabel('Opacity (cm$^2$ g$^{-1}$)') 
plt.title('Opacity components at T=3000 K, rho=1.01e-8 g cm$^{-3}$')
plt.legend()
plt.ylim(1e-8, 1e2)
plt.savefig("opacity_components_3000K_inv_lambda.png", dpi=300)
plt.show()


# Lambdas Im interested are in between 97 GHz and 671 GHz
print(1/(c/97e9), 1/(c/671e9))  # in 1/cm

#Finally at 1468 K
## example for rho=1.0394482157052386e-08 (my maximun mass density in the disk model)
n_tot2=1.0394482157052386e-08/(mu*mH)
opacity_Hminus_array=np.array([opacity_H_minus(nu, n_tot2, 1468) for nu in nu_array])
opacity_H2minus_array=np.array([opacity_H2_minus(nu, n_tot2, 1468) for nu in nu_array])
opacity_metals_array=np.array([opacity_metals(nu, n_tot2, 1468) for nu in nu_array])
# total opacity
opacity_tot_array=opacity_Hminus_array+opacity_H2minus_array+opacity_metals_array
plt.figure(figsize=(8,6))
plt.loglog(nu_array/1e9, opacity_Hminus_array, label='H$^-$ opacity')
plt.loglog(nu_array/1e9, opacity_H2minus_array, label='H$_2^-$ opacity')
plt.loglog(nu_array/1e9, opacity_metals_array, label='Metals opacity')
plt.loglog(nu_array/1e9, opacity_tot_array, label='Total opacity', linestyle='--')
plt.xlabel('Frequency (GHz)')
plt.ylabel('Opacity (cm$^2$ g$^{-1}$)')
plt.title('Opacity components at T=1468 K, rho=1.04e-8 g cm$^{-3}$')
plt.legend()
plt.ylim(1e-10, 1e1)
plt.savefig("opacity_components_1468K.png", dpi=300)
plt.show()